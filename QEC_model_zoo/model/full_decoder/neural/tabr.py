from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from model.full_decoder.base_classifier import BaseClassifier
from model.full_decoder.checkpoint_utils import load_torch_checkpoint


class _Backbone(nn.Module):
    """
    Backbone network for TabR classifier.
    """
    def __init__(self, d_in, d_hid=256, d_emb=256, dropout=0.1):
        """
        Initialize the backbone network.

        :param d_in: input dimension.
        :param d_hid: hidden dimension.
        :param d_emb: embedding dimension.
        :param dropout: dropout rate.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_emb)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the backbone network.

        :param x: input tensor.
        :return: output tensor.
        """
        return self.net(x)

class TabRClassifier(BaseClassifier, nn.Module):
    """
    MLP embeddings + L2 kNN retrieval on GPU using PyTorch only.
    Scikit-like API: fit, predict_proba, predict.
    """
    def __init__(self, params, args):
        nn.Module.__init__(self)
        super().__init__(params, args)
        self.device = torch.device("cuda" if args.use_gpu else "cpu")
        self.n_features = args.num_features
        self.n_classes = args.num_classes

        # hparams
        self.d_hid   = int(params.get("hidden_dim", 256))
        self.d_emb   = int(params.get("emb_dim", 256))
        self.k       = int(params.get("k", 32))
        self.tau     = float(params.get("tau", 0.07))
        self.epochs  = int(params.get("epochs", getattr(args, "epochs", 100)))
        self.bs      = int(params.get("batch_size", getattr(args, "batch_size", 256)))
        self.pat     = int(params.get("patience", 15))
        self.lr      = float(params.get("learning_rate", 1e-3))
        self.dropout = float(params.get("dropout", 0.1))
        self.knn_bs  = int(params.get("knn_batch", 4096))  # batch size for kNN queries

        self.backbone = _Backbone(self.n_features, self.d_hid, self.d_emb, self.dropout).to(self.device)
        self.head = nn.Linear(self.d_emb * 2, self.n_classes).to(self.device)  # concat(q, ctx)

        # train memory
        self.train_emb = None   # (N_tr, d_emb) on GPU
        self.y_train = None

    def _normalize(self, e):
        return e / (e.norm(dim=1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def _embed(self, X):
        self.backbone.eval()
        Xt = torch.from_numpy(X).float().to(self.device)
        E = []
        for i in range(0, Xt.size(0), self.knn_bs):
            e = self.backbone(Xt[i:i+self.knn_bs])
            E.append(self._normalize(e))
        return torch.cat(E, dim=0)  # (N, d)

    @torch.no_grad()
    def _search_topk(self, Q, Kmat, k):
        """
        Q: (B,d) queries on GPU, L2 to Kmat: (N,d) on GPU. Returns (dist, idx) each (B,k)
        Uses chunking to limit memory if B*N is large.
        """
        # L2(q,k)^2 = ||q||^2 + ||k||^2 - 2 q·k
        q2 = (Q**2).sum(dim=1, keepdim=True)        # (B,1)
        k2 = (Kmat**2).sum(dim=1, keepdim=True).T   # (1,N)
        # compute in chunks over N if needed
        B, N = Q.size(0), Kmat.size(0)
        max_cols = max(1, int(1e7 // max(1, B)))    # heuristic to cap memory
        best_d = torch.full((B, k), float("inf"), device=Q.device)
        best_i = torch.full((B, k), -1, device=Q.device, dtype=torch.long)

        for start in range(0, N, max_cols):
            end = min(N, start + max_cols)
            KK = Kmat[start:end]                    # (Nc,d)
            d = q2 + k2[:, start:end] - 2.0 * (Q @ KK.T)  # (B,Nc)
            # topk smallest distances in this block
            d_blk, i_blk = torch.topk(d, k=min(k, d.size(1)), dim=1, largest=False)
            i_blk = i_blk + start
            # merge with global topk
            merged_d = torch.cat([best_d, d_blk], dim=1)
            merged_i = torch.cat([best_i, i_blk], dim=1)
            keep_d, keep_idx = torch.topk(merged_d, k=k, dim=1, largest=False)
            gather_rows = torch.arange(B, device=Q.device).unsqueeze(1).expand_as(keep_idx)
            keep_i = merged_i[gather_rows, keep_idx]
            best_d, best_i = keep_d, keep_i

        return best_d, best_i

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None,
            verbose: bool = True,
    ) -> None:
        Xtr = torch.from_numpy(X_train).float().to(self.device)
        ytr = torch.from_numpy(y_train).long().to(self.device)

        train_loader = DataLoader(
            TensorDataset(Xtr, ytr),
            batch_size=self.bs,
            shuffle=True,
        )

        opt = torch.optim.Adam(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=self.lr,
        )

        ce = nn.CrossEntropyLoss()

        best = float("inf")
        wait = 0
        best_state = None
        best_epoch = None

        if verbose:
            print("=" * 90)
            print("Training TabR")
            print(f"Device:           {self.device}")
            print(f"Train samples:    {len(X_train)}")
            print(f"Val samples:      {len(X_val) if X_val is not None else 'None'}")
            print(f"Epochs:           {self.epochs}")
            print(f"Batch size:       {self.bs}")
            print(f"Patience:         {self.pat}")
            print(f"Learning rate:    {self.lr:.3e}")
            print("Loss:             cross_entropy")
            print("=" * 90)
            print("Phase 1: warm-up without retrieval context")
            print("-" * 90)

        # -------------------------------------------------------------
        # Warm-up without retrieval context
        # -------------------------------------------------------------
        for epoch in range(self.epochs):
            self.backbone.train()
            self.head.train()

            run = 0.0

            for xb, yb in train_loader:
                opt.zero_grad()

                q = self._normalize(self.backbone(xb))
                ctx = torch.zeros_like(q)

                logits = self.head(
                    torch.cat([q, ctx], dim=1)
                )

                loss = ce(logits, yb)

                loss.backward()
                opt.step()

                run += loss.item()

            cur = run / max(1, len(train_loader))

            # ---------------------------------------------------------
            # Early stopping on training loss
            # ---------------------------------------------------------
            improved = cur < best

            if improved:
                best = cur
                wait = 0
                best_epoch = epoch + 1

                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.state_dict().items()
                }
            else:
                wait += 1

            # ---------------------------------------------------------
            # Logging
            # ---------------------------------------------------------
            if verbose:
                message = (
                    f"Epoch {epoch + 1:03d}/{self.epochs:03d} | "
                    f"train_loss={cur:.6f} | "
                    f"best_train_loss={best:.6f} | "
                    f"lr={opt.param_groups[0]['lr']:.3e} | "
                    f"patience={wait}/{self.pat}"
                )

                if improved:
                    message += " | BEST"

                print(message)

            # ---------------------------------------------------------
            # Early stopping
            # ---------------------------------------------------------
            if wait >= self.pat:
                if verbose:
                    print()
                    print(
                        f"Early stopping warm-up at epoch {epoch + 1}. "
                        f"Best epoch: {best_epoch}, "
                        f"best train loss: {best:.6f}"
                    )

                break

        # -------------------------------------------------------------
        # Restore best warm-up checkpoint
        # -------------------------------------------------------------
        if best_state:
            self.load_state_dict(best_state)

        if verbose:
            print("-" * 90)
            print("Warm-up complete")
            print(f"Best epoch:       {best_epoch}")
            print(f"Best train loss:  {best:.6f}")
            print()
            print("Phase 2: building retrieval memory")

        # -------------------------------------------------------------
        # Build retrieval memory on train
        # -------------------------------------------------------------
        with torch.no_grad():
            self.train_emb = self._embed(X_train).to(self.device)
            self.y_train = torch.from_numpy(y_train).long().to(self.device)

        if verbose:
            print(f"Retrieval samples:    {len(self.train_emb)}")
            print(f"Embedding dimension:  {self.train_emb.shape[1]}")
            print("Retrieval memory built")
            print()
            print("Phase 3: fine-tuning with retrieval")
            print("-" * 90)

        # -------------------------------------------------------------
        # Short fine-tune with retrieval
        # -------------------------------------------------------------
        self._finetune_with_retrieval(Xtr, ytr)

        if verbose:
            print("Retrieval fine-tuning complete")
            print("=" * 90)
            print("Training complete")
            print("=" * 90)

        super().fit(X_train, y_train, X_val, y_val)  # Simply sets is_trained to true

    def _retrieval_ctx(self, Q):
        # Q: (B,d) normalized
        D, I = self._search_topk(Q, self.train_emb, k=min(self.k, self.train_emb.size(0)))
        W = torch.softmax(-D / max(self.tau, 1e-6), dim=1)          # (B,k)
        neigh = self.train_emb[I]                                    # (B,k,d)
        ctx = (W.unsqueeze(-1) * neigh).sum(dim=1)                   # (B,d)
        return ctx

    def _finetune_with_retrieval(self, Xtr, ytr):
        loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=self.bs, shuffle=True)
        opt = torch.optim.Adam(list(self.backbone.parameters()) + list(self.head.parameters()), lr=self.lr*0.5)
        ce = nn.CrossEntropyLoss()
        self.backbone.train(); self.head.train()
        steps = max(3, self.pat // 2)
        for _ in range(steps):
            for xb, yb in loader:
                opt.zero_grad()
                q = self._normalize(self.backbone(xb))
                ctx = self._retrieval_ctx(q)
                logits = self.head(torch.cat([q, ctx], dim=1))
                loss = ce(logits, yb)
                loss.backward(); opt.step()

    @torch.no_grad()
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for X_test.

        :param X_test: Test data feature matrix
        :return: Class probabilities
        """
        self.backbone.eval(); self.head.eval()
        probs_all = []
        Xt = torch.from_numpy(X_test).float().to(self.device)
        for i in range(0, Xt.size(0), self.knn_bs):
            q = self._normalize(self.backbone(Xt[i:i+self.knn_bs]))
            ctx = self._retrieval_ctx(q)
            logits = self.head(torch.cat([q, ctx], dim=1))
            probs = torch.softmax(logits, dim=1)
            probs_all.append(probs.detach().cpu())
        return torch.cat(probs_all, dim=0).numpy()

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save_model(self, path: Path) -> Path:
        checkpoint = {
            "state_dict": self.state_dict(),
            "train_emb": self.train_emb.detach().cpu() if self.train_emb is not None else None,
            "y_train": self.y_train.detach().cpu() if self.y_train is not None else None,
        }

        torch.save(checkpoint, path)

        return path

    @classmethod
    def load_model(
            cls,
            path: Path,
            params: Dict[str, Any],
            model_args: Any
    ) -> "TabRClassifier":

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = cls(params=params, args=model_args)

        # Supports .pt, .pt.gz, .pt.xz and multipart .pt.xz
        checkpoint = load_torch_checkpoint(
            path,
            map_location=device
        )

        model.load_state_dict(checkpoint["state_dict"], strict=True)

        model.train_emb = checkpoint["train_emb"]
        if model.train_emb is not None:
            model.train_emb = model.train_emb.to(device)

        model.y_train = checkpoint["y_train"]
        if model.y_train is not None:
            model.y_train = model.y_train.to(device)

        model.to(device)
        model.backbone.to(device)
        model.head.to(device)

        model.eval()
        model.backbone.eval()
        model.head.eval()

        model.is_trained = True

        return model
