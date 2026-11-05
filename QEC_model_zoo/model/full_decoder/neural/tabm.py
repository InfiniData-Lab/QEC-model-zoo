from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from tabm import TabM
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model.full_decoder.base_classifier import BaseClassifier


class TabMClassifier(BaseClassifier, nn.Module):
    def __init__(self, params, args):
        nn.Module.__init__(self)
        super().__init__(params, args)
        self.device = torch.device("cuda" if args.use_gpu else "cpu")
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        self.k = int(params.get("k", 32))
        self.epochs = int(params.get("epochs", getattr(args, "epochs", 100)))
        self.batch_size = int(params.get("batch_size", getattr(args, "batch_size", 256)))
        self.patience = int(params.get("patience", 20))
        self.batch_size_inference = params.get("batch_size_inference", None)
        d_out = self.num_classes if self.num_classes > 2 else 1
        n_blocks = int(params.get("n_layers", 3))
        d_block = int(params.get("hidden_size", 512))
        self.net = TabM.make(n_num_features=self.num_features, d_out=d_out, k=self.k, n_blocks=n_blocks, d_block=d_block).to(self.device)
        self.is_multiclass = self.num_classes > 2

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
            batch_size=self.batch_size,
            shuffle=True,
        )

        crit_ce = nn.CrossEntropyLoss()
        crit_bce = nn.BCEWithLogitsLoss()

        lr = self.params.get("learning_rate", 1e-3)
        weight_decay = self.params.get("weight_decay", 0.0)

        opt = torch.optim.Adam(
            self.net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        has_val = X_val is not None and y_val is not None

        if verbose:
            print("=" * 90)
            print("Training TabM")
            print(f"Device:           {self.device}")
            print(f"Train samples:    {len(X_train)}")
            print(f"Val samples:      {len(X_val) if has_val else 'None'}")
            print(f"Features:         {self.num_features}")
            print(f"Classes:          {self.num_classes}")
            print(f"Ensemble size k:  {self.k}")
            print(f"Epochs:           {self.epochs}")
            print(f"Batch size:       {self.batch_size}")
            print(f"Patience:         {self.patience}")
            print(f"Learning rate:    {lr:.3e}")
            print(f"Weight decay:     {weight_decay:.3e}")
            print(
                "Loss:             "
                + ("cross_entropy" if self.is_multiclass else "binary_cross_entropy")
            )
            print("=" * 90)

        best_loss = float("inf")
        best_state = None
        best_epoch = None
        wait = 0

        for epoch in range(self.epochs):
            # ---------------------------------------------------------
            # Training
            # ---------------------------------------------------------
            self.net.train()

            running_loss = 0.0
            num_batches = 0

            for xb, yb in train_loader:
                opt.zero_grad(set_to_none=True)

                yk = self.net(xb)

                if self.is_multiclass:
                    B, K, C = yk.shape

                    loss = nn.functional.cross_entropy(
                        yk.reshape(B * K, C),
                        yb[:, None].expand(B, K).reshape(-1),
                    )
                else:
                    logits = yk.squeeze(-1)  # (B, K)

                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logits,
                        yb.float()[:, None].expand_as(logits),
                    )

                loss.backward()
                opt.step()

                running_loss += loss.item()
                num_batches += 1

            train_loss = running_loss / max(1, num_batches)

            # ---------------------------------------------------------
            # Evaluation
            # ---------------------------------------------------------
            if has_val:
                val_loss, val_bal_acc = self._evaluate(
                    X_val,
                    y_val,
                )

                monitored_loss = val_loss
            else:
                _, train_bal_acc = self._evaluate(
                    X_train,
                    y_train,
                )

                val_loss, val_bal_acc = self._evaluate(X_val, y_val)
                monitored_loss = val_loss

            # ---------------------------------------------------------
            # Early stopping on BALANCED ACCURACY
            # ---------------------------------------------------------
            improved = monitored_loss < best_loss

            if improved:
                best_loss = monitored_loss
                best_epoch = epoch + 1
                wait = 0

                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.net.state_dict().items()
                }
            else:
                wait += 1

                # ---------------------------------------------------------
                # Logging
                # ---------------------------------------------------------
                if verbose:
                    current_lr = opt.param_groups[0]["lr"]

                    if has_val:
                        message = (
                            f"Epoch {epoch + 1:03d}/{self.epochs:03d} | "
                            f"train_loss={train_loss:.6f} | "
                            f"val_loss={val_loss:.6f} | "
                            f"val_bal_acc={val_bal_acc:.4f} | "
                            f"lr={current_lr:.3e} | "
                            f"patience={wait}/{self.patience}"
                        )
                    else:
                        message = (
                            f"Epoch {epoch + 1:03d}/{self.epochs:03d} | "
                            f"train_loss={train_loss:.6f} | "
                            f"train_bal_acc={val_bal_acc:.4f} | "
                            f"lr={current_lr:.3e} | "
                            f"patience={wait}/{self.patience}"
                        )

                    if improved:
                        message += " | BEST"

                    print(message)

                # ---------------------------------------------------------
                # Early stop
                # ---------------------------------------------------------
                if wait >= self.patience:
                    if verbose:
                        print()
                        print(
                            f"Early stopping at epoch {epoch + 1}. "
                            f"Best epoch: {best_epoch}, "
                            f"best loss: {best_loss:.4f}"
                        )
                    break

            # ---------------------------------------------------------
            # Logging
            # ---------------------------------------------------------
            if verbose:
                current_lr = opt.param_groups[0]["lr"]

                if has_val:
                    message = (
                        f"Epoch {epoch + 1:03d}/{self.epochs:03d} | "
                        f"train_loss={train_loss:.6f} | "
                        f"val_loss={val_loss:.6f} | "
                        f"val_bal_acc={val_bal_acc:.4f} | "
                        f"lr={current_lr:.3e} | "
                        f"patience={wait}/{self.patience}"
                    )
                else:
                    message = (
                        f"Epoch {epoch + 1:03d}/{self.epochs:03d} | "
                        f"train_loss={train_loss:.6f} | "
                        f"train_bal_acc={val_bal_acc:.4f} | "
                        f"lr={current_lr:.3e} | "
                        f"patience={wait}/{self.patience}"
                    )

                if improved:
                    message += " | BEST"

                print(message)

            # ---------------------------------------------------------
            # Stop
            # ---------------------------------------------------------
            if wait >= self.patience:
                if verbose:
                    print()
                    print(
                        f"Early stopping at epoch {epoch + 1}. "
                        f"Best epoch: {best_epoch}, "
                        f"best {'val' if has_val else 'train'} loss: "
                        f"{best_loss:.6f}"
                    )
                break

        # -------------------------------------------------------------
        # Restore best model
        # -------------------------------------------------------------
        if best_state is not None:
            self.net.load_state_dict(best_state)

        super().fit(X_train, y_train, X_val, y_val)  # Simply sets is_trained to true

        if verbose:
            print("=" * 90)
            print("Training complete")
            print(f"Best epoch:       {best_epoch}")
            print(f"Best loss:        {best_loss:.6f}")

            if has_val:
                final_loss, final_bal_acc = self._evaluate(
                    X_val,
                    y_val,
                )
                print(f"Restored val loss:     {final_loss:.6f}")
                print(f"Restored val bal acc:  {final_bal_acc:.4f}")

            print("=" * 90)

    @torch.no_grad()
    def _evaluate(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> tuple[float, float]:
        """
        Evaluate TabM using both ensemble-member loss and the final
        ensemble prediction.

        Returns:
            loss:
                Mean loss across the k TabM ensemble members.

            balanced_accuracy:
                Balanced accuracy after averaging the k ensemble
                predictions.
        """
        self.net.eval()

        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).long().to(self.device)

        yk = self.net(X_tensor)

        if self.is_multiclass:
            criterion = nn.CrossEntropyLoss()

            loss = sum(
                criterion(yk[:, i, :], y_tensor)
                for i in range(self.k)
            ) / self.k

            # Average ensemble probabilities, then classify
            probs = yk.softmax(dim=-1).mean(dim=1)
            preds = probs.argmax(dim=1)

        else:
            criterion = nn.BCEWithLogitsLoss()
            y_float = y_tensor.float()

            loss = sum(
                criterion(yk[:, i, 0], y_float)
                for i in range(self.k)
            ) / self.k

            # Average ensemble probabilities
            p1 = torch.sigmoid(yk).mean(dim=1).squeeze(-1)
            preds = (p1 >= 0.5).long()

        balanced_acc = balanced_accuracy_score(
            y,
            preds.cpu().numpy(),
        )

        return loss.item(), balanced_acc

    @torch.no_grad()
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities on test set.

        :param X_test: Test data feature matrix
        :return: Test predictions
        """
        self.net.eval()
        Xt = torch.from_numpy(X_test).float().to(self.device)
        yk = self.net(Xt)
        if self.is_multiclass:
            probs = yk.softmax(dim=-1).mean(dim=1)                 # (N, C)
            return probs.cpu().numpy()
        p1 = torch.sigmoid(yk).mean(dim=1).squeeze(-1)            # (N,)
        p0 = 1.0 - p1
        probs = torch.stack([p0, p1], dim=1)                       # (N, 2)
        return probs.cpu().numpy()

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save_model(self, path: Path) -> Path:
        torch.save(self.state_dict(), path)

        return path

    @classmethod
    def load_model(cls, path: Path, params: Dict[str, Any], model_args: Any) -> "TabMClassifier":
        device = torch.device("cuda")

        model = cls(params=params, args=model_args)

        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict, strict=True)

        model.to(device)
        model.net.to(device)
        model.eval()
        model.net.eval()

        model.is_trained = True

        return model
