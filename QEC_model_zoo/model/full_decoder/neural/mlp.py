import copy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn, Tensor
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.croissant_data_loader import CroissantDataLoader
from model.full_decoder.base_classifier import BaseClassifier


class MLP(nn.Module, BaseClassifier):
    def __init__(self, params: Dict[str, Any], args: Any):
        nn.Module.__init__(self)
        BaseClassifier.__init__(self, params, args)

        self._set_seed(args.seed)

        input_size = args.num_features
        num_classes = args.num_classes

        self.model_name = "MLP"

        hidden_size = params.get("hidden_size", 512)
        num_layers = params.get("num_layers", 2)
        activation = params.get("activation", "relu")
        dropout = params.get("dropout", 0.0)

        activation_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }

        if activation not in activation_map:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        in_features = input_size

        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(activation_map[activation]())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_features = hidden_size

        layers.append(nn.Linear(in_features, num_classes))

        self.layers = nn.Sequential(*layers)

    def _set_seed(self, seed: int) -> None:
        """
        Set random seed of the model.

        :param seed: Random seed.
        """
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the neural network.

        :param x: input tensor
        :return: output tensor
        """
        return self.layers(x)

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None,
            verbose: bool = True,
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        X_train = torch.as_tensor(
            X_train,
            dtype=torch.float32,
            device=device,
        )
        y_train = torch.as_tensor(
            y_train,
            dtype=torch.long,
            device=device,
        )

        has_val = X_val is not None and y_val is not None

        if has_val:
            X_val = torch.as_tensor(
                X_val,
                dtype=torch.float32,
                device=device,
            )
            y_val = torch.as_tensor(
                y_val,
                dtype=torch.long,
                device=device,
            )

        lr = self.params.get(
            "learning_rate",
            self.params.get("lr", 1e-3),
        )
        weight_decay = self.params.get("weight_decay", 1e-5)
        epochs = int(self.params.get("epochs", 50))
        patience = int(self.params.get("patience", 10))
        max_grad_norm = self.params.get("max_grad_norm", None)

        optimizer_type = self.params.get(
            "optimizer",
            "Adam",
        ).lower()

        if optimizer_type == "adam":
            optimizer = Adam(
                self.parameters(),
                lr=lr,
                eps=1e-8,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=lr,
                eps=1e-8,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(
                f"Unknown optimizer type: {optimizer_type} for MLP"
            )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )

        criterion_type = (
            self.params
            .get("loss_function", "cross_entropy")
            .lower()
            .replace(" ", "_")
        )

        if criterion_type == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                f"Unknown loss function: {criterion_type} for MLP"
            )

        best_val = -float("inf")
        best_state = None
        counter = 0

        if verbose:
            print("=" * 80)
            print(f"Training {self.model_name}")
            print(f"Device:          {device}")
            print(f"Train samples:   {len(X_train)}")
            print(
                f"Val samples:     {len(X_val) if has_val else 'None'}"
            )
            print(f"Epochs:          {epochs}")
            print(f"Patience:        {patience}")
            print(f"Optimizer:       {optimizer_type}")
            print(f"Learning rate:   {lr:.3e}")
            print(f"Weight decay:    {weight_decay:.3e}")
            print(f"Loss:            {criterion_type}")

            if max_grad_norm is not None:
                print(f"Max grad norm:   {max_grad_norm}")

            print("=" * 80)

        for epoch in range(epochs):
            # ---------------------------------------------------------
            # Training
            # ---------------------------------------------------------
            self.train()

            optimizer.zero_grad(set_to_none=True)

            logits = self(X_train)
            train_loss = criterion(logits, y_train)

            train_loss.backward()

            if max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    max_grad_norm,
                )
            else:
                grad_norm = None

            optimizer.step()

            # ---------------------------------------------------------
            # Evaluation
            # ---------------------------------------------------------
            self.eval()

            with torch.no_grad():
                if has_val:
                    eval_logits = self(X_val)
                    eval_loss = criterion(eval_logits, y_val)

                    eval_preds = torch.argmax(
                        eval_logits,
                        dim=1,
                    )

                    eval_preds_np = eval_preds.cpu().numpy()

                    val_score = balanced_accuracy_score(
                        y_val.detach().cpu().numpy(),
                        eval_preds_np,
                    )
                else:
                    eval_logits = self(X_train)
                    eval_loss = criterion(
                        eval_logits,
                        y_train,
                    )

                    eval_preds = torch.argmax(
                        eval_logits,
                        dim=1,
                    )

                    eval_preds_np = eval_preds.cpu().numpy()

                    val_score = balanced_accuracy_score(
                        y_train.detach().cpu().numpy(),
                        eval_preds_np,
                    )

            current_lr = optimizer.param_groups[0]["lr"]

            # ---------------------------------------------------------
            # Learning-rate scheduler
            # ---------------------------------------------------------
            scheduler.step(val_score)

            new_lr = optimizer.param_groups[0]["lr"]
            lr_changed = new_lr != current_lr

            # ---------------------------------------------------------
            # Best model / early stopping
            # ---------------------------------------------------------
            improved = val_score > best_val

            if improved:
                best_val = val_score
                best_state = copy.deepcopy(self.state_dict())
                counter = 0
            else:
                counter += 1

            # ---------------------------------------------------------
            # Logging
            # ---------------------------------------------------------
            if verbose:
                split_name = "val" if has_val else "train"

                message = (
                    f"Epoch {epoch + 1:03d}/{epochs:03d} | "
                    f"train_loss={train_loss.item():.6f} | "
                    f"{split_name}_loss={eval_loss.item():.6f} | "
                    f"{split_name}_bal_acc={val_score:.4f} | "
                    f"lr={new_lr:.3e}"
                )

                if grad_norm is not None:
                    message += (
                        f" | grad_norm={float(grad_norm):.4f}"
                    )

                message += (
                    f" | patience={counter}/{patience}"
                )

                if improved:
                    message += " | BEST"

                print(message)

                if lr_changed:
                    print(
                        f"  ↳ Learning rate reduced: "
                        f"{current_lr:.3e} -> {new_lr:.3e}"
                    )

            if counter >= patience:
                if verbose:
                    print(
                        f"\nEarly stopping at epoch {epoch + 1}. "
                        f"Best balanced accuracy: {best_val:.4f}"
                    )
                break

        # -------------------------------------------------------------
        # Restore best model
        # -------------------------------------------------------------
        if best_state is not None:
            self.load_state_dict(best_state, strict=True)

        self.is_trained = True

        if verbose:
            print("=" * 80)
            print("Training complete")
            print(f"Best balanced accuracy: {best_val:.4f}")
            print("=" * 80)

    def predict(self, X: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device

        X = torch.tensor(X, dtype=torch.float32).to(device)

        self.eval()
        with torch.no_grad():
            logits = self(X)
            preds = torch.argmax(logits, dim=1)

        return preds.cpu().numpy()

    def save_model(self, path: Path) -> Path:
        torch.save(self.state_dict(), path)
        return path

    @classmethod
    def load_model(cls, path: Path, params: Dict[str, Any], model_args: Any) -> "MLP":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = cls(params=params, args=model_args)
        state_dict = torch.load(path, map_location=device)

        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        model.is_trained = True
        return model
