from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from data.croissant_data_loader import CroissantDataLoader
from model.full_decoder.base_classifier import BaseClassifier


class _RawShotDataset(Dataset):
    """
    Internal PyTorch dataset for variable-length raw shot sequences.

    Each X[i] is expected to have shape:
        (n_rounds_i, n_features)

    Labels are optional so the same dataset can be used for training and
    inference.
    """

    def __init__(self, X: Iterable[np.ndarray], y: np.ndarray | None = None):
        self.X = [torch.as_tensor(x, dtype=torch.float32) for x in X]
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)

        if self.y is not None and len(self.X) != len(self.y):
            raise ValueError("X and y must contain the same number of shots.")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class BaseRawSequenceClassifier(BaseClassifier, ABC):
    """
    Base class for models operating on raw, variable-length shot sequences.

    The supplied data loader is expected to expose:
        get_train(raw=True)
        get_val(raw=True)
        get_test(raw=True)

    Subclasses implement only architecture-specific behavior via:
        _build_model(n_features)
        _forward_batch(x, lengths, padding_mask)

    Checkpointing and early stopping are handled here so recurrent and
    Transformer-based models can share the same training behavior.
    """

    params: Dict[str, Any]
    args: Any
    model: torch.nn.Module | None
    model_name: str
    is_trained: bool

    def __init__(
        self,
        params: Dict[str, Any],
        args: Any,
    ):
        super().__init__(params=params, args=args)

        self.model = None

        self.device = self._resolve_device()
        self.history: Dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        self.n_features: int | None = None
        self.checkpoint_path_: Path | None = None
        self.best_epoch_: int | None = None
        self.best_val_loss_: float | None = None

    # ------------------------------------------------------------------
    # Architecture-specific API
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_model(self, n_features: int) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def _forward_batch(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public training / inference API
    # ------------------------------------------------------------------

    def fit(self, dataloader: CroissantDataLoader) -> None:
        """
        Fit on the raw training split from the supplied dataloader.

        Relevant params
        ---------------
        epochs : int, default 50
        gradient_clip : float | None, default 1.0
        verbose : 0/1/2 or bool, default 1

        early_stopping : bool, default False
        patience : int, default 10
        min_delta : float, default 0.0

        checkpoint : dict, optional
            {
                "enabled": True,
                "directory": "checkpoints",
                "filename": "lstm.pt",      # optional
                "save_best_only": True,
            }

        If validation data are available and save_best_only=True, the
        checkpoint is updated whenever validation loss improves. If there is
        no validation split, the final trained model is saved instead.
        """
        verbose_level = self._verbose_level()

        if verbose_level >= 1:
            print("Preparing raw training data...")

        X_train, y_train = dataloader.get_train(raw=True)

        if verbose_level >= 1:
            print(f"Training data ready: {len(X_train):,} shots")
            print("Preparing validation data...")

        try:
            X_val, y_val = dataloader.get_val(raw=True)
        except (AttributeError, ValueError):
            X_val, y_val = None, None

        X_train, self.n_features = self._validate_X(X_train)
        y_train = self._validate_y(y_train, len(X_train))

        if X_val is not None and len(X_val) > 0:
            X_val, val_n_features = self._validate_X(X_val)
            y_val = self._validate_y(y_val, len(X_val))

            if val_n_features != self.n_features:
                raise ValueError(
                    "Training and validation data have different numbers of "
                    f"features: {self.n_features} vs {val_n_features}."
                )

            if verbose_level >= 1:
                print(f"Validation data ready: {len(X_val):,} shots")
        else:
            X_val, y_val = None, None

        self.model = self._build_model(self.n_features).to(self.device)

        train_loader = self._make_torch_loader(
            X_train,
            y_train,
            shuffle=True,
        )

        val_loader = None
        if X_val is not None:
            val_loader = self._make_torch_loader(
                X_val,
                y_val,
                shuffle=False,
            )

        optimizer = self._build_optimizer()
        criterion = self._build_loss()

        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

        epochs = int(self.params.get("epochs", 50))
        gradient_clip = self.params.get("gradient_clip", 1.0)

        early_stopping = bool(self.params.get("early_stopping", False))
        patience = int(self.params.get("patience", 10))
        min_delta = float(self.params.get("min_delta", 0.0))
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if early_stopping and patience <= 0:
            raise ValueError("patience must be positive when early_stopping is enabled.")
        if not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be a finite non-negative number.")

        checkpoint_cfg = self._checkpoint_config()
        checkpoint_enabled = checkpoint_cfg["enabled"]
        save_best_only = checkpoint_cfg["save_best_only"]

        best_val_loss = float("inf")
        patience_reference_loss = float("inf")
        best_state = None
        best_optimizer_state = None
        epochs_without_improvement = 0
        saved_any_checkpoint = False

        if early_stopping and val_loader is None and verbose_level >= 1:
            print(
                "Early stopping is enabled, but no validation data are "
                "available; training will run for all configured epochs."
            )

        for epoch in range(epochs):
            train_loss = self._train_epoch(
                train_loader,
                optimizer,
                criterion,
                gradient_clip,
            )
            self.history["train_loss"].append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader, criterion)
                self.history["val_loss"].append(val_loss)

            if verbose_level >= 1:
                msg = (
                    f"Epoch {epoch + 1:03d}/{epochs} "
                    f"- train_loss: {train_loss:.6f}"
                )
                if val_loss is not None:
                    msg += f" - val_loss: {val_loss:.6f}"
                print(msg)

            raw_improved = False
            significant_improved = False
            if val_loss is not None:
                if not np.isfinite(val_loss):
                    raise ValueError(
                        f"Validation loss is not finite at epoch {epoch + 1}: "
                        f"{val_loss}."
                    )
                raw_improved = val_loss < best_val_loss
                if raw_improved:
                    best_val_loss = val_loss
                    self.best_val_loss_ = val_loss
                    self.best_epoch_ = epoch + 1
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    best_optimizer_state = deepcopy(optimizer.state_dict())

                significant_improved = (
                    val_loss < patience_reference_loss - min_delta
                )
                if significant_improved:
                    patience_reference_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            # Checkpoint behavior:
            # - with validation + save_best_only: only save improvements
            # - otherwise: overwrite with the latest epoch
            if checkpoint_enabled:
                should_save = False
                if val_loss is not None and save_best_only:
                    should_save = raw_improved
                elif not save_best_only:
                    should_save = True

                if should_save:
                    self.checkpoint_path_ = self._save_checkpoint(
                        optimizer=optimizer,
                        epoch=epoch + 1,
                        train_loss=train_loss,
                        val_loss=val_loss,
                    )
                    saved_any_checkpoint = True

                    if verbose_level >= 1:
                        print(f"Checkpoint saved: {self.checkpoint_path_}")

            if (
                early_stopping
                and val_loss is not None
                and epochs_without_improvement >= patience
            ):
                if verbose_level >= 1:
                    print(
                        f"Early stopping at epoch {epoch + 1}. "
                        f"Best val_loss: {best_val_loss:.6f} "
                        f"(epoch {self.best_epoch_})."
                    )
                break

        # Restore best validation weights after training if we tracked them.
        if best_state is not None and early_stopping:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
            if best_optimizer_state is not None:
                optimizer.load_state_dict(best_optimizer_state)
            # When configured to keep the latest checkpoint during training,
            # rewrite it after restoring the best validation epoch.
            if checkpoint_enabled and not save_best_only:
                best_epoch_index = self.best_epoch_ - 1
                self.checkpoint_path_ = self._save_checkpoint(
                    optimizer=optimizer,
                    epoch=self.best_epoch_,
                    train_loss=self.history["train_loss"][best_epoch_index],
                    val_loss=self.best_val_loss_,
                )
                saved_any_checkpoint = True

        self.is_trained = True

        # If there was no validation split and save_best_only=True, there was
        # no "best" validation epoch to trigger a save. Save the final model.
        if checkpoint_enabled and not saved_any_checkpoint:
            final_epoch = len(self.history["train_loss"])
            final_train_loss = self.history["train_loss"][-1]
            final_val_loss = (
                self.history["val_loss"][-1]
                if self.history["val_loss"]
                else None
            )

            self.checkpoint_path_ = self._save_checkpoint(
                optimizer=optimizer,
                epoch=final_epoch,
                train_loss=final_train_loss,
                val_loss=final_val_loss,
            )

            if verbose_level >= 1:
                print(f"Checkpoint saved: {self.checkpoint_path_}")

    def predict(self, dataloader: CroissantDataLoader) -> np.ndarray:
        probabilities = self.predict_proba(dataloader, split="test")[:, 1]
        threshold = float(self.params.get("classification_threshold", 0.5))
        return (probabilities >= threshold).astype(np.uint8)

    def predict_proba(
        self,
        dataloader: CroissantDataLoader,
        split: str = "test",
    ) -> np.ndarray:
        logits = self.decision_function(dataloader, split=split)
        p1 = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - p1, p1])

    def decision_function(
        self,
        dataloader: CroissantDataLoader,
        split: str = "test",
    ) -> np.ndarray:
        self._check_is_trained()

        X, _ = self._get_split(dataloader, split)
        X, n_features = self._validate_X(X)

        if n_features != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {n_features}."
            )

        loader = self._make_torch_loader(
            X,
            y=None,
            shuffle=False,
        )

        self.model.eval()
        outputs = []

        with torch.no_grad():
            for x, lengths, padding_mask in loader:
                x = x.to(self.device, non_blocking=True)
                lengths = lengths.to(self.device, non_blocking=True)
                padding_mask = padding_mask.to(self.device, non_blocking=True)

                logits = self._forward_batch(
                    x,
                    lengths,
                    padding_mask,
                )

                outputs.append(logits.detach().cpu().numpy())

        return np.concatenate(outputs)

    def get_labels(
        self,
        dataloader: CroissantDataLoader,
        split: str = "test",
    ) -> np.ndarray:
        _, y = self._get_split(dataloader, split)
        return np.asarray(y)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        gradient_clip: float | None,
    ) -> float:
        self.model.train()

        total_loss = 0.0
        n_samples = 0

        for x, lengths, padding_mask, y in loader:
            x = x.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)
            padding_mask = padding_mask.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = self._forward_batch(
                x,
                lengths,
                padding_mask,
            )

            loss = criterion(logits, y)
            loss.backward()

            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(gradient_clip),
                )

            optimizer.step()

            batch_size = len(y)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        return total_loss / n_samples

    @torch.no_grad()
    def _evaluate_loss(
        self,
        loader: DataLoader,
        criterion: torch.nn.Module,
    ) -> float:
        self.model.eval()

        total_loss = 0.0
        n_samples = 0

        for x, lengths, padding_mask, y in loader:
            x = x.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)
            padding_mask = padding_mask.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            logits = self._forward_batch(
                x,
                lengths,
                padding_mask,
            )

            loss = criterion(logits, y)

            batch_size = len(y)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        return total_loss / n_samples

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.params.get("learning_rate", 5e-4)),
            weight_decay=float(self.params.get("weight_decay", 0.0)),
        )

    def _build_loss(self) -> torch.nn.Module:
        return torch.nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Raw data -> PyTorch batching
    # ------------------------------------------------------------------

    def _make_torch_loader(
        self,
        X: Iterable[np.ndarray],
        y: np.ndarray | None,
        shuffle: bool,
    ) -> DataLoader:
        dataset = _RawShotDataset(X, y)

        return DataLoader(
            dataset,
            batch_size=int(self.params.get("batch_size", 64)),
            shuffle=shuffle,
            collate_fn=(
                self._collate_labeled
                if y is not None
                else self._collate_unlabeled
            ),
            num_workers=int(self.params.get("num_workers", 0)),
            pin_memory=bool(self.params.get("pin_memory", False)),
        )

    @staticmethod
    def _collate_labeled(batch):
        sequences, labels = zip(*batch)
        x, lengths, padding_mask = BaseRawSequenceClassifier._pad_sequences(
            sequences
        )
        y = torch.stack(labels)
        return x, lengths, padding_mask, y

    @staticmethod
    def _collate_unlabeled(batch):
        return BaseRawSequenceClassifier._pad_sequences(batch)

    @staticmethod
    def _pad_sequences(
        sequences: Iterable[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences = list(sequences)

        lengths = torch.tensor(
            [len(sequence) for sequence in sequences],
            dtype=torch.long,
        )

        x = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=0.0,
        )

        max_length = x.shape[1]
        padding_mask = (
            torch.arange(max_length)[None, :]
            >= lengths[:, None]
        )

        return x, lengths, padding_mask

    # ------------------------------------------------------------------
    # Data-loader interaction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_split(
        dataloader: CroissantDataLoader,
        split: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        split = split.lower()

        if split == "train":
            return dataloader.get_train(raw=True)
        if split == "val":
            return dataloader.get_val(raw=True)
        if split == "test":
            return dataloader.get_test(raw=True)

        raise ValueError(
            f"Unknown split '{split}'. Expected 'train', 'val', or 'test'."
        )

    @staticmethod
    def _validate_X(X: Iterable[np.ndarray]) -> Tuple[list[np.ndarray], int]:
        X = list(X)

        if len(X) == 0:
            raise ValueError("X is empty.")

        first = np.asarray(X[0])
        if first.ndim != 2:
            raise ValueError(
                "Each shot must have shape (n_rounds, n_features)."
            )

        n_features = first.shape[1]

        for i, shot in enumerate(X):
            shot = np.asarray(shot)

            if shot.ndim != 2:
                raise ValueError(
                    f"Shot {i} must be 2D, got shape {shot.shape}."
                )

            if len(shot) == 0:
                raise ValueError(f"Shot {i} contains zero rounds.")

            if shot.shape[1] != n_features:
                raise ValueError(
                    f"Shot {i} has {shot.shape[1]} features; "
                    f"expected {n_features}."
                )

        return X, n_features

    @staticmethod
    def _validate_y(y: np.ndarray, n_samples: int) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)

        if len(y) != n_samples:
            raise ValueError(
                f"X contains {n_samples} shots but y contains {len(y)} labels."
            )

        unique = np.unique(y)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError(
                f"Expected binary labels 0/1, found {unique.tolist()}."
            )

        return y

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _checkpoint_config(self) -> Dict[str, Any]:
        """
        Return normalized checkpoint configuration.

        params['checkpoint'] can be omitted entirely. If supplied, the
        directory is created automatically when a checkpoint is written.
        """
        raw = self.params.get("checkpoint", {})

        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("params['checkpoint'] must be a dictionary.")

        return {
            "enabled": bool(raw.get("enabled", False)),
            "directory": Path(raw.get("directory", "checkpoints")),
            "filename": raw.get("filename", None),
            "save_best_only": bool(raw.get("save_best_only", True)),
        }

    def _checkpoint_path(self) -> Path:
        cfg = self._checkpoint_config()
        directory = Path(cfg["directory"])
        directory.mkdir(parents=True, exist_ok=True)

        filename = cfg["filename"]
        if filename is None:
            filename = f"{self.model_name}.pt"

        filename = str(filename)
        if not filename.endswith(".pt"):
            filename += ".pt"

        return directory / filename

    def _save_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        train_loss: float,
        val_loss: float | None,
    ) -> Path:
        path = self._checkpoint_path()

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "params": self.params,
                "n_features": self.n_features,
                "model_name": self.model_name,
                "history": self.history,
                "best_epoch": self.best_epoch_,
                "best_val_loss": self.best_val_loss_,
            },
            path,
        )

        return path

    # ------------------------------------------------------------------
    # General model utilities
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        configured = self.params.get("device", None)
        if configured is not None:
            return torch.device(configured)

        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _verbose_level(self) -> int:
        verbose = self.params.get("verbose", 1)
        if isinstance(verbose, bool):
            return 2 if verbose else 0
        return int(verbose)

    def _check_is_trained(self) -> None:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model has not been trained yet.")

    def save_model(self, path: Path) -> Path:
        """
        Manually save a trained model checkpoint to a directory.
        """
        self._check_is_trained()

        path = Path(path)
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "params": self.params,
                "n_features": self.n_features,
                "model_name": self.model_name,
                "history": self.history,
                "best_epoch": self.best_epoch_,
                "best_val_loss": self.best_val_loss_,
            },
            path,
        )

        return path

    @classmethod
    def load_model(
        cls,
        path: Path,
        params: Dict[str, Any],
        model_args: Any,
    ) -> "BaseRawSequenceClassifier":
        instance = cls(
            params=params,
            args=model_args,
        )

        checkpoint = torch.load(
            path,
            map_location=instance.device,
        )

        instance.n_features = int(checkpoint["n_features"])
        instance.model = instance._build_model(instance.n_features).to(
            instance.device
        )
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.history = checkpoint.get(
            "history",
            {"train_loss": [], "val_loss": []},
        )
        instance.best_epoch_ = checkpoint.get("best_epoch")
        instance.best_val_loss_ = checkpoint.get("best_val_loss")
        instance.checkpoint_path_ = Path(path)
        instance.is_trained = True

        return instance
