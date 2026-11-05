import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from model.full_decoder.sequence.base_sequence_classifier import BaseRawSequenceClassifier


class _LSTMRawShotDataset(Dataset):
    """
    Lightweight dataset for variable-length LSTM sequences.

    Tensors remain on CPU. With pin_memory=True, batches can then be copied
    efficiently to CUDA with non_blocking=True.
    """

    def __init__(
        self,
        X: Iterable[np.ndarray],
        y: np.ndarray | None = None,
        shot_features: np.ndarray | None = None,
    ):
        self.X = [
            torch.as_tensor(x, dtype=torch.float32)
            for x in X
        ]
        self.y = (
            None
            if y is None
            else torch.as_tensor(y, dtype=torch.float32)
        )
        self.shot_features = (
            None
            if shot_features is None
            else torch.as_tensor(shot_features, dtype=torch.float32)
        )

        if self.y is not None and len(self.X) != len(self.y):
            raise ValueError("X and y must contain the same number of shots.")
        if self.shot_features is not None:
            if self.shot_features.ndim != 2:
                raise ValueError("shot_features must have shape (n_shots, n_features).")
            if len(self.X) != len(self.shot_features):
                raise ValueError("X and shot_features must contain the same number of shots.")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.shot_features is not None:
            if self.y is None:
                return self.X[idx], self.shot_features[idx]
            return self.X[idx], self.shot_features[idx], self.y[idx]
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class _LSTMNetwork(nn.Module):
    """
    Stacked LSTM followed by a binary classification head.

    Important:
        lengths stays on CPU. pack_padded_sequence requires CPU lengths, so
        sending lengths to CUDA and immediately copying them back causes an
        avoidable synchronization on every batch.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.05,
        shot_feature_size: int = 0,
        hidden_out: int = 24,
        head_dropout: float = 0.1,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # This mirrors the detector/final-feature fusion pattern: the LSTM
        # encodes only the round-by-round sequence, then its final hidden state
        # is concatenated with the shot-level values before classification.
        # Keep a small nonlinear head so the classifier can learn interactions
        # between the sequence summary and final-round features.
        head_input_size = hidden_size + shot_feature_size
        self.hidden_layer = nn.Linear(head_input_size, hidden_out)
        self.hidden_activation = nn.ReLU()
        self.head_dropout = nn.Dropout(head_dropout)
        self.output_layer = nn.Linear(hidden_out, 1)
        self.shot_feature_size = shot_feature_size

    def encode(
        self,
        x: Tensor,
        lengths: Tensor,
        shot_features: Tensor | None = None,
    ) -> Tensor:
        """Return the hidden_out representation for each shot."""
        # lengths is intentionally kept on CPU.
        packed = pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)
        final_hidden = hidden[-1]

        if self.shot_feature_size:
            if shot_features is None:
                raise ValueError(
                    "This model was trained with shot-level features; "
                    "provide shot_features for every shot."
                )
            if shot_features.ndim != 2 or shot_features.shape[1] != self.shot_feature_size:
                raise ValueError(
                    "shot_features must have shape "
                    f"(batch, {self.shot_feature_size})."
                )
            final_hidden = torch.cat((final_hidden, shot_features), dim=1)
        elif shot_features is not None:
            raise ValueError(
                "shot_features were supplied, but this model was built "
                "without shot-level features."
            )

        # This is the learned encoding before dropout and the classification
        # output layer. It is stable for extraction when the model is in eval.
        return self.hidden_activation(self.hidden_layer(final_hidden))

    def forward(
        self,
        x: Tensor,
        lengths: Tensor,
        shot_features: Tensor | None = None,
    ) -> Tensor:
        hidden_out = self.encode(x, lengths, shot_features)
        return self.output_layer(self.head_dropout(hidden_out)).squeeze(-1)


class LSTMSequenceClassifier(BaseRawSequenceClassifier):
    """
    CUDA-backed LSTM classifier for variable-length raw QEC shot sequences.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        args: Any,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "LSTMSequenceClassifier requires CUDA, but "
                "torch.cuda.is_available() returned False."
            )

        params = dict(params)
        params["device"] = "cuda"

        # Throughput-oriented defaults. Explicit user values still win.
        params.setdefault("pin_memory", True)
        # On Windows, worker processes must spawn and receive the eagerly
        # tensorized dataset. Start with the main process to avoid a long
        # first-batch stall; users can raise this after benchmarking.
        params.setdefault("num_workers", 0)
        params.setdefault("prefetch_factor", 4)
        params.setdefault("inference_batch_size", 4096)
        params.setdefault("amp", True)
        params.setdefault("verbose", 2)
        params.setdefault("hidden_out", 24)
        params.setdefault("head_dropout", 0.1)

        super().__init__(
            params=params,
            args=args,
        )

        # Useful for repeated, similarly shaped batches.
        torch.backends.cudnn.benchmark = True

        # Ampere/Ada GPUs (including RTX 4070 Ti) can benefit from TF32.
        # This mainly affects supported CUDA/cuDNN kernels and can introduce
        # very small floating-point differences.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.n_shot_features = int(self.params.get("shot_feature_size", 0))

        if self._verbose_level() > 0:
            gpu_index = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_index)
            print(f"Using GPU: cuda:{gpu_index} ({gpu_name})")

    def _build_model(self, n_features: int) -> nn.Module:
        return _LSTMNetwork(
            input_size=n_features,
            hidden_size=int(self.params.get("hidden_size", 64)),
            num_layers=int(self.params.get("num_layers", 2)),
            dropout=float(self.params.get("dropout", 0.05)),
            shot_feature_size=self.n_shot_features,
            hidden_out=int(self.params.get("hidden_out", 24)),
            head_dropout=float(self.params.get("head_dropout", 0.1)),
        )

    def _forward_batch(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor | None,
        shot_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # padding_mask is intentionally unused.
        return self.model(x, lengths, shot_features)

    @staticmethod
    def _validate_shot_features(
        shot_features: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        shot_features = np.asarray(shot_features, dtype=np.float32)
        if shot_features.ndim == 1:
            shot_features = shot_features[:, None]
        if shot_features.ndim != 2:
            raise ValueError(
                "shot_features must have shape (n_shots, n_features)."
            )
        if len(shot_features) != n_samples:
            raise ValueError(
                f"Expected shot_features for {n_samples} shots, "
                f"got {len(shot_features)}."
            )
        if shot_features.shape[1] == 0:
            raise ValueError("shot_features must contain at least one column.")
        if not np.isfinite(shot_features).all():
            raise ValueError("shot_features contains NaN or infinity.")
        return shot_features

    def _prepare_shot_features(
        self,
        shot_features: np.ndarray | None,
        n_samples: int,
    ) -> np.ndarray | None:
        """Validate optional shot features, treating a zero-width matrix as absent."""
        if self.n_shot_features:
            if shot_features is None:
                raise ValueError(
                    "This model uses shot-level features. Pass the aligned "
                    "shot_features matrix."
                )
            shot_features = self._validate_shot_features(
                shot_features, n_samples
            )
            if shot_features.shape[1] != self.n_shot_features:
                raise ValueError(
                    f"Expected {self.n_shot_features} shot-level features, "
                    f"got {shot_features.shape[1]}."
                )
            return shot_features

        if shot_features is None:
            return None

        candidate = np.asarray(shot_features)
        if candidate.ndim == 2 and candidate.shape == (n_samples, 0):
            return None
        raise ValueError(
            "This model was trained without shot-level features; "
            "omit shot_features or pass an empty (n_shots, 0) matrix."
        )

    def fit(self, dataloader) -> None:
        """Fit using shot columns listed in Croissant metadata, when present."""
        resolved_columns = dataloader.get_shot_feature_columns()
        if not resolved_columns:
            self.n_shot_features = 0
            self.params["shot_feature_size"] = 0
            self.params.pop("shot_feature_columns_ordered", None)
            return super().fit(dataloader)

        self.params["shot_feature_columns_ordered"] = resolved_columns

        verbose_level = self._verbose_level()
        if verbose_level >= 1:
            print("Preparing raw training data and shot-level features...")

        X_train, y_train, shot_train = dataloader.get_train(
            raw=True,
            include_shot_features=True,
        )
        try:
            X_val, y_val, shot_val = dataloader.get_val(
                raw=True,
                include_shot_features=True,
            )
        except (AttributeError, ValueError):
            X_val, y_val, shot_val = None, None, None

        X_train, self.n_features = self._validate_X(X_train)
        y_train = self._validate_y(y_train, len(X_train))
        shot_train = self._validate_shot_features(shot_train, len(X_train))
        self.n_shot_features = shot_train.shape[1]
        self.params["shot_feature_size"] = self.n_shot_features

        if X_val is not None and len(X_val) > 0:
            X_val, val_n_features = self._validate_X(X_val)
            y_val = self._validate_y(y_val, len(X_val))
            shot_val = self._validate_shot_features(shot_val, len(X_val))
            if val_n_features != self.n_features:
                raise ValueError(
                    "Training and validation sequences have different "
                    "feature counts."
                )
            if shot_val.shape[1] != self.n_shot_features:
                raise ValueError(
                    "Training and validation shot-level feature counts differ."
                )
            if verbose_level >= 1:
                print(f"Validation data ready: {len(X_val):,} shots")
        else:
            X_val, y_val, shot_val = None, None, None

        self.model = self._build_model(self.n_features).to(self.device)
        train_loader = self._make_torch_loader(
            X_train, y_train, shuffle=True, shot_features=shot_train
        )
        val_loader = (
            None if X_val is None else self._make_torch_loader(
                X_val, y_val, shuffle=False, shot_features=shot_val
            )
        )

        optimizer = self._build_optimizer()
        criterion = self._build_loss()
        self.history = {"train_loss": [], "val_loss": []}

        epochs = int(self.params.get("epochs", 50))
        gradient_clip = self.params.get("gradient_clip", 1.0)
        early_stopping = bool(self.params.get("early_stopping", False))
        patience = int(self.params.get("patience", 10))
        min_delta = float(self.params.get("min_delta", 0.0))
        checkpoint_cfg = self._checkpoint_config()
        checkpoint_enabled = checkpoint_cfg["enabled"]
        save_best_only = checkpoint_cfg["save_best_only"]

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        saved_any_checkpoint = False
        self.best_epoch_ = None
        self.best_val_loss_ = None

        for epoch in range(epochs):
            train_loss = self._train_epoch(
                train_loader, optimizer, criterion, gradient_clip
            )
            self.history["train_loss"].append(train_loss)
            val_loss = None
            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader, criterion)
                self.history["val_loss"].append(val_loss)

            if verbose_level >= 1:
                message = (
                    f"Epoch {epoch + 1:03d}/{epochs} "
                    f"- train_loss: {train_loss:.6f}"
                )
                if val_loss is not None:
                    message += f" - val_loss: {val_loss:.6f}"
                print(message)

            improved = (
                val_loss is not None
                and val_loss < best_val_loss - min_delta
            )
            if improved:
                best_val_loss = val_loss
                self.best_val_loss_ = val_loss
                self.best_epoch_ = epoch + 1
                epochs_without_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            elif val_loss is not None:
                epochs_without_improvement += 1

            if checkpoint_enabled:
                should_save = (
                    improved if val_loss is not None and save_best_only
                    else not save_best_only
                )
                if should_save:
                    self.checkpoint_path_ = self._save_checkpoint(
                        optimizer=optimizer,
                        epoch=epoch + 1,
                        train_loss=train_loss,
                        val_loss=val_loss,
                    )
                    saved_any_checkpoint = True

            if (
                early_stopping
                and val_loss is not None
                and epochs_without_improvement >= patience
            ):
                if verbose_level >= 1:
                    print(
                        f"Early stopping at epoch {epoch + 1}; "
                        f"best validation loss {best_val_loss:.6f}."
                    )
                break

        if best_state is not None and early_stopping:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        self.is_trained = True

        if checkpoint_enabled and not saved_any_checkpoint:
            self.checkpoint_path_ = self._save_checkpoint(
                optimizer=optimizer,
                epoch=len(self.history["train_loss"]),
                train_loss=self.history["train_loss"][-1],
                val_loss=(self.history["val_loss"][-1]
                          if self.history["val_loss"] else None),
            )

    # ------------------------------------------------------------------
    # LSTM-specific batching
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_only(
        sequences: Iterable[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        return x, lengths

    @staticmethod
    def _collate_labeled_lstm(batch):
        if len(batch[0]) == 3:
            sequences, shot_features, labels = zip(*batch)
            x, lengths = LSTMSequenceClassifier._pad_only(sequences)
            return (
                x,
                lengths,
                None,
                torch.stack(shot_features),
                torch.stack(labels),
            )

        sequences, labels = zip(*batch)
        x, lengths = LSTMSequenceClassifier._pad_only(sequences)
        y = torch.stack(labels)

        # Keep API compatible with BaseRawSequenceClassifier:
        # (x, lengths, padding_mask, y)
        return x, lengths, None, y

    @staticmethod
    def _collate_unlabeled_lstm(batch):
        if isinstance(batch[0], tuple):
            sequences, shot_features = zip(*batch)
            x, lengths = LSTMSequenceClassifier._pad_only(sequences)
            return x, lengths, None, torch.stack(shot_features)

        x, lengths = LSTMSequenceClassifier._pad_only(batch)

        # Keep API compatible with BaseRawSequenceClassifier:
        # (x, lengths, padding_mask)
        return x, lengths, None

    def _make_torch_loader(
        self,
        X: Iterable[np.ndarray],
        y: np.ndarray | None,
        shuffle: bool,
        shot_features: np.ndarray | None = None,
    ) -> DataLoader:
        dataset = _LSTMRawShotDataset(X, y, shot_features)

        is_inference = y is None
        batch_size = int(
            self.params.get(
                "inference_batch_size" if is_inference else "batch_size",
                2048 if is_inference else 256,
            )
        )

        num_workers = int(self.params.get("num_workers", 0))
        pin_memory = bool(self.params.get("pin_memory", True))

        kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=(
                self._collate_unlabeled_lstm
                if is_inference
                else self._collate_labeled_lstm
            ),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

        if num_workers > 0:
            kwargs["prefetch_factor"] = int(
                self.params.get("prefetch_factor", 4)
            )

        return DataLoader(**kwargs)

    # ------------------------------------------------------------------
    # AMP helper
    # ------------------------------------------------------------------

    def _autocast_context(self):
        enabled = bool(self.params.get("amp", True))

        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=enabled,
        )

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        gradient_clip: float | None,
    ) -> float:
        self.model.train()

        # Keep loss aggregation on-device so reading it does not synchronize
        # CUDA once per batch.
        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        n_samples = 0
        verbose_level = self._verbose_level()

        # AMP scaler only matters during training.
        amp_enabled = bool(self.params.get("amp", True))
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        iterator = loader
        if verbose_level >= 2:
            iterator = tqdm(
                loader,
                desc="Training",
                leave=False,
                dynamic_ncols=True,
            )

        for batch in iterator:
            if len(batch) == 5:
                x, lengths, _, shot_features, y = batch
                shot_features = shot_features.to(
                    self.device, non_blocking=True
                )
            else:
                x, lengths, _, y = batch
                shot_features = None

            # x and y go to CUDA; lengths deliberately stays on CPU.
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with self._autocast_context():
                logits = self.model(x, lengths, shot_features)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            if gradient_clip is not None:
                # Unscale before clipping.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(gradient_clip),
                )

            scaler.step(optimizer)
            scaler.update()

            batch_size = len(y)
            total_loss.add_(loss.detach().float() * batch_size)
            n_samples += batch_size

        return total_loss.item() / n_samples

    @torch.inference_mode()
    def _evaluate_loss(
        self,
        loader: DataLoader,
        criterion: torch.nn.Module,
    ) -> float:
        self.model.eval()

        # Defer the CUDA-to-CPU read until the epoch's validation is done.
        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        n_samples = 0

        for batch in loader:
            if len(batch) == 5:
                x, lengths, _, shot_features, y = batch
                shot_features = shot_features.to(
                    self.device, non_blocking=True
                )
            else:
                x, lengths, _, y = batch
                shot_features = None

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with self._autocast_context():
                logits = self.model(x, lengths, shot_features)
                loss = criterion(logits, y)

            batch_size = len(y)
            total_loss.add_(loss.detach().float() * batch_size)
            n_samples += batch_size

        return total_loss.item() / n_samples

    @torch.inference_mode()
    def decision_function(
        self,
        X: np.ndarray,
        shot_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one logit per shot, converting NumPy sequences to tensors.

        X contains sequences shaped (timesteps, features), either as a dense
        array shaped (shots, timesteps, features) or a ragged object array.
        """
        self._check_is_trained()

        # Handle empty input before validation, which may inspect the first shot.
        if len(X) == 0:
            return np.empty(0, dtype=np.float32)

        X, n_features = self._validate_X(X)
        if n_features != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {n_features}."
            )

        shot_features = self._prepare_shot_features(
            shot_features, len(X)
        )

        batch_size = int(self.params.get("inference_batch_size", 4096))
        if batch_size <= 0:
            raise ValueError("inference_batch_size must be positive.")

        self.model.eval()
        outputs = []

        for start in range(0, len(X), batch_size):
            stop = start + batch_size
            sequences = [
                torch.as_tensor(sequence, dtype=torch.float32)
                for sequence in X[start:stop]
            ]
            if any(sequence.ndim != 2 or len(sequence) == 0
                   for sequence in sequences):
                raise ValueError(
                    "Each shot must be a non-empty 2D sequence "
                    "with shape (timesteps, features)."
                )

            x, lengths = self._pad_only(sequences)
            if bool(self.params.get("pin_memory", True)):
                x = x.pin_memory()
            x = x.to(self.device, non_blocking=True)
            shot_batch = None
            if shot_features is not None:
                shot_batch = torch.as_tensor(
                    shot_features[start:stop], dtype=torch.float32
                ).to(self.device, non_blocking=True)

            # lengths stays on CPU for pack_padded_sequence.
            with self._autocast_context():
                logits = self.model(x, lengths, shot_batch)

            outputs.append(logits.float().cpu())

        return torch.cat(outputs, dim=0).numpy()

    @torch.inference_mode()
    def encode(
        self,
        X: np.ndarray,
        shot_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one learned ``hidden_out`` encoding per shot.

        The encoding is the 24-dimensional-by-default ReLU activation after
        fusing the LSTM's final hidden state with any configured shot features.
        It is taken before dropout and the final classification layer.
        """
        self._check_is_trained()
        encoding_size = int(self.params.get("hidden_out", 24))
        if len(X) == 0:
            return np.empty((0, encoding_size), dtype=np.float32)

        X, n_features = self._validate_X(X)
        if n_features != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {n_features}."
            )
        shot_features = self._prepare_shot_features(
            shot_features, len(X)
        )

        batch_size = int(self.params.get("inference_batch_size", 4096))
        if batch_size <= 0:
            raise ValueError("inference_batch_size must be positive.")

        self.model.eval()
        outputs = []
        for start in range(0, len(X), batch_size):
            stop = start + batch_size
            sequences = [
                torch.as_tensor(sequence, dtype=torch.float32)
                for sequence in X[start:stop]
            ]
            if any(sequence.ndim != 2 or len(sequence) == 0
                   for sequence in sequences):
                raise ValueError(
                    "Each shot must be a non-empty 2D sequence "
                    "with shape (timesteps, features)."
                )

            x, lengths = self._pad_only(sequences)
            if bool(self.params.get("pin_memory", True)):
                x = x.pin_memory()
            x = x.to(self.device, non_blocking=True)
            shot_batch = None
            if shot_features is not None:
                shot_batch = torch.as_tensor(
                    shot_features[start:stop], dtype=torch.float32
                ).to(self.device, non_blocking=True)

            with self._autocast_context():
                hidden_out = self.model.encode(x, lengths, shot_batch)
            outputs.append(hidden_out.float().cpu())

        return torch.cat(outputs, dim=0).numpy()

    def save_encodings(
        self,
        X: np.ndarray,
        path: str | Path,
        shot_features: np.ndarray | None = None,
    ) -> Path:
        """Encode shots and save the resulting matrix as a NumPy ``.npy`` file.

        If ``path`` does not end in ``.npy``, that extension is appended.
        Returns the actual path written.
        """
        encodings = self.encode(X, shot_features=shot_features)
        output_path = Path(path)
        if output_path.suffix.lower() != ".npy":
            output_path = Path(f"{output_path}.npy")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, encodings, allow_pickle=False)
        return output_path

    def predict_proba(
        self,
        X: np.ndarray,
        shot_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return probabilities for classes 0 and 1, in that order."""
        logits = self.decision_function(X, shot_features=shot_features)

        # Stable sigmoid avoids overflow for large negative logits.
        p1 = np.exp(-np.logaddexp(0.0, -logits))
        return np.column_stack((1.0 - p1, p1))

    def predict(
        self,
        X: np.ndarray,
        shot_features: np.ndarray | None = None,
    ) -> np.ndarray:
        threshold = float(
            self.params.get("classification_threshold", 0.5)
        )
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "classification_threshold must be strictly between 0 and 1."
            )

        # Avoid constructing a two-column probability matrix for hard labels.
        logits = self.decision_function(X, shot_features=shot_features)
        logit_threshold = np.log(threshold / (1.0 - threshold))
        return (logits >= logit_threshold).astype(np.uint8)

    def predict_with_timing(self, X: np.ndarray, batch_size = None, shot_features: np.ndarray | None = None) -> Tuple[np.ndarray, float]:
        """
        Predict class labels for input data and measure execution time (override of base_classifier.py to include shot_features)
        If batch size set, do predictions in batches, execution time measured per batch and summed.

        :param X: Input data (sequence data).
        :param batch_size: Batch size, defaults to None (all in one batch)
        :param shot_features: Shot-level features
        :return: Predicted class labels and execution time.
        """
        if batch_size is None: # Default: all predictions done in single batch
            time_start = time.perf_counter()
            predictions = self.predict(X, shot_features=shot_features)
            time_end = time.perf_counter()
            return predictions, time_end - time_start

        # Prediction in batches, timing is summed up over all batches
        chunks = []
        timings = 0.0

        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, batch_size)
            chunk, chunk_timing = self.predict_with_timing(X[start:end], shot_features=shot_features)
            chunks.append(chunk)
            timings += chunk_timing

        predictions = np.concatenate(chunks, axis=0)
        return predictions, timings

    @classmethod
    def load_model(
        cls,
        path,
        params: Dict[str, Any],
        model_args: Any,
    ):
        """Load checkpoints while restoring the shot-feature input width."""
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint_params = dict(checkpoint.get("params", {}))
        resolved_params = dict(checkpoint_params)
        resolved_params.update(params)

        # Architecture and feature ordering must come from the saved model.
        for key in (
            "shot_feature_size",
            "shot_feature_columns_ordered",
            "hidden_out",
            "head_dropout",
        ):
            if key in checkpoint_params:
                resolved_params[key] = checkpoint_params[key]
        resolved_params["shot_feature_size"] = int(
            checkpoint_params.get(
                "shot_feature_size",
                checkpoint.get("n_shot_features", 0),
            )
        )

        instance = cls(params=resolved_params, args=model_args)
        instance.n_features = int(checkpoint["n_features"])
        instance.n_shot_features = int(resolved_params["shot_feature_size"])
        instance.model = instance._build_model(instance.n_features).to(
            instance.device
        )
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.history = checkpoint.get(
            "history", {"train_loss": [], "val_loss": []}
        )
        instance.best_epoch_ = checkpoint.get("best_epoch")
        instance.best_val_loss_ = checkpoint.get("best_val_loss")
        instance.checkpoint_path_ = Path(path)
        instance.is_trained = True
        return instance
