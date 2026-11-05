"""Dataset loading and reproducible split helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_split_indices(
    num_examples: int,
    *,
    seed: int = 12345,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic train/validation/test split indices."""
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("fractions must leave a non-empty test split")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_examples, dtype=np.int64)
    rng.shuffle(indices)

    n_train = int(train_fraction * num_examples)
    n_val = int(val_fraction * num_examples)
    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train : n_train + n_val])
    test_idx = np.sort(indices[n_train + n_val :])
    return train_idx, val_idx, test_idx


def load_split_indices(npz: np.lib.npyio.NpzFile, *, seed: int = 12345) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load saved split indices or create them for older files."""
    if {"train_idx", "val_idx", "test_idx"}.issubset(npz.files):
        return (
            npz["train_idx"].astype(np.int64),
            npz["val_idx"].astype(np.int64),
            npz["test_idx"].astype(np.int64),
        )

    num_examples = len(npz["labels"] if "labels" in npz.files else npz["y"])
    return make_split_indices(num_examples, seed=seed)


def load_graph_dataset(path: Path | str) -> dict[str, np.ndarray]:
    """Load graph tensors and split indices from an ``.npz`` file."""
    with np.load(path, allow_pickle=False) as data:
        required = {"X", "A", "mask", "y"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Missing arrays from graph dataset: {sorted(missing)}")

        train_idx, val_idx, test_idx = load_split_indices(data)
        result = {
            "X": data["X"].astype(np.float32),
            "A": data["A"].astype(np.float32),
            "mask": data["mask"].astype(np.float32),
            "y": data["y"].astype(np.float32).reshape(-1),
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        }

        for key in (
            "distance",
            "rounds",
            "p",
            "shots",
            "max_nodes",
            "k_neighbors",
            "feature_dim",
            "truncation_rate",
            "empty_graph_fraction",
            "graph_type",
            "node_features",
            "dem_edge_count",
            "boundary_detector_count",
        ):
            if key in data.files:
                result[key] = np.asarray(data[key])

    return result
