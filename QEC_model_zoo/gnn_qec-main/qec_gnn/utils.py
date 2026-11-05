"""Shared serialization and metric helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def binary_classification_metrics(
    y_true: np.ndarray,
    y_score_or_pred: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute robust binary metrics for imbalanced logical-flip labels."""
    y_true = np.asarray(y_true).astype(np.uint8).reshape(-1)
    values = np.asarray(y_score_or_pred).reshape(-1)
    if np.issubdtype(values.dtype, np.floating):
        y_pred = (values >= threshold).astype(np.uint8)
    else:
        y_pred = values.astype(np.uint8)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = len(y_true)
    positives = tp + fn
    negatives = tn + fp
    pred_positives = tp + fp

    accuracy = (tp + tn) / total if total else 0.0
    recall = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    precision = tp / pred_positives if pred_positives else None
    available_rates = [x for x in (recall, specificity) if x is not None]
    balanced_accuracy = float(np.mean(available_rates)) if available_rates else None

    return {
        "accuracy": float(accuracy),
        "error_rate": float(1.0 - accuracy),
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_label_fraction": float(np.mean(y_true)) if total else 0.0,
        "predicted_positive_fraction": float(np.mean(y_pred)) if total else 0.0,
        "num_examples": int(total),
    }


def print_metric_block(title: str, metrics: dict[str, Any]) -> None:
    print(f"{title}:")
    for key in (
        "accuracy",
        "error_rate",
        "balanced_accuracy",
        "positive_label_fraction",
        "tp",
        "tn",
        "fp",
        "fn",
    ):
        value = metrics.get(key)
        if isinstance(value, float):
            print(f"  {key} = {value:.6f}")
        else:
            print(f"  {key} = {value}")
