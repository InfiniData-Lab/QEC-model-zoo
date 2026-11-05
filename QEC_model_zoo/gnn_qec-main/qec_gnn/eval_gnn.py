"""Evaluate a saved GNN decoder on the graph dataset test split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from qec_gnn.config import METRICS_DIR
from qec_gnn.dataset import load_graph_dataset
from qec_gnn.model import load_gnn_model
from qec_gnn.utils import binary_classification_metrics, print_metric_block, save_json


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.shape == () else array.tolist()


def evaluate_gnn(
    model_path: Path,
    data_path: Path,
    metrics_path: Path,
    *,
    batch_size: int = 128,
    threshold: float = 0.5,
) -> dict[str, Any]:
    data = load_graph_dataset(data_path)
    test_idx = data["test_idx"]
    model = load_gnn_model(str(model_path))
    probs = model.predict(
        [data["X"][test_idx], data["A"][test_idx], data["mask"][test_idx]],
        batch_size=batch_size,
        verbose=0,
    )
    metrics = binary_classification_metrics(data["y"][test_idx], probs[:, 0], threshold=threshold)

    config = {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "batch_size": batch_size,
        "threshold": threshold,
    }
    for key in ("distance", "rounds", "p", "shots", "max_nodes", "k_neighbors"):
        if key in data:
            config[key] = _scalar(data[key])

    payload = {"kind": "gnn", "config": config, "test": metrics}
    save_json(metrics_path, payload)
    print_metric_block("GNN test", metrics)
    print(f"metrics = {metrics_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.out or METRICS_DIR / f"gnn_eval_{args.model.stem}.json"
    evaluate_gnn(args.model, args.data, metrics_path, batch_size=args.batch_size, threshold=args.threshold)


if __name__ == "__main__":
    main()
