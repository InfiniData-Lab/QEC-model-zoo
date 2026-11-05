"""Evaluate a PyMatching MWPM baseline on the same raw test shots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pymatching

from qec_gnn.config import METRICS_DIR
from qec_gnn.dataset import load_split_indices
from qec_gnn.stim_utils import make_surface_code_circuit
from qec_gnn.utils import binary_classification_metrics, print_metric_block, save_json


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.shape == () else array.tolist()


def evaluate_mwpm(raw_data_path: Path, metrics_path: Path, *, seed: int = 12345) -> dict[str, Any]:
    with np.load(raw_data_path, allow_pickle=False) as raw:
        detectors = raw["detectors"].astype(np.uint8)
        labels = raw["labels"].astype(np.uint8).reshape(-1)
        train_idx, val_idx, test_idx = load_split_indices(raw, seed=seed)
        distance = int(raw["distance"])
        rounds = int(raw["rounds"])
        p = float(raw["p"])
        shots = int(raw["shots"]) if "shots" in raw.files else len(labels)

    circuit = make_surface_code_circuit(distance=distance, rounds=rounds, p=p)
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    predictions = matching.decode_batch(detectors[test_idx])

    if predictions.ndim == 1:
        pred = predictions.astype(np.uint8)
    else:
        pred = predictions[:, 0].astype(np.uint8)

    metrics = binary_classification_metrics(labels[test_idx], pred)
    payload = {
        "kind": "mwpm",
        "config": {
            "raw_data_path": str(raw_data_path),
            "distance": distance,
            "rounds": rounds,
            "p": p,
            "shots": shots,
            "test_shots": int(len(test_idx)),
            "train_shots": int(len(train_idx)),
            "val_shots": int(len(val_idx)),
        },
        "test": metrics,
    }
    save_json(metrics_path, payload)
    print_metric_block("MWPM test", metrics)
    print(f"metrics = {metrics_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.out or METRICS_DIR / f"mwpm_{args.raw_data.stem}.json"
    evaluate_mwpm(args.raw_data, metrics_path, seed=args.seed)


if __name__ == "__main__":
    main()
