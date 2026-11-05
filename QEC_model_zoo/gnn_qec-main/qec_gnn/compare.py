"""Print and save a GNN-vs-MWPM metrics comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qec_gnn.config import METRICS_DIR
from qec_gnn.utils import load_json, save_json


def _get_test_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("test", payload)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def compare_metrics(gnn_path: Path, mwpm_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    gnn = load_json(gnn_path)
    mwpm = load_json(mwpm_path)
    gnn_metrics = _get_test_metrics(gnn)
    mwpm_metrics = _get_test_metrics(mwpm)
    gnn_config = gnn.get("config", {})
    mwpm_config = mwpm.get("config", {})

    config = {
        "distance": gnn_config.get("distance", mwpm_config.get("distance")),
        "rounds": gnn_config.get("rounds", mwpm_config.get("rounds")),
        "p": gnn_config.get("p", mwpm_config.get("p")),
        "shots": gnn_config.get("shots", mwpm_config.get("shots")),
        "max_nodes": gnn_config.get("max_nodes"),
        "graph_type": gnn_config.get("graph_type", "unknown"),
        "node_features": gnn_config.get("node_features", "unknown"),
    }
    summary = {
        **config,
        "gnn_accuracy": gnn_metrics.get("accuracy"),
        "gnn_error_rate": gnn_metrics.get("error_rate"),
        "gnn_balanced_accuracy": gnn_metrics.get("balanced_accuracy"),
        "mwpm_accuracy": mwpm_metrics.get("accuracy"),
        "mwpm_error_rate": mwpm_metrics.get("error_rate"),
        "mwpm_balanced_accuracy": mwpm_metrics.get("balanced_accuracy"),
        "positive_label_fraction": gnn_metrics.get("positive_label_fraction"),
        "gnn_metrics_path": str(gnn_path),
        "mwpm_metrics_path": str(mwpm_path),
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key} = {_fmt(value)}")

    print("")
    print("Metrics:")
    print(f"  GNN accuracy = {_fmt(summary['gnn_accuracy'])}")
    print(f"  GNN error rate = {_fmt(summary['gnn_error_rate'])}")
    print(f"  GNN balanced accuracy = {_fmt(summary['gnn_balanced_accuracy'])}")
    print(f"  MWPM accuracy = {_fmt(summary['mwpm_accuracy'])}")
    print(f"  MWPM error rate = {_fmt(summary['mwpm_error_rate'])}")
    print(f"  MWPM balanced accuracy = {_fmt(summary['mwpm_balanced_accuracy'])}")
    print(f"  class balance = {_fmt(summary['positive_label_fraction'])}")

    if output_path is not None:
        save_json(output_path, summary)
        print(f"summary = {output_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn", type=Path, required=True)
    parser.add_argument("--mwpm", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.out or METRICS_DIR / f"compare_{args.gnn.stem}_vs_{args.mwpm.stem}.json"
    compare_metrics(args.gnn, args.mwpm, output_path)


if __name__ == "__main__":
    main()
