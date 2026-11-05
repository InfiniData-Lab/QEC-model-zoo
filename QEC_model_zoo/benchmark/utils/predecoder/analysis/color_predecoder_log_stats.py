#!/usr/bin/env python3
"""Summarize independent Color Model B logs, grouped by measurement basis.

LER values are per round. Speedup is Chromobius decoder-only timing, not the
end-to-end neural pre-decoder pipeline. Standard deviations are across runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
FIELDS = {
    "num_shots": "shots", "n_rounds": "rounds",
    "logical_errors": "pd_errors", "chromobius_errors": "baseline_errors",
    "logical_error_rate (mean)": "pd_ler_per_round",
    "chromobius_error_rate (mean)": "baseline_ler_per_round",
}
METRICS = ("pd_ler_per_round", "baseline_ler_per_round", "ler_improvement",
           "chromobius_speedup", "density_reduction_pct")


def parse_log(path: Path) -> list[dict]:
    # Persist portable source references without local account names.
    try:
        source = str(path.resolve().relative_to(Path(__file__).resolve().parents[5]))
    except ValueError:
        source = path.name
    # The runner appends a grep of historical lines after Done. Never count it.
    text = path.read_text(encoding="utf-8", errors="replace").split("== Done ==", 1)[0]
    diagnostics: dict[str, dict] = {}
    results: dict[str, dict] = {}
    basis = None
    in_results = False
    for line in text.splitlines():
        match = re.search(r"\[Color Code LER\] Basis: ([XZ])", line)
        if match:
            basis = match[1]
            diagnostics[basis] = {}
        if basis and not in_results:
            for label, key in (("Total speedup:", "chromobius_speedup"),
                               ("Density reduction:", "density_reduction_pct")):
                match = re.search(re.escape(label) + rf"\s*({FLOAT})", line)
                if match:
                    diagnostics[basis][key] = float(match[1])
        if "Color Code Inference Results" in line:
            in_results = True
            basis = None
        if not in_results:
            continue
        match = re.fullmatch(r"\s*([XZ])-basis:\s*", line)
        if match:
            basis = match[1]
            results[basis] = {"source": source, "basis": basis,
                              **diagnostics.get(basis, {})}
        elif basis:
            for label, key in FIELDS.items():
                match = re.fullmatch(rf"\s*{re.escape(label)}:\s*({FLOAT})\s*", line)
                if match:
                    results[basis][key] = (int(match[1]) if key in
                        {"shots", "rounds", "pd_errors", "baseline_errors"} else float(match[1]))
    for result in results.values():
        if any(key not in result for key in FIELDS.values()):
            raise ValueError(f"Incomplete {result['basis']}-basis result in {path}")
        # Undefined/infinite ratios stay null and are excluded from mean/std.
        result["ler_improvement"] = (result["baseline_errors"] / result["pd_errors"]
                                     if result["pd_errors"] else None)
    if not results:
        raise ValueError(f"No complete Color inference result in {path}")
    return list(results.values())


def summarize(runs: list[dict]) -> list[dict]:
    summary = []
    for basis in sorted({run["basis"] for run in runs}):
        for metric in METRICS:
            values = [r[metric] for r in runs if r["basis"] == basis
                      and r.get(metric) is not None and math.isfinite(r[metric])]
            summary.append({"basis": basis, "metric": metric, "n": len(values),
                            "mean": statistics.mean(values) if values else None,
                            "std": statistics.stdev(values) if len(values) > 1 else None})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    runs = [run for path in args.logs for run in parse_log(path)]
    summary = summarize(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"n_runs": len(args.logs), "runs": runs, "summary": summary},
                   indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in runs for k in r)))
        writer.writeheader()
        writer.writerows(runs)
    lines = ["LER is per round; speedup is Chromobius decoder-only; std is across runs.",
             f"{'Basis':<6} {'Metric':<26} {'n':>4} {'mean':>12} {'std':>12}"]
    for row in summary:
        mean = f"{row['mean']:.6g}" if row['mean'] is not None else "N/A"
        std = f"{row['std']:.6g}" if row['std'] is not None else "N/A"
        lines.append(f"{row['basis']:<6} {row['metric']:<26} {row['n']:>4} {mean:>12} {std:>12}")
    table = "\n".join(lines) + "\n"
    (args.output_dir / "summary.txt").write_text(table, encoding="utf-8")
    print(table, end="")
    print(f"Saved summary.json, summary.txt and runs.csv in {args.output_dir}")


if __name__ == "__main__":
    main()
