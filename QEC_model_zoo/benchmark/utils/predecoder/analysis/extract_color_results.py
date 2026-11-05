#!/usr/bin/env python3
"""Extract Color LER before/after pre-decoding and Chromobius speedup to CSV.

Defaults to runs 1–6, with one row per run and measurement basis. LER is per
round; speedup measures Chromobius decoding alone, excluding neural inference.
Repeated summary echoes are ignored by the shared Color log parser.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from color_predecoder_log_stats import parse_log

FIELDS = ("run", "basis", "shots", "rounds", "ler_no_predecoder_per_round",
          "ler_after_predecoder_per_round", "chromobius_speedup", "source")


def extract(path: Path, run: int, outputs_dir: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if len(re.findall(r"^={10,}\r?\nCustom inference / LER runner\s*$", text, re.M)) > 1:
        raise ValueError(f"Multiple evaluations in {path}; use a separate log per run")
    rows = []
    for result in parse_log(path):
        if "chromobius_speedup" not in result:
            raise ValueError(f"Missing Chromobius speedup for {result['basis']} in {path}")
        rows.append(dict(zip(FIELDS, (
            run, result["basis"], result["shots"], result["rounds"],
            result["baseline_ler_per_round"], result["pd_ler_per_round"],
            result["chromobius_speedup"], path.relative_to(outputs_dir).as_posix(),
        ))))
    return rows


def main() -> None:
    default_outputs = (Path(__file__).resolve().parents[3] /
                       "color code/NVIDIA-predecoder/outputs")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=default_outputs)
    parser.add_argument("--prefix", default="bench_color_modelB_cpu_d5")
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--output", type=Path, help="Default: OUTPUTS_DIR/PREFIX_results.csv")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    try:
        rows = [row for i in range(1, args.runs + 1)
                for row in extract(args.outputs_dir / f"{args.prefix}_run{i}" /
                                   "logs/inference.console.log", i, args.outputs_dir)]
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    output = args.output or args.outputs_dir / f"{args.prefix}_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows from {args.runs} runs to {output}")


if __name__ == "__main__":
    main()
