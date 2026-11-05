#!/usr/bin/env python3
"""Extract Surface average LER and PyMatching speedup into one CSV row per run.

By default, reads bench_surface_fast_cpu_run1 through run6. Only the result
before '== Done ==' is used, excluding the runner's repeated summary lines.
Appended logs containing multiple evaluations are rejected as ambiguous.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
LER = re.compile(rf"^\s*LER\s*-\s*Avg\s*:\s*({NUMBER})\s+({NUMBER})\s*$", re.M)
SPEEDUP = re.compile(
    rf"^\s*PyMatching speedup\s*\(Avg X/Z\)\s*:\s*({NUMBER})\s*[x×]\s*$", re.M
)
FIELDS = ("run", "ler_avg_no_predecoder", "ler_avg_after_predecoder",
          "pymatching_speedup_avg_xz", "source")


def extract(path: Path, run: int, outputs_dir: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    # Summary echoes omit the separator before the runner header.
    if len(re.findall(r"^={10,}\r?\nCustom inference / LER runner\s*$", text, re.M)) > 1:
        raise ValueError(f"Multiple evaluations in {path}; use a separate log per run")
    body = text.split("== Done ==", 1)[0]
    lers, speedups = LER.findall(body), SPEEDUP.findall(body)
    if len(lers) != 1 or len(speedups) != 1:
        raise ValueError(f"Expected one average LER and speedup in {path}; "
                         f"found {len(lers)} and {len(speedups)}")
    return dict(zip(FIELDS, (run, *lers[0], speedups[0],
                            path.relative_to(outputs_dir).as_posix())))


def main() -> None:
    default_outputs = (Path(__file__).resolve().parents[3] /
                       "surface code/NVIDIA-predecoder/outputs")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=default_outputs)
    parser.add_argument("--prefix", default="bench_surface_fast_cpu")
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--output", type=Path, help="Default: OUTPUTS_DIR/PREFIX_results.csv")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    try:
        rows = [extract(args.outputs_dir / f"{args.prefix}_run{i}" /
                        "logs/inference.console.log", i, args.outputs_dir)
                for i in range(1, args.runs + 1)]
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    output = args.output or args.outputs_dir / f"{args.prefix}_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} runs to {output}")


if __name__ == "__main__":
    main()
