#!/usr/bin/env python3
"""Summarize repeated pre-decoder inference log blocks.

The script parses console logs containing tables with "No pre-decoder" and
"After pre-decoder" columns, then reports LER improvement factors
NoPD / AfterPD and downstream PyMatching speedup statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, TextIO


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
HEADER_RE = re.compile(
    r"\[Inference\]\s+d=(?P<distance>\d+),\s*n_rounds=(?P<n_rounds>\d+),"
    r"\s*noise_model=(?P<noise_model>\S+)"
)
TWO_VALUE_RE = re.compile(rf"(?P<raw>{FLOAT_RE})\s+(?P<res>{FLOAT_RE})\s*$")
SPEEDUP_RE = re.compile(rf"PyMatching speedup .*?:\s*(?P<speedup>{FLOAT_RE})x")


@dataclass
class InferenceRun:
    distance: int | None = None
    n_rounds: int | None = None
    noise_model: str | None = None
    latency_x_raw: float | None = None
    latency_x_res: float | None = None
    latency_z_raw: float | None = None
    latency_z_res: float | None = None
    latency_avg_raw: float | None = None
    latency_avg_res: float | None = None
    ler_x_raw: float | None = None
    ler_x_res: float | None = None
    ler_z_raw: float | None = None
    ler_z_res: float | None = None
    ler_avg_raw: float | None = None
    ler_avg_res: float | None = None
    speedup_pm: float | None = None

    @property
    def complete(self) -> bool:
        return all(
            getattr(self, field) is not None
            for field in (
                "latency_x_raw",
                "latency_x_res",
                "latency_z_raw",
                "latency_z_res",
                "latency_avg_raw",
                "latency_avg_res",
                "ler_x_raw",
                "ler_x_res",
                "ler_z_raw",
                "ler_z_res",
                "ler_avg_raw",
                "ler_avg_res",
                "speedup_pm",
            )
        )

    def ler_factor(self, basis: str) -> float:
        raw = getattr(self, f"ler_{basis}_raw")
        res = getattr(self, f"ler_{basis}_res")
        if raw is None or res is None:
            raise ValueError(f"Missing LER values for {basis}")
        if res == 0:
            # With zero observed errors in both columns, the improvement ratio is
            # undefined rather than infinite. A positive numerator and zero
            # denominator is infinite, but neither value can contribute to a
            # conventional mean/std calculation.
            return float("nan") if raw == 0 else float("inf")
        return raw / res


@dataclass
class SummaryRow:
    metric: str
    n: int
    mean: float
    std: float
    values: list[float]


def parse_two_values(line: str) -> tuple[float, float] | None:
    match = TWO_VALUE_RE.search(line)
    if not match:
        return None
    return float(match.group("raw")), float(match.group("res"))


def parse_log(lines: Iterable[str]) -> list[InferenceRun]:
    runs: list[InferenceRun] = []
    current: InferenceRun | None = None

    def finish_current() -> None:
        nonlocal current
        if current is not None and current.complete:
            runs.append(current)
        current = None

    for line in lines:
        header = HEADER_RE.search(line)
        if header:
            finish_current()
            current = InferenceRun(
                distance=int(header.group("distance")),
                n_rounds=int(header.group("n_rounds")),
                noise_model=header.group("noise_model"),
            )
            continue

        if current is None:
            continue

        if "PyMatching latency - X basis" in line:
            current.latency_x_raw, current.latency_x_res = parse_two_values(line) or (None, None)
        elif "PyMatching latency - Z basis" in line:
            current.latency_z_raw, current.latency_z_res = parse_two_values(line) or (None, None)
        elif "PyMatching latency - Avg" in line:
            current.latency_avg_raw, current.latency_avg_res = parse_two_values(line) or (None, None)
        elif "LER - X basis" in line:
            current.ler_x_raw, current.ler_x_res = parse_two_values(line) or (None, None)
        elif "LER - Z basis" in line:
            current.ler_z_raw, current.ler_z_res = parse_two_values(line) or (None, None)
        elif "LER - Avg" in line:
            current.ler_avg_raw, current.ler_avg_res = parse_two_values(line) or (None, None)
        else:
            speedup = SPEEDUP_RE.search(line)
            if speedup:
                current.speedup_pm = float(speedup.group("speedup"))

    finish_current()
    return runs


def summarize(values: list[float], metric: str) -> SummaryRow:
    # statistics.stdev() raises on mixed finite/non-finite floats on Python 3.12.
    # More importantly, 0/0 LER ratios carry no statistical information. Report
    # statistics only over defined, finite improvement factors.
    finite_values = [value for value in values if math.isfinite(value)]
    return SummaryRow(
        metric=metric,
        n=len(finite_values),
        mean=statistics.mean(finite_values) if finite_values else float("nan"),
        std=statistics.stdev(finite_values) if len(finite_values) > 1 else 0.0,
        values=finite_values,
    )


def build_summary(runs: list[InferenceRun], z_label: str) -> list[SummaryRow]:
    if not runs:
        return []
    return [
        summarize([run.ler_factor("x") for run in runs], "X-basis LER improvement factor"),
        summarize([run.ler_factor("z") for run in runs], f"{z_label}-basis LER improvement factor"),
        summarize([run.ler_factor("avg") for run in runs], "Avg LER improvement factor"),
        summarize([run.speedup_pm for run in runs if run.speedup_pm is not None], "PyMatching speedup"),
    ]


def print_table(rows: list[SummaryRow], stream: TextIO) -> None:
    if not rows:
        print("No complete inference result blocks found.", file=stream)
        return

    metric_width = max(len("Metric"), *(len(row.metric) for row in rows))
    print(f"{'Metric':<{metric_width}}  {'n':>3}  {'mean':>12}  {'std':>12}  values", file=stream)
    print(f"{'-' * metric_width}  {'-' * 3}  {'-' * 12}  {'-' * 12}  {'-' * 20}", file=stream)
    for row in rows:
        values = ", ".join(f"{value:.6g}" for value in row.values)
        print(
            f"{row.metric:<{metric_width}}  {row.n:>3d}  {row.mean:>12.6g}  "
            f"{row.std:>12.6g}  {values}",
            file=stream,
        )


def print_csv(rows: list[SummaryRow], stream: TextIO) -> None:
    writer = csv.writer(stream)
    writer.writerow(["metric", "n", "mean", "std", "values"])
    for row in rows:
        writer.writerow([row.metric, row.n, f"{row.mean:.12g}", f"{row.std:.12g}", json.dumps(row.values)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute LER improvement factors and PyMatching speedup stats from inference logs."
    )
    parser.add_argument("logs", nargs="+", type=Path, help="Path(s) to inference logs")
    parser.add_argument(
        "--format",
        choices=("table", "csv", "json"),
        default="table",
        help="Output format. Default: table",
    )
    parser.add_argument(
        "--z-label",
        default="Z",
        help="Label used for the second basis in output, e.g. Z or Y. Default: Z",
    )
    parser.add_argument(
        "--show-runs",
        action="store_true",
        help="Include parsed complete runs in JSON output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = [log for log in args.logs if not log.exists()]
    if missing:
        for log in missing:
            print(f"Log file not found: {log}", file=sys.stderr)
        return 2

    runs: list[InferenceRun] = []
    run_sources: list[str] = []
    for log in args.logs:
        with log.open("r", encoding="utf-8", errors="replace") as handle:
            parsed = parse_log(handle)
        runs.extend(parsed)
        run_sources.extend([str(log)] * len(parsed))

    rows = build_summary(runs, args.z_label)
    if args.format == "json":
        output = {"n_logs": len(args.logs), "n_runs": len(runs), "summary": [asdict(row) for row in rows]}
        if args.show_runs:
            output["runs"] = [
                {"source": source, **asdict(run)}
                for source, run in zip(run_sources, runs, strict=True)
            ]
        json.dump(output, sys.stdout, indent=2)
        print()
    elif args.format == "csv":
        print_csv(rows, sys.stdout)
    else:
        print_table(rows, sys.stdout)

    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
