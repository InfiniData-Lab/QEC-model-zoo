#!/usr/bin/env python3
"""Collect GNN-vs-MWPM comparison JSON files into CSV and LaTeX tables."""

import argparse
import csv
import json
import math
import sys
from decimal import Decimal
from pathlib import Path

HEADERS = [
    "d", "p", "test shots", "Trivial LER", "GNN LER",
    "MWPM LER", "GNN gain", "MWPM gain",
]

# Expected shot counts from the planned experiment table. These are used
# ONLY for warnings; output always uses the shot count in each JSON file.
EXPECTED_TEST_SHOTS = {
    (3, 0.0005): 7500,
    (3, 0.001): 7500,
    (3, 0.005): 15000,
    (3, 0.01): 7500,
    (5, 0.005): 7500,
    (7, 0.005): 15000,
}


def gain(trivial_ler: float, decoder_ler: float) -> float:
    return math.inf if decoder_ler == 0 else trivial_ler / decoder_ler


def format_gain(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def scientific_p(p: float) -> str:
    """Convert 0.0005 to LaTeX: $5\\times 10^{-4}$."""
    value = Decimal(f"{p:.6g}")
    exponent = value.adjusted()
    coefficient = value.scaleb(-exponent).normalize()
    return rf"${coefficient}\times 10^{{{exponent}}}$"


def collect_rows(directory: Path, rounds: int) -> list[dict]:
    files = sorted(directory.rglob("compare_gnn*_vs_mwpm*.json"))
    if not files:
        raise SystemExit(f"No comparison JSON files found under {directory}")

    rows = []
    seen = {}
    for path in files:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)

        if int(data["rounds"]) != rounds:
            continue

        d = int(data["distance"])
        p = float(f"{float(data['p']):.6g}")  # remove float32 artifacts
        shots = int(data["shots"])
        trivial = float(data["positive_label_fraction"])
        gnn = float(data["gnn_error_rate"])
        mwpm = float(data["mwpm_error_rate"])

        if shots <= 0 or not all(0 <= value <= 1 for value in (trivial, gnn, mwpm)):
            raise ValueError(f"Invalid shot count or LER in {path}")

        key = (d, p)
        if key in seen:
            raise ValueError(f"Duplicate d={d}, p={p}: {seen[key]} and {path}")
        seen[key] = path

        expected = EXPECTED_TEST_SHOTS.get(key)
        if expected is not None and shots != expected:
            print(
                f"WARNING: {path.name} reports shots={shots}, "
                f"but the reference table expects {expected}. Using JSON shots.",
                file=sys.stderr,
            )

        rows.append({
            "d": d,
            "p": p,
            "test shots": shots,
            "Trivial LER": trivial,
            "GNN LER": gnn,
            "MWPM LER": mwpm,
            "GNN gain": gain(trivial, gnn),
            "MWPM gain": gain(trivial, mwpm),
        })

    if not rows:
        raise SystemExit(f"No comparison JSON files with rounds={rounds} under {directory}")
    return sorted(rows, key=lambda row: (row["d"], row["p"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADERS)
        for row in rows:
            writer.writerow([
                row["d"],
                f"{row['p']:.6g}",
                row["test shots"],
                f"{row['Trivial LER']:.4f}",
                f"{row['GNN LER']:.4f}",
                f"{row['MWPM LER']:.4f}",
                f"{format_gain(row['GNN gain'])}x",
                f"{format_gain(row['MWPM gain'])}x",
            ])


def latex_gain(value: float) -> str:
    return r"$\infty$" if math.isinf(value) else f"${format_gain(value)}\\times$"


def write_latex(path: Path, rows: list[dict], rounds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (r"\caption{Summary of GNN results on synthetic surface-code "
         rf"datasets ($r={rounds}$). MWPM (PyMatching) is evaluated on the "
         r"same test shots. Gains are relative to the trivial LER.}"),
        r"\label{tab:gnn-surface-code}",
        r"\begin{tabular}{rrrrrrrr}",
        r"\hline",
        r"$d$ & $p$ & Test shots & Trivial LER & GNN LER & MWPM LER & GNN gain & MWPM gain \\",
        r"\hline",
    ]
    for row in rows:
        test_shots = f"{row['test shots']:,}".replace(",", r"\,")
        lines.append(
            f"{row['d']} & {scientific_p(row['p'])} & {test_shots} & "
            f"{row['Trivial LER']:.4f} & {row['GNN LER']:.4f} & "
            f"{row['MWPM LER']:.4f} & "
            f"{latex_gain(row['GNN gain'])} & "
            f"{latex_gain(row['MWPM gain'])} " + r"\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--csv", type=Path, default=Path("outputs/tables/table_e.csv"))
    parser.add_argument("--latex", type=Path, default=Path("outputs/tables/table_e.tex"))
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    rows = collect_rows(args.input_dir, args.rounds)
    write_csv(args.csv, rows)
    write_latex(args.latex, rows, args.rounds)
    print(f"Wrote {len(rows)} experiment rows to {args.csv} and {args.latex}")


if __name__ == "__main__":
    main()
