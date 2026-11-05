# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Color-code baseline benchmark: sweep distance x physical error rate and report the
Chromobius logical error rate (no pre-decoder).

This is the fundamental color-code benchmark — the baseline every color pre-decoder is
measured against. It reuses the verified color memory circuit
(qec/color_code/memory_circuit.py) and the global-decoder abstraction
(evaluation/global_decoder.py). Runnable on CPU.

Usage:
    python -m evaluation.color_baseline_benchmark --distances 3 5 7 --p 0.008 --shots 20000
    python QEC_model_zoo/evaluation/color_baseline_benchmark.py --distances 3 5 --p 0.004 0.008
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def _ensure_path():
    here = Path(__file__).resolve().parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def run_point(distance: int, n_rounds: int, p: float, shots: int, seed: int = 0) -> Dict:
    """One (distance, p) benchmark point -> Chromobius LER + error bar."""
    _ensure_path()
    from qec.color_code.phenomenological_circuit import ColorCodeMemoryCircuit
    from evaluation.global_decoder import build_global_decoder

    mc = ColorCodeMemoryCircuit(distance, n_rounds, p)
    decoder = build_global_decoder("chromobius", mc.detector_error_model(), mc.num_observables)
    sampler = mc.stim_circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    pred = decoder.decode_batch(dets.astype(np.uint8))
    errors = int(np.sum(pred.reshape(-1) != obs.reshape(-1).astype(np.uint8)))
    ler = errors / shots if shots else float("nan")
    stderr = (ler * (1.0 - ler) / shots) ** 0.5 if shots else float("nan")
    return {
        "distance": distance,
        "n_rounds": n_rounds,
        "p": p,
        "shots": shots,
        "chromobius_errors": errors,
        "chromobius_ler": ler,
        "stderr": stderr,
        "num_detectors": mc.num_detectors,
    }


def run_sweep(distances: List[int], ps: List[float], shots: int, seed: int = 0) -> List[Dict]:
    rows = []
    for d in distances:
        for p in ps:
            rows.append(run_point(d, d, p, shots, seed=seed))
    return rows


def _format_table(rows: List[Dict]) -> str:
    lines = [f"{'distance':>8} {'n_rounds':>8} {'p':>8} {'shots':>9} {'Chromobius LER':>16} {'stderr':>10}"]
    for r in rows:
        lines.append(
            f"{r['distance']:>8} {r['n_rounds']:>8} {r['p']:>8.4f} {r['shots']:>9} "
            f"{r['chromobius_ler']:>16.6f} {r['stderr']:>10.6f}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--p", type=float, nargs="+", default=[0.008], dest="ps")
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    rows = run_sweep(args.distances, args.ps, args.shots, seed=args.seed)
    print("[Color Baseline Benchmark] Chromobius LER (no pre-decoder)")
    print(_format_table(rows))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
