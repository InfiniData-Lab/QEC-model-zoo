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
Chromobius-decodable triangular color-code memory circuit.

Built on top of the ported ColorCode lattice (qec/color_code/color_code.py). This is a
**phenomenological** memory experiment (per-round data depolarization + noisy stabilizer
measurement via stim MPP), NOT the full superdense circuit-level schedule from
arXiv:2607.10058. It is deliberately compact and produces a stim circuit whose detectors
carry the Chromobius basis+color annotation, so Chromobius decodes it directly.

Key facts (validated empirically at p=0.01, Z-basis memory, n_rounds=distance):
- Detector 4th coordinate = color_index (X-check) or color_index + 3 (Z-check), with
  color_index {red:0, green:1, blue:2} — the Chromobius convention.
- The DEM MUST be built with decompose_errors=False: color-code bulk errors produce
  3-symptom "rainbow triplet" hyperedges that Chromobius handles natively but that
  stim's graphlike (<=2 symptom) decomposition rejects.
- Resulting Chromobius LER falls ~4x per distance step (d3 5.4e-3, d5 1.3e-3, d7 3e-4),
  confirming correct sub-threshold decoding.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import stim

from qec.color_code.color_code import ColorCode

# Chromobius 4th-coordinate convention (see module docstring).
_COLOR_TO_INDEX = {"red": 0, "green": 1, "blue": 2}


class ColorCodeMemoryCircuit:
    """A phenomenological, Chromobius-decodable color-code memory experiment.

    Args:
        distance: odd code distance >= 3.
        n_rounds: number of stabilizer-measurement rounds (>= 1).
        p_error: scalar physical error rate applied as data depolarization and
            measurement-flip probability.
        basis: "Z" (default) or "X" memory basis.
    """

    def __init__(self, distance: int, n_rounds: int, p_error: float, basis: str = "Z"):
        self.distance = int(distance)
        self.n_rounds = max(int(n_rounds), 1)
        self.p_error = float(p_error)
        self.basis = str(basis).upper()
        if self.basis not in ("X", "Z"):
            raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")

        self.code = ColorCode(self.distance)
        cc = self.code
        nd = cc.num_data
        self.num_data = nd
        self.num_plaquettes = cc.num_plaquettes

        # Per-plaquette data-qubit support and Chromobius color index.
        self._x_support: List[List[int]] = [
            np.nonzero(cc.hx[i, :nd])[0].tolist() for i in range(cc.num_plaquettes)
        ]
        self._z_support: List[List[int]] = [
            np.nonzero(cc.hz[i, :nd])[0].tolist() for i in range(cc.num_plaquettes)
        ]
        self._color_idx: List[int] = [
            _COLOR_TO_INDEX[cc.plaquettes[i]["color"]] for i in range(cc.num_plaquettes)
        ]
        self._grid_pos: List[Tuple[int, int]] = [
            tuple(cc.plaquettes[i]["grid_pos"]) for i in range(cc.num_plaquettes)
        ]
        self._logical_qubits: List[int] = list(cc.logical_qubits)

        # (H, W) of the CNN grid embedding.
        self.grid_hw: Tuple[int, int] = (int(cc.n_rows), int(cc.n_cols))

        self.stim_circuit = self._build_circuit()

    # ---- circuit construction -------------------------------------------------
    def _mpp_round(self, circ: stim.Circuit) -> None:
        p = self.p_error
        circ.append("DEPOLARIZE1", list(range(self.num_data)), p)
        for supp in self._x_support:
            targets = []
            for j, q in enumerate(supp):
                if j:
                    targets.append(stim.target_combiner())
                targets.append(stim.target_x(q))
            circ.append("MPP", targets, p)
        for supp in self._z_support:
            targets = []
            for j, q in enumerate(supp):
                if j:
                    targets.append(stim.target_combiner())
                targets.append(stim.target_z(q))
            circ.append("MPP", targets, p)

    def _det_coords(self, plaq_idx: int, t: int, is_z: bool) -> List[float]:
        col, row = self._grid_pos[plaq_idx][1], self._grid_pos[plaq_idx][0]
        color = self._color_idx[plaq_idx] + (3 if is_z else 0)
        return [float(col), float(row), float(t), float(color)]

    def _build_circuit(self) -> stim.Circuit:
        cc = self.code
        nd = self.num_data
        nX = nZ = self.num_plaquettes
        per_round = nX + nZ
        circ = stim.Circuit()

        # Z-basis memory: reset all data qubits in Z. (X basis would use RX / MX at the
        # end; the phenomenological detectors are symmetric so we keep Z here.)
        circ.append("R", list(range(nd)))

        # Round 0: only the reset-basis stabilizers are deterministic.
        self._mpp_round(circ)
        det_basis_is_z = self.basis == "Z"
        base_offset = nX if det_basis_is_z else 0
        for i in range(self.num_plaquettes):
            rec = -per_round + base_offset + i
            circ.append("DETECTOR", [stim.target_rec(rec)], self._det_coords(i, 0, det_basis_is_z))

        # Rounds 1..T-1: compare consecutive rounds for both bases.
        for r in range(1, self.n_rounds):
            self._mpp_round(circ)
            for i in range(nX):
                circ.append(
                    "DETECTOR",
                    [stim.target_rec(-per_round + i), stim.target_rec(-2 * per_round + i)],
                    self._det_coords(i, r, False),
                )
            for i in range(nZ):
                circ.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-per_round + nX + i),
                        stim.target_rec(-2 * per_round + nX + i),
                    ],
                    self._det_coords(i, r, True),
                )

        # Final data measurement (Z basis) reconstructs Z stabilizers and the logical.
        circ.append("M", list(range(nd)))

        def data_rec(q: int) -> int:
            return -nd + q

        for i in range(nZ):
            targets = [stim.target_rec(data_rec(q)) for q in self._z_support[i]]
            last_z = -nd - per_round + nX + i
            targets.append(stim.target_rec(last_z))
            circ.append("DETECTOR", targets, self._det_coords(i, self.n_rounds, True))

        circ.append(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(data_rec(q)) for q in self._logical_qubits],
            0,
        )
        return circ

    # ---- decoding helpers -----------------------------------------------------
    def detector_error_model(self) -> stim.DetectorErrorModel:
        """DEM for Chromobius. decompose_errors=False is REQUIRED (rainbow triplets)."""
        return self.stim_circuit.detector_error_model(decompose_errors=False)

    @property
    def num_detectors(self) -> int:
        return self.stim_circuit.num_detectors

    @property
    def num_observables(self) -> int:
        return self.stim_circuit.num_observables

    def detector_grid_index(self) -> np.ndarray:
        """Flat (row*W + col) grid index for every detector, in emission order."""
        return self.detector_layout()[:, 3]

    def detector_layout(self) -> np.ndarray:
        """Per-detector layout in emission order.

        Returns an (num_detectors, 4) int64 array with columns
        [basis_is_z, round, row, flat_grid_index], where flat_grid_index = row*W + col.
        The final reconstruction round is mapped to round index n_rounds. This is the
        injective detector <-> (basis, round, grid) map used to rasterize syndrome
        volumes and to project pre-decoder predictions back onto detectors.
        """
        _, w = self.grid_hw
        rows = []
        # Round 0: reset-basis stabilizers only.
        basis_is_z0 = 1 if self.basis == "Z" else 0
        for i in range(self.num_plaquettes):
            r, c = self._grid_pos[i]
            rows.append([basis_is_z0, 0, r, r * w + c])
        # Rounds 1..T-1: X block then Z block.
        for rd in range(1, self.n_rounds):
            for basis_is_z in (0, 1):
                for i in range(self.num_plaquettes):
                    r, c = self._grid_pos[i]
                    rows.append([basis_is_z, rd, r, r * w + c])
        # Final Z reconstruction detectors.
        for i in range(self.num_plaquettes):
            r, c = self._grid_pos[i]
            rows.append([1, self.n_rounds, r, r * w + c])
        return np.asarray(rows, dtype=np.int64)


__all__ = ["ColorCodeMemoryCircuit"]
