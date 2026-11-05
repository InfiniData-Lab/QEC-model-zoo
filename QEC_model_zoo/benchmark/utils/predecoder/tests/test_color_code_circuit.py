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
"""Tests for the ported color-code lattice and the Chromobius-decodable memory circuit."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from qec.color_code.color_code import ColorCode
from qec.color_code.phenomenological_circuit import ColorCodeMemoryCircuit

_HAS_CHROMOBIUS = importlib.util.find_spec("chromobius") is not None

# Reference counts from color_code.py docstring / formulas.
_REFERENCE = {3: (7, 3), 5: (19, 9), 7: (37, 18)}


class TestColorCodeLattice(unittest.TestCase):

    def test_reference_counts(self):
        for d, (num_data, num_plaq) in _REFERENCE.items():
            cc = ColorCode(d)
            self.assertEqual(cc.num_data, num_data, f"num_data d={d}")
            self.assertEqual(cc.num_plaquettes, num_plaq, f"num_plaquettes d={d}")

    def test_three_colors_present(self):
        cc = ColorCode(5)
        colors = {p["color"] for p in cc.plaquettes}
        self.assertEqual(colors, {"red", "green", "blue"})

    def test_even_distance_rejected(self):
        with self.assertRaises(ValueError):
            ColorCode(4)


class TestColorCodeMemoryCircuit(unittest.TestCase):

    def test_circuit_builds_and_layout_matches_detectors(self):
        for d in (3, 5, 7):
            mc = ColorCodeMemoryCircuit(d, d, 0.01)
            self.assertEqual(mc.num_observables, 1)
            layout = mc.detector_layout()
            self.assertEqual(layout.shape[0], mc.num_detectors)
            # 4th detector coord (chromobius annotation) is always 0..5.
            self.assertEqual(mc.grid_hw, (cc_rows(d), d))

    def test_detector_grid_index_in_range(self):
        mc = ColorCodeMemoryCircuit(5, 5, 0.01)
        h, w = mc.grid_hw
        idx = mc.detector_grid_index()
        self.assertTrue(np.all(idx >= 0))
        self.assertTrue(np.all(idx < h * w))


def cc_rows(d: int) -> int:
    return d + (d - 1) // 2


@unittest.skipUnless(_HAS_CHROMOBIUS, "chromobius not installed")
class TestChromobiusDecodes(unittest.TestCase):

    def test_ler_decreases_with_distance(self):
        import chromobius

        lers = {}
        for d in (3, 5):
            mc = ColorCodeMemoryCircuit(d, d, 0.008)
            dem = mc.detector_error_model()  # decompose_errors=False internally
            dec = chromobius.compile_decoder_for_dem(dem)
            sampler = mc.stim_circuit.compile_detector_sampler(seed=1234)
            dets, obs = sampler.sample(shots=20000, separate_observables=True)
            packed = np.packbits(dets.astype(np.uint8), axis=1, bitorder="little")
            pred = np.asarray(
                dec.predict_obs_flips_from_dets_bit_packed(packed)
            ).reshape(-1) & 1
            lers[d] = float(np.mean(pred != obs.reshape(-1).astype(np.uint8)))

        # Sub-threshold color code: LER should drop with distance and be small.
        self.assertLess(lers[3], 0.05)
        self.assertLess(lers[5], lers[3])


if __name__ == "__main__":
    unittest.main()
