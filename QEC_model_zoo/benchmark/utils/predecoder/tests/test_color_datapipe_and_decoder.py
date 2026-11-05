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
"""Tests for the color datapipe and the global-decoder abstraction."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference

_HAS_CHROMOBIUS = importlib.util.find_spec("chromobius") is not None
_HAS_PYMATCHING = importlib.util.find_spec("pymatching") is not None


class TestFaithfulColorDatapipe(unittest.TestCase):
    """Smoke tests for the faithful datapipe (NVIDIA detector_input + superdense circuit)."""

    def test_builds_and_returns_trainx_4ch(self):
        ds = QCDataPipePreDecoder_ColorCode_inference(
            distance=3,
            n_rounds=3,
            num_samples=8,
            error_mode="circuit_level_color_code",
            p_error=0.005,
            measure_basis="Z",
        )
        self.assertEqual(len(ds), 8)
        item = ds[0]
        self.assertIn("trainX", item)
        # Faithful color input carries 4 channels (Model B's input_channels).
        self.assertEqual(item["trainX"].shape[0], 4)

    def test_rejects_wrong_error_mode(self):
        with self.assertRaises(ValueError):
            QCDataPipePreDecoder_ColorCode_inference(
                distance=3, n_rounds=3, num_samples=4,
                error_mode="circuit_level_surface_custom", p_error=0.005, measure_basis="Z",
            )


@unittest.skipUnless(_HAS_PYMATCHING, "pymatching not installed")
class TestPyMatchingDecoderWrapper(unittest.TestCase):

    def test_decodes_surface_repetition(self):
        import stim
        from evaluation.global_decoder import build_global_decoder

        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=3, rounds=3,
            after_clifford_depolarization=0.005,
        )
        dem = circ.detector_error_model(decompose_errors=True)
        dec = build_global_decoder("pymatching", dem)
        sampler = circ.compile_detector_sampler(seed=1)
        dets, obs = sampler.sample(shots=5000, separate_observables=True)
        pred = dec.decode_batch(dets.astype(np.uint8))
        ler = float(np.mean(pred.reshape(-1) != obs.reshape(-1).astype(np.uint8)))
        self.assertLess(ler, 0.2)


@unittest.skipUnless(_HAS_CHROMOBIUS, "chromobius not installed")
class TestChromobiusDecoderWrapper(unittest.TestCase):

    def test_decodes_color_memory(self):
        from qec.color_code.phenomenological_circuit import ColorCodeMemoryCircuit
        from evaluation.global_decoder import build_global_decoder

        mc = ColorCodeMemoryCircuit(5, 5, 0.008)
        dec = build_global_decoder("chromobius", mc.detector_error_model(), mc.num_observables)
        sampler = mc.stim_circuit.compile_detector_sampler(seed=2)
        dets, obs = sampler.sample(shots=10000, separate_observables=True)
        pred = dec.decode_batch(dets.astype(np.uint8))
        ler = float(np.mean(pred.reshape(-1) != obs.reshape(-1).astype(np.uint8)))
        self.assertLess(ler, 0.05)

    def test_unknown_decoder_raises(self):
        from evaluation.global_decoder import build_global_decoder
        with self.assertRaises(ValueError):
            build_global_decoder("not_a_decoder", None)


@unittest.skipUnless(_HAS_CHROMOBIUS, "chromobius not installed")
class TestColorBaselineBenchmark(unittest.TestCase):

    def test_sweep_ler_decreases_with_distance(self):
        from evaluation.color_baseline_benchmark import run_sweep

        rows = run_sweep(distances=[3, 5], ps=[0.008], shots=8000, seed=11)
        self.assertEqual(len(rows), 2)
        by_d = {r["distance"]: r["chromobius_ler"] for r in rows}
        self.assertLess(by_d[3], 0.05)
        self.assertLessEqual(by_d[5], by_d[3])
        for r in rows:
            self.assertIn("stderr", r)
            self.assertIn("num_detectors", r)


if __name__ == "__main__":
    unittest.main()
