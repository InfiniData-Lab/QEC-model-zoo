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
"""Tests for qec.code_family (code-family registry) and shape wiring."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from qec.code_family import get_code_family, valid_code_families
from model.tensor_models import infer_tensor_shapes_from_cfg


class TestCodeFamilyRegistry(unittest.TestCase):

    def test_surface_and_color_registered(self):
        self.assertIn("surface", valid_code_families())
        self.assertIn("color", valid_code_families())

    def test_surface_shape_is_square(self):
        self.assertEqual(get_code_family("surface").spatial_shape(9), (9, 9))

    def test_surface_partition_aliases_surface(self):
        self.assertEqual(get_code_family("surface_partition").name, "surface")

    def test_decoder_names(self):
        self.assertEqual(get_code_family("surface").global_decoder_name, "pymatching")
        self.assertEqual(get_code_family("color").global_decoder_name, "chromobius")

    def test_unknown_family_raises(self):
        with self.assertRaises(ValueError):
            get_code_family("toric_nonsense")


class TestInferTensorShapes(unittest.TestCase):

    def _cfg(self, code, distance=9, n_rounds=9):
        return SimpleNamespace(
            code=code,
            distance=distance,
            n_rounds=n_rounds,
            model=SimpleNamespace(input_channels=4, out_channels=4),
        )

    def test_surface_shape_unchanged(self):
        input_shape, output_shape, c_in, c_out = infer_tensor_shapes_from_cfg(self._cfg("surface"))
        self.assertEqual(input_shape, (4, 9, 9, 9))
        self.assertEqual(output_shape, (4, 9, 9, 9))
        self.assertEqual((c_in, c_out), (4, 4))

    def test_default_code_is_surface(self):
        cfg = SimpleNamespace(distance=5, n_rounds=5, model=SimpleNamespace(input_channels=2, out_channels=2))
        input_shape, _, _, _ = infer_tensor_shapes_from_cfg(cfg)
        self.assertEqual(input_shape, (2, 5, 5, 5))

    def test_custom_hw_override_wins(self):
        cfg = self._cfg("color")
        cfg.custom_height = 7
        cfg.custom_width_spatial = 5
        input_shape, output_shape, _, _ = infer_tensor_shapes_from_cfg(cfg)
        self.assertEqual(input_shape[2:], (7, 5))
        self.assertEqual(output_shape[2:], (7, 5))


if __name__ == "__main__":
    unittest.main()
