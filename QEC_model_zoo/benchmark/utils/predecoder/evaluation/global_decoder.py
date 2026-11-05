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
Global-decoder abstraction: a uniform interface over the classical decoder that sits
behind the pre-decoder.

- PyMatching (MWPM) for surface codes.
- Chromobius for color codes.

Both take a stim DetectorErrorModel and decode a batch of detection events into predicted
observable flips. The pre-decoder pipeline selects one via the code family
(qec.code_family.get_code_family(cfg.code).global_decoder_name).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class GlobalDecoder(Protocol):
    name: str

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        """detection_events: (N, num_detectors) uint8 -> (N, num_observables) uint8."""
        ...


class PyMatchingDecoder:
    """MWPM decoder (surface codes). Requires a graphlike (decomposed) DEM."""

    name = "pymatching"

    def __init__(self, dem):
        import pymatching

        self._matcher = pymatching.Matching.from_detector_error_model(dem)

    @classmethod
    def from_dem(cls, dem) -> "PyMatchingDecoder":
        return cls(dem)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        dets = np.ascontiguousarray(detection_events, dtype=np.uint8)
        preds = self._matcher.decode_batch(dets)
        return np.asarray(preds, dtype=np.uint8).reshape(dets.shape[0], -1)


class ChromobiusDecoder:
    """Chromobius decoder (color codes).

    The DEM must be built with decompose_errors=False and carry the 4th basis+color
    detector coordinate (see qec.color_code.memory_circuit).
    """

    name = "chromobius"

    def __init__(self, dem, num_observables: int):
        import chromobius

        self._decoder = chromobius.compile_decoder_for_dem(dem)
        self._num_observables = int(num_observables)

    @classmethod
    def from_dem(cls, dem, num_observables: int) -> "ChromobiusDecoder":
        return cls(dem, num_observables)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        dets = np.ascontiguousarray(detection_events, dtype=np.uint8)
        packed = np.packbits(dets, axis=1, bitorder="little")
        preds = self._decoder.predict_obs_flips_from_dets_bit_packed(packed)
        preds = np.asarray(preds, dtype=np.uint8).reshape(dets.shape[0], -1)
        # Chromobius returns a bit-packed observable byte; keep only the low observable bits.
        if preds.shape[1] * 8 >= self._num_observables:
            unpacked = np.unpackbits(preds, axis=1, bitorder="little")
            return unpacked[:, : self._num_observables]
        return preds & 1


def build_global_decoder(name: str, dem, num_observables: int = 1) -> GlobalDecoder:
    """Construct a decoder by code-family decoder name ('pymatching' | 'chromobius')."""
    key = str(name).lower()
    if key == "pymatching":
        return PyMatchingDecoder.from_dem(dem)
    if key == "chromobius":
        return ChromobiusDecoder.from_dem(dem, num_observables)
    raise ValueError(f"Unknown global decoder {name!r}. Valid: 'pymatching', 'chromobius'.")


__all__ = [
    "GlobalDecoder",
    "PyMatchingDecoder",
    "ChromobiusDecoder",
    "build_global_decoder",
]
