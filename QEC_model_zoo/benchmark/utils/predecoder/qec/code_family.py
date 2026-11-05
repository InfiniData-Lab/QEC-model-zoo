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
Code-family registry for the pre-decoder subsystem.

`cfg.code` selects a quantum error-correcting code family (e.g. "surface", "color").
Historically the subsystem hardcoded "surface" in two places (config defaults and the
run dispatcher). This registry centralizes the per-family differences that the
otherwise code-agnostic pipeline needs:

- spatial_shape(distance): the (H, W) of the syndrome tensor fed to the pre-decoder.
  Surface codes use a square (D, D) grid; other families may differ.
- global_decoder_name: which classical global decoder sits behind the pre-decoder
  ("pymatching" for surface codes, "chromobius" for color codes).

The model architectures themselves (CNN / MLP / LSTM / TabM / TabR) are family-agnostic
and are selected separately via model_id / custom_model_id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple


@dataclass(frozen=True)
class CodeFamilySpec:
    name: str
    # distance -> (height, width) of the per-round syndrome grid.
    spatial_shape_fn: Callable[[int], Tuple[int, int]]
    # Classical global decoder that the pre-decoder feeds into.
    global_decoder_name: str
    # Model versions valid for the built-in (non-custom) model path.
    valid_model_versions: Tuple[str, ...] = field(default=("predecoder_memory_v1",))

    def spatial_shape(self, distance: int) -> Tuple[int, int]:
        d = int(distance)
        if d <= 0:
            raise ValueError(f"distance must be a positive integer, got {distance!r}")
        h, w = self.spatial_shape_fn(d)
        return int(h), int(w)


def _square_shape(distance: int) -> Tuple[int, int]:
    return distance, distance


_CODE_FAMILIES: Dict[str, CodeFamilySpec] = {}


def register(spec: CodeFamilySpec) -> None:
    _CODE_FAMILIES[spec.name] = spec


def get_code_family(name: str) -> CodeFamilySpec:
    """Return the CodeFamilySpec for `name`, or raise with the list of valid names."""
    key = str(name)
    # "surface_partition" is a historical alias handled by the surface family.
    if key == "surface_partition":
        key = "surface"
    if key not in _CODE_FAMILIES:
        raise ValueError(
            f"Unknown code family {name!r}. Valid families: {sorted(_CODE_FAMILIES)}"
        )
    return _CODE_FAMILIES[key]


def valid_code_families() -> Tuple[str, ...]:
    return tuple(sorted(_CODE_FAMILIES))


register(
    CodeFamilySpec(
        name="surface",
        spatial_shape_fn=_square_shape,
        global_decoder_name="pymatching",
    )
)

# Color codes: syndrome detectors are rasterized onto a grid derived from stim detector
# coordinates at datapipe-construction time. The bounding box is not known from distance
# alone, so we fall back to a square (D, D) placeholder here; the color datapipe passes
# explicit custom_height / custom_width_spatial overrides based on the real coord extents.
register(
    CodeFamilySpec(
        name="color",
        spatial_shape_fn=_square_shape,
        global_decoder_name="chromobius",
    )
)
