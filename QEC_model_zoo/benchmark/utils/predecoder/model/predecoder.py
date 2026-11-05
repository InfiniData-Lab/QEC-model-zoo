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

## Model architecture with CNN networks for pre-decoders

import math

import torch
import torch.nn as nn
from types import SimpleNamespace


def _normalize_3d_param(value, name):
    if isinstance(value, int):
        return (value, value, value)
    try:
        values = list(value)
    except TypeError:
        values = None
    if values is not None and len(values) == 3:
        return tuple(int(v) for v in values)
    raise ValueError(f"{name} must be an int or a length-3 sequence, got {value!r}")


class CascadeBottleneckBlock3D(nn.Module):
    """Bottleneck residual block for the cascade Conv3D pre-decoder ("Model B").

    Ported from NVIDIA/Ising-Decoding (Apache-2.0). Can change channel count so the
    pre-decoder can enter a fixed hidden width directly from the raw input.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, bottleneck_ratio=4, num_blocks=1):
        super().__init__()

        in_channels = int(in_channels)
        out_channels = int(out_channels)
        bottleneck_ratio = int(bottleneck_ratio)
        num_blocks = int(num_blocks)
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"in_channels and out_channels must be positive, got {(in_channels, out_channels)}"
            )
        if bottleneck_ratio <= 0:
            raise ValueError(f"bottleneck_ratio must be positive, got {bottleneck_ratio}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        bottleneck_channels = max(1, out_channels // bottleneck_ratio)
        kernel_size = _normalize_3d_param(kernel_size, "kernel_size")
        padding = tuple(k // 2 for k in kernel_size)

        self.residual_scale = 1.0 / math.sqrt(2.0 * num_blocks)
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.reduce = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, bottleneck_channels, kernel_size=1),
        )
        self.message_passing = nn.Sequential(
            nn.BatchNorm3d(bottleneck_channels),
            nn.SiLU(),
            nn.Conv3d(
                bottleneck_channels, bottleneck_channels, kernel_size=kernel_size, padding=padding
            ),
        )
        self.restore = nn.Sequential(
            nn.BatchNorm3d(bottleneck_channels),
            nn.SiLU(),
            nn.Conv3d(bottleneck_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        identity = self.skip(x) * self.residual_scale
        out = self.reduce(x)
        out = self.message_passing(out)
        out = self.restore(out)
        return out + identity


class PreDecoderModelMemory_Cascade(nn.Module):
    """Cascade Conv3D pre-decoder ("Model B"). Ported from NVIDIA/Ising-Decoding.

    Plain Conv3d stem lifting input_channels -> embed_dim, then (num_blocks - 1)
    bottleneck residual blocks, then a 1x1x1 output projection. Matches the published
    color-code cascade checkpoint (~2.94M params).
    """

    def __init__(self, cfg):
        super().__init__()
        self.distance = cfg.distance
        self.n_rounds = cfg.n_rounds

        input_channels = int(cfg.model.input_channels)
        out_channels = int(cfg.model.out_channels)
        embed_dim = int(cfg.model.embed_dim)
        num_blocks = int(cfg.model.num_blocks)
        bottleneck_ratio = int(getattr(cfg.model, "bottleneck_ratio", 4))
        kernel_size = getattr(cfg.model, "kernel_size", 3)

        if embed_dim <= 0:
            raise ValueError(f"model.embed_dim must be positive, got {embed_dim}")
        if num_blocks <= 0:
            raise ValueError(f"model.num_blocks must be positive, got {num_blocks}")

        if bool(getattr(cfg.model, "plain_stem", False)):
            _pad = kernel_size // 2 if isinstance(kernel_size, int) else tuple(
                k // 2 for k in kernel_size
            )
            self.input_block = nn.Conv3d(
                input_channels, embed_dim, kernel_size=kernel_size, padding=_pad
            )
        else:
            self.input_block = CascadeBottleneckBlock3D(
                in_channels=input_channels,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                bottleneck_ratio=bottleneck_ratio,
                num_blocks=num_blocks,
            )
        self.blocks = nn.ModuleList(
            [
                CascadeBottleneckBlock3D(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    kernel_size=kernel_size,
                    bottleneck_ratio=bottleneck_ratio,
                    num_blocks=num_blocks,
                ) for _ in range(max(0, num_blocks - 1))
            ]
        )
        self.output_projection = nn.Conv3d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.input_block(x)
        for block in self.blocks:
            x = block(x)
        return self.output_projection(x)


class ResidualBlock3D(nn.Module):

    def __init__(self, channels, kernel_sizes, activation):
        """
        channels: List of 4 ints = [in1, out1, out2, out3]
        kernel_sizes: List of 3 ints (or tuples) = k1, k2, k3
        """
        super(ResidualBlock3D, self).__init__()
        self.activation = activation()  # instantiate once

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                channels[0], channels[1], kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2
            ),
            nn.BatchNorm3d(channels[1]),
            self.activation  # instance
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                channels[1], channels[2], kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2
            ),
            nn.BatchNorm3d(channels[2]),
            self.activation  # instance
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                channels[2], channels[3], kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2
            ), nn.BatchNorm3d(channels[3])
        )

        self.skip = nn.Identity()
        if channels[0] != channels[3]:
            self.skip = nn.Conv3d(channels[0], channels[3], kernel_size=1)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return self.activation(out + identity)


class PreDecoderModelMemory_v1(nn.Module):

    def __init__(self, cfg):
        super(PreDecoderModelMemory_v1, self).__init__()

        self.distance = cfg.distance
        self.n_rounds = cfg.n_rounds
        self.dropout_p = cfg.model.dropout_p
        self.activation_fn = self._get_activation(cfg.model.activation)

        filters = cfg.model.num_filters
        kernel_sizes = cfg.model.kernel_size

        assert len(filters) == len(kernel_sizes), \
            "Mismatch: num_filters and kernel_size must be the same length."

        # === Configurable input and output channels ===
        input_channels = cfg.model.input_channels
        out_channels = cfg.model.out_channels
        assert filters[-1] == out_channels, \
            f"The last element of num_filters must match the configured out_channels ({out_channels}), but got {filters[-1]}"

        layers = []
        in_channels = input_channels  # 4 input channels from trainX

        for i in range(len(filters)):
            layers.append(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=filters[i],
                    kernel_size=kernel_sizes[i],
                    padding=kernel_sizes[i] // 2  # keeps same shape (optional)
                )
            )
            if i < len(filters) - 1:  # last layer should not have dropout or activation
                layers.append(nn.Dropout3d(p=self.dropout_p))
                layers.append(self.activation_fn)
            in_channels = filters[i]

        self.net = nn.Sequential(*layers)

    def _get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU(approximate='tanh')
        elif name == "leakyrelu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        return self.net(x)  # x: (B, 4, T, D, D)


# === Define a mock config using SimpleNamespace ===
def get_mock_config():
    cfg = SimpleNamespace()
    cfg.model = SimpleNamespace()
    cfg.distance = 11
    cfg.n_rounds = 3
    cfg.model.dropout_p = 0.1
    cfg.model.activation = 'relu'
    cfg.model.input_channels = 4
    cfg.model.out_channels = 2
    cfg.model.num_filters = [8, 4, 2]
    cfg.model.kernel_size = [3, 3, 3]
    return cfg


# === Run the test ===
def test_model():
    cfg = get_mock_config()
    model = PreDecoderModelMemory_v1(cfg)

    B, C_in, T, D = 2, cfg.model.input_channels, cfg.n_rounds, cfg.distance
    input_tensor = torch.randn(B, C_in, T, D, D)

    output = model(input_tensor)

    expected_shape = (B, cfg.model.out_channels, T, D, D)
    assert output.shape == expected_shape, \
        f"Output shape mismatch: expected {expected_shape}, got {output.shape}"

    print("✅ Model test passed. Output shape:", output.shape)


if __name__ == "__main__":
    test_model()
