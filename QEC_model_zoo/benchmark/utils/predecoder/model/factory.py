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
Factory module for creating models.

Provides ModelFactory for instantiating pre-decoder models from config.
"""


class ModelFactory:

    @staticmethod
    def create_model(cfg):
        # Custom tensor predecoders for reproduction experiments.
        # Keep cfg.model_id in 1..5 for public validation,
        # and use cfg.custom_model_id in {90,91,92,93} to select custom models.
        custom_model_id = getattr(cfg, 'custom_model_id', None)
        if custom_model_id is not None and is_custom_model_id(custom_model_id):
            return build_custom_model(cfg)

        # The cascade architecture ("Model B") is code-family-agnostic and selected by
        # model version, so it is handled before the surface/color code branch.
        model_version = getattr(getattr(cfg, "model", None), "version", None)
        if model_version == "predecoder_memory_cascade":
            from model.predecoder import PreDecoderModelMemory_Cascade
            return PreDecoderModelMemory_Cascade(cfg)

        if cfg.code == "surface":
            return ModelFactory._create_surface_model(cfg)
        elif cfg.code == "color":
            return ModelFactory._create_color_model(cfg)
        else:
            raise ValueError("Invalid model name")

    @staticmethod
    def _create_surface_model(cfg):
        if cfg.model.version == "predecoder_memory_v1":
            from model.predecoder import PreDecoderModelMemory_v1
            model = PreDecoderModelMemory_v1(cfg)
            return model
        else:
            raise ValueError(f"Invalid model version: {cfg.model.version}")

    @staticmethod
    def _create_color_model(cfg):
        version = cfg.model.version
        if version == "predecoder_memory_cascade":
            from model.predecoder import PreDecoderModelMemory_Cascade
            return PreDecoderModelMemory_Cascade(cfg)
        if version == "predecoder_memory_v1":
            from model.predecoder import PreDecoderModelMemory_v1
            return PreDecoderModelMemory_v1(cfg)
        raise ValueError(f"Invalid color model version: {version}")


from .tensor_models import (
    MLPPreDecoder,
    LSTMPreDecoder,
    TabMStylePreDecoder,
    TabRStylePreDecoder,
    infer_tensor_shapes_from_cfg,
)


CUSTOM_MODEL_IDS = {
    90: "mlp",
    91: "lstm",
    92: "tabm_style",
    93: "tabr_style",
}


def is_custom_model_id(model_id):
    return int(model_id) in CUSTOM_MODEL_IDS


def build_custom_model(cfg):
    model_id = int(getattr(cfg, "custom_model_id", cfg.model_id))

    input_shape, output_shape, c_in, c_out = infer_tensor_shapes_from_cfg(cfg)

    if model_id == 90:
        return MLPPreDecoder(
            input_shape=input_shape,
            output_shape=output_shape,
            hidden_dim=int(getattr(cfg, "custom_hidden_dim", 512)),
            depth=int(getattr(cfg, "custom_depth", 3)),
            dropout=float(getattr(cfg, "custom_dropout", 0.1)),
            debug_shapes=bool(getattr(cfg, "custom_debug_shapes", True)),
        )

    if model_id == 91:
        return LSTMPreDecoder(
            in_channels=c_in,
            out_channels=c_out,
            hidden_dim=int(getattr(cfg, "custom_hidden_dim", 128)),
            num_layers=int(getattr(cfg, "custom_num_layers", 2)),
            dropout=float(getattr(cfg, "custom_dropout", 0.1)),
            bidirectional=bool(getattr(cfg, "custom_bidirectional", True)),
            debug_shapes=bool(getattr(cfg, "custom_debug_shapes", True)),
        )

    # if model_id == 92:
    #     return TabMStylePreDecoder(
    #         input_shape=input_shape,
    #         output_shape=output_shape,
    #         width=int(getattr(cfg, "custom_model_width", 256)),
    #         depth=int(getattr(cfg, "custom_depth", 3)),
    #         num_heads=int(getattr(cfg, "custom_num_heads", 8)),
    #         dropout=float(getattr(cfg, "custom_dropout", 0.1)),
    #         debug_shapes=bool(getattr(cfg, "custom_debug_shapes", True)),
    #     )
    if model_id == 92:
        return TabMStylePreDecoder(
            input_shape=input_shape,
            output_shape=output_shape,
            width=int(getattr(cfg, "custom_model_width", 256)),
            depth=int(getattr(cfg, "custom_depth", 3)),
            num_heads=int(getattr(cfg, "custom_num_heads", 8)),
            dropout=float(getattr(cfg, "custom_dropout", 0.1)),
            head_hidden_dim=int(
                getattr(
                    cfg,
                    "custom_head_hidden_dim",
                    getattr(cfg, "custom_model_width", 256),
                )
            ),
            debug_shapes=bool(getattr(cfg, "custom_debug_shapes", True)),
        )

    if model_id == 93:
        return TabRStylePreDecoder(
            in_channels=c_in,
            out_channels=c_out,
            embed_dim=int(getattr(cfg, "custom_embed_dim", 128)),
            memory_size=int(getattr(cfg, "custom_memory_size", 256)),
            top_k=int(getattr(cfg, "custom_top_k", 16)),
            dropout=float(getattr(cfg, "custom_dropout", 0.1)),
            mlp_ratio=float(getattr(cfg, "custom_mlp_ratio", 2.0)),
            debug_shapes=bool(getattr(cfg, "custom_debug_shapes", True)),
        )