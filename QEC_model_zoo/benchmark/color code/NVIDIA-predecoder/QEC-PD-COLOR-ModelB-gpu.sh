#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Color-code inference for the pretrained cascade "Model B"
# (nvidia/Ising-Decoder-ColorCode-1-Fast) on GPU (fp16) — the full trained-distance run.
#
# Same faithful pipeline as the CPU script (qec/color_code/detector_input.py + superdense
# memory circuit + evaluation/logical_error_rate_color.py), but runs at the model's trained
# window (d=13, n_rounds=13) with fp16 weights on CUDA and a production-size shot count.
# The SafeTensors loader keeps fp16 on CUDA (fast) and only upcasts to fp32 on CPU.
#
# Usage (single GPU):
#   bash "QEC_model_zoo/benchmark/color code/NVIDIA-predecoder/QEC-PD-COLOR-ModelB-gpu.sh"
# Pick a GPU / sweep p:
#   CUDA_VISIBLE_DEVICES=1 PREDECODER_TEST_P_ERROR=0.001 \
#     bash "QEC_model_zoo/benchmark/color code/NVIDIA-predecoder/QEC-PD-COLOR-ModelB-gpu.sh"
# On DAIC, submit inside the container via an sbatch wrapper (see root gpu_script*.sh),
# calling this script with WORKFLOW=inference.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Shared pre-decoder engine (import root), and QEC_model_zoo (holds model/predecoders).
CODE_DIR="$(cd -- "${SCRIPT_DIR}/../../utils/predecoder" && pwd)"
ZOO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

CKPT="${PREDECODER_SAFETENSORS_CHECKPOINT:-${ZOO_DIR}/model/predecoders/Ising-Decoder-ColorCode-1-Fast/ising_decoder_color_code_1_fast_r13_v1.0.400_fp16.safetensors}"
if [ ! -f "${CKPT}" ]; then
  echo "ERROR: color checkpoint not found: ${CKPT}"
  echo "Download it (gated) with:"
  echo "  hf download nvidia/Ising-Decoder-ColorCode-1-Fast \\"
  echo "    --local-dir QEC_model_zoo/model/predecoders/Ising-Decoder-ColorCode-1-Fast"
  exit 1
fi
export PREDECODER_SAFETENSORS_CHECKPOINT="${CKPT}"

# GPU device selection (defaults to GPU 0). Set CUDA_VISIBLE_DEVICES to pick a device.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GPUS="${GPUS:-1}"

# Save results in this code family's outputs/ folder.
export RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-${SCRIPT_DIR}/outputs}"

# Test parameters go through PREDECODER_TEST_* (the public config rejects a `test:` block).
export PREDECODER_TEST_P_ERROR="${PREDECODER_TEST_P_ERROR:-0.002}"
export PREDECODER_TEST_MEAS_BASIS="${PREDECODER_TEST_MEAS_BASIS:-both}"

CUSTOM_CONFIG="${CUSTOM_CONFIG:-${SCRIPT_DIR}/configs/config_color_inference_modelB.yaml}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-bench_color_modelB_gpu_d13}" \
DISTANCE="${DISTANCE:-13}" \
N_ROUNDS="${N_ROUNDS:-13}" \
# TEST_SAMPLES="${TEST_SAMPLES:-262144}" \
# TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-2048}" \
USE_CHECKPOINT="${USE_CHECKPOINT:-1}" \
"${CODE_DIR}/scripts/custom/run_custom_inference.sh"
