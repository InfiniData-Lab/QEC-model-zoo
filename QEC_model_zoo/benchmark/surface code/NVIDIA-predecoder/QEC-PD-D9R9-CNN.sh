#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/../../utils/predecoder" && pwd)"
LOCAL_RUN="${CODE_DIR}/scripts/nvidia/local_run.sh"

# Checkpoints/outputs and logs go to this code family's outputs/ folder.
export PREDECODER_BASE_OUTPUT_DIR="${PREDECODER_BASE_OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
export PREDECODER_LOG_BASE_DIR="${PREDECODER_LOG_BASE_DIR:-${SCRIPT_DIR}/outputs/logs}"

# official CNN model training
EXPERIMENT_NAME=official_train_d9r9_single_gpu_2080ti \
PREDECODER_TRAIN_EPOCHS=10 \
PREDECODER_TRAIN_SAMPLES=4096 \
WORKFLOW=train \
bash "${LOCAL_RUN}"

# official pre-trained CNN model inference
export EXPERIMENT_NAME=official_train_d9r9_single_gpu_2080ti
export CUDA_VISIBLE_DEVICES=0
export GPUS=1
export TORCH_COMPILE=0
export PREDECODER_INFERENCE_NUM_WORKERS=0

WORKFLOW=inference bash "${LOCAL_RUN}"
