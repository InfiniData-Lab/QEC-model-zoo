#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/../../utils/predecoder" && pwd)"
export RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-${SCRIPT_DIR}/outputs}"

# CUSTOM_CONFIG="${SCRIPT_DIR}/configs/config_2080ti_train_tabr_p003.yaml" \
# EXPERIMENT_NAME=bench_d9r9_tabr \
# TRAIN_EPOCHS=20 \
# TRAIN_SAMPLES=131072 \
# VAL_SAMPLES=512 \
# BATCH_SIZE=128 \
# ACCUMULATE_STEPS=1 \
# CLEAN_OUTPUT=1 \
# "${CODE_DIR}/scripts/custom/run_custom_train.sh"

# EXPERIMENT_NAME=bench_d9r9_tabr \
# MODEL_NAME=TabRStylePreDecoder \
# CKPT_INDEX=10 \
# "${CODE_DIR}/scripts/custom/prepare_custom_checkpoint4inference.sh"

CUSTOM_CONFIG="${SCRIPT_DIR}/configs/config_2080ti_inference_tabr_p003.yaml" \
EXPERIMENT_NAME=bench_d9r9_tabr \
DISTANCE=9 \
N_ROUNDS=9 \
TEST_SAMPLES=4096 \
TEST_BATCH_SIZE=128 \
USE_CHECKPOINT=10 \
"${CODE_DIR}/scripts/custom/run_custom_inference.sh"
