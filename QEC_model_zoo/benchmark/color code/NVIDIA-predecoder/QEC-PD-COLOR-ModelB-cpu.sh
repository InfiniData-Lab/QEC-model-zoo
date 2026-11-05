#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Color-code inference for the pretrained cascade "Model B"
# (nvidia/Ising-Decoder-ColorCode-1-Fast) on CPU, with independent replicate runs.
# Example: RUNS=3 PREDECODER_INFERENCE_NUM_SAMPLES=2000 TEST_BATCH_SIZE=100 bash <this-script>
#
# Faithful pipeline: qec/color_code/detector_input.py + reference_superdense_noise +
# memory_circuit + evaluation/logical_error_rate_color.py (vendored from NVIDIA/Ising-
# Decoding), so Model B receives the exact 4-channel input it was trained on and reproduces
# its real before/after LER. Verified at d=5, p=0.003: PD+Chromobius ~7e-3 vs Chromobius
# baseline ~1.0e-2, ~83% detector-density reduction.
#
# CPU note: distance defaults to 5 because the superdense circuit + Chromobius at d=13 is
# heavy on CPU (~minutes/basis). For the full trained d=13 sweep use the GPU script
# (QEC-PD-COLOR-ModelB-gpu.sh).
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

# Force CPU (fp32 load path). Empty CUDA_VISIBLE_DEVICES is honored by run_custom_inference.sh.
export CUDA_VISIBLE_DEVICES=""
export GPUS="1"
export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
export PREDECODER_DIRECT_CONFIG=1
export PREDECODER_INFERENCE_NUM_WORKERS="${PREDECODER_INFERENCE_NUM_WORKERS:-0}"

# Save results in this code family's outputs/ folder.
export RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-${SCRIPT_DIR}/outputs}"

# Test parameters go through PREDECODER_TEST_* (the public config rejects a `test:` block).
export PREDECODER_TEST_P_ERROR="${PREDECODER_TEST_P_ERROR:-0.003}"
export PREDECODER_TEST_MEAS_BASIS="${PREDECODER_TEST_MEAS_BASIS:-Z}"

RUNS="${RUNS:-6}"
# Match the Surface CPU launcher's sample-count variable; retain legacy aliases.
TEST_SAMPLES="${PREDECODER_INFERENCE_NUM_SAMPLES:-${TEST_SAMPLES:-${SHOTS:-4096}}}"



TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-400}"
DISTANCE="${DISTANCE:-5}"
N_ROUNDS="${N_ROUNDS:-5}"
BASE_NAME="${EXPERIMENT_NAME:-bench_color_modelB_cpu_d${DISTANCE}}"
for PARAM in RUNS TEST_SAMPLES TEST_BATCH_SIZE DISTANCE N_ROUNDS; do
  VALUE="${!PARAM}"
  if ! [[ "$VALUE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: $PARAM must be a positive integer (got '$VALUE')." >&2
    exit 1
  fi
done
case "$BASE_NAME" in
  ""|.|..|*/*) echo "ERROR: EXPERIMENT_NAME must be a directory name, not a path." >&2; exit 1 ;;
esac
case "$RESULTS_BASE_DIR" in
  /*) RESULTS_ABS="$RESULTS_BASE_DIR" ;;
  *) RESULTS_ABS="${CODE_DIR}/${RESULTS_BASE_DIR}" ;;
esac
# Use one output root for runner logs and checkpoints/config metadata.
export PREDECODER_BASE_OUTPUT_DIR="$RESULTS_ABS"

# Match Surface: run directories sit directly under outputs/.
mkdir -p "$RESULTS_ABS"
export RESULTS_BASE_DIR="$RESULTS_ABS"
export PREDECODER_BASE_OUTPUT_DIR="$RESULTS_ABS"
echo "Color Model B: ${RUNS} CPU runs; ${TEST_SAMPLES} shots/basis/run"
echo "Results: $RESULTS_ABS"

declare -a LOGS=()
for ((i=1; i<=RUNS; i++)); do
  NAME="${BASE_NAME}_run${i}"
  # The runner appends logs; clear this run's log as the Surface launcher does.
  mkdir -p "${RESULTS_ABS}/${NAME}/logs"
  : > "${RESULTS_ABS}/${NAME}/logs/inference.console.log"
  echo "===== run ${i}/${RUNS}: ${NAME} ====="
  CUSTOM_CONFIG="${CUSTOM_CONFIG:-${SCRIPT_DIR}/configs/config_color_inference_modelB.yaml}" \
  EXPERIMENT_NAME="$NAME" \
  DISTANCE="$DISTANCE" N_ROUNDS="$N_ROUNDS" \
  TEST_SAMPLES="$TEST_SAMPLES" TEST_BATCH_SIZE="$TEST_BATCH_SIZE" \
  USE_CHECKPOINT="${USE_CHECKPOINT:-1}" \
  bash "${CODE_DIR}/scripts/custom/run_custom_inference.sh"
  LOGS+=("${RESULTS_ABS}/${NAME}/logs/inference.console.log")
done

python "${CODE_DIR}/analysis/color_predecoder_log_stats.py" \
  "${LOGS[@]}" --output-dir "${RESULTS_ABS}/${BASE_NAME}_summary"
