#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Surface-code inference for the pretrained CNN "Model 1 / Fast"
# (nvidia/Ising-Decoder-SurfaceCode-1-Fast, R=9) on CPU (fp32) — runs inference MULTIPLE
# times to produce replicate runs for statistics (mean/std/CI across runs).
#
# Each iteration is an independent, unseeded Monte-Carlo evaluation (before/after LER +
# PyMatching speedup) written to its own run dir. After the loop, the repo's own
# analysis/predecoder_log_stats.py aggregates all runs into a mean/std table.
#
# Usage:
#   bash "QEC_model_zoo/benchmark/surface code/NVIDIA-predecoder/QEC-PD-SURFACE-Fast-cpu.sh"
#   RUNS=5 PREDECODER_INFERENCE_NUM_SAMPLES=8192 \
#     bash "QEC_model_zoo/benchmark/surface code/NVIDIA-predecoder/QEC-PD-SURFACE-Fast-cpu.sh"
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Shared pre-decoder engine (import root), and QEC_model_zoo (holds model/predecoders).
CODE_DIR="$(cd -- "${SCRIPT_DIR}/../../utils/predecoder" && pwd)"
ZOO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

CKPT="${PREDECODER_SAFETENSORS_CHECKPOINT:-${ZOO_DIR}/model/predecoders/Ising-Decoder-SurfaceCode-1-Fast/ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors}"
if [ ! -f "${CKPT}" ]; then
  echo "ERROR: surface checkpoint not found: ${CKPT}"
  echo "Download it (gated) with:"
  echo "  hf download nvidia/Ising-Decoder-SurfaceCode-1-Fast \\"
  echo "    --local-dir QEC_model_zoo/model/predecoders/Ising-Decoder-SurfaceCode-1-Fast"
  exit 1
fi
export PREDECODER_SAFETENSORS_CHECKPOINT="${CKPT}"

# Force CPU (fp32 load path). Empty CUDA_VISIBLE_DEVICES is honored by run_custom_inference.sh.
export CUDA_VISIBLE_DEVICES=""
export GPUS="1"

# Save results in this code family's outputs/ folder (a relative path is resolved from CODE_DIR).
export RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-${SCRIPT_DIR}/outputs}"
case "${RESULTS_BASE_DIR}" in
  /*) RESULTS_ABS="${RESULTS_BASE_DIR}" ;;
  *)  RESULTS_ABS="${CODE_DIR}/${RESULTS_BASE_DIR}" ;;
esac

# Shot count is read by evaluation/inference.py (surface path) from these env vars.
export PREDECODER_INFERENCE_NUM_SAMPLES="${PREDECODER_INFERENCE_NUM_SAMPLES:-4096}"
export PREDECODER_INFERENCE_LATENCY_SAMPLES="${PREDECODER_INFERENCE_LATENCY_SAMPLES:-2000}"

RUNS="${RUNS:-2}"
BASE_NAME="${EXPERIMENT_NAME:-bench_surface_fast_cpu_today}"

echo "=============================================================================="
echo "Official surface CNN (R=9) — ${RUNS} CPU inference run(s)"
echo "  checkpoint: ${CKPT##*/}"
echo "  shots/basis: ${PREDECODER_INFERENCE_NUM_SAMPLES}   latency subset: ${PREDECODER_INFERENCE_LATENCY_SAMPLES}"
echo "=============================================================================="

declare -a LOGS=()
for i in $(seq 1 "${RUNS}"); do
  name="${BASE_NAME}_run${i}"
  # run_custom_inference.sh appends to its console log. Clear a previous run
  # with the same name so stale result blocks are not aggregated again.
  mkdir -p "${RESULTS_ABS}/${name}/logs"
  : > "${RESULTS_ABS}/${name}/logs/inference.console.log"
  echo; echo "===== run ${i}/${RUNS}: ${name} ====="
  CUSTOM_CONFIG="${SCRIPT_DIR}/configs/config_surface_inference_fast.yaml" \
  EXPERIMENT_NAME="${name}" \
  DISTANCE=9 \
  N_ROUNDS=9 \
  TEST_SAMPLES="${PREDECODER_INFERENCE_NUM_SAMPLES}" \
  TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}" \
  USE_CHECKPOINT=1 \
  "${CODE_DIR}/scripts/custom/run_custom_inference.sh"
  LOGS+=("${RESULTS_ABS}/${name}/logs/inference.console.log")
done

echo; echo "=============================================================================="
echo "Aggregated statistics across ${RUNS} runs (analysis/predecoder_log_stats.py):"
echo "=============================================================================="
python3 "${CODE_DIR}/analysis/predecoder_log_stats.py" "${LOGS[@]}" --format table || \
  echo "(aggregation skipped — check individual run logs)"
