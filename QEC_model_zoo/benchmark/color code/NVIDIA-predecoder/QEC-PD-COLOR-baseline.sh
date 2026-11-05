#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Color-code BASELINE benchmark: Chromobius logical error rate (no pre-decoder) swept
# over distance x physical error rate. This is the fundamental color-code benchmark and
# the baseline every color pre-decoder is measured against. Runs on CPU.
#
# Usage:
#   bash "QEC_model_zoo/benchmark/color code/NVIDIA-predecoder/QEC-PD-COLOR-baseline.sh"
#   DISTANCES="3 5 7" P_VALUES="0.004 0.008" SHOTS=20000 \
#     bash "QEC_model_zoo/benchmark/color code/NVIDIA-predecoder/QEC-PD-COLOR-baseline.sh"
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Shared pre-decoder engine (import root): benchmark/utils/predecoder
CODE_DIR="$(cd -- "${SCRIPT_DIR}/../../utils/predecoder" && pwd)"

DISTANCES="${DISTANCES:-3 5 7}"
P_VALUES="${P_VALUES:-0.004 0.008}"
SHOTS="${SHOTS:-20000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-bench_color_baseline}"
# Save results in this code family's outputs/ folder (a relative path is resolved from CODE_DIR).
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-${SCRIPT_DIR}/outputs}"
case "${RESULTS_BASE_DIR}" in
  /*) OUT_DIR="${RESULTS_BASE_DIR}/${EXPERIMENT_NAME}" ;;
  *)  OUT_DIR="${CODE_DIR}/${RESULTS_BASE_DIR}/${EXPERIMENT_NAME}" ;;
esac
# Match the QEC-PD-* run layout: logs/ (console log) + config_used/ + result JSON.
mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/config_used"
CONSOLE_LOG="${OUT_DIR}/logs/inference.console.log"

# Record the effective run parameters, mirroring config_used/ in the other runs.
cat > "${OUT_DIR}/config_used/color_baseline_params.yaml" <<YAML
code: color
workflow:
  task: baseline_benchmark
distances: [${DISTANCES// /, }]
p_values: [${P_VALUES// /, }]
shots: ${SHOTS}
global_decoder: chromobius
YAML

# Tee all console output to the log with repo-relative paths, like run_custom_inference.sh does.
# shellcheck source=../../utils/predecoder/scripts/custom/log_paths.sh
source "${CODE_DIR}/scripts/custom/log_paths.sh"
{
  echo "== Color baseline (Chromobius) benchmark =="
  echo "Distances:   ${DISTANCES}"
  echo "P values:    ${P_VALUES}"
  echo "Shots:       ${SHOTS}"
  echo "Output:      ${OUT_DIR}/color_baseline.json"
  echo "Console log: ${CONSOLE_LOG}"

  cd "${CODE_DIR}"
  PYTHONPATH="${CODE_DIR}" python -m evaluation.color_baseline_benchmark \
    --distances ${DISTANCES} \
    --p ${P_VALUES} \
    --shots "${SHOTS}" \
    --json-out "${OUT_DIR}/color_baseline.json"

  echo "== Done =="
  echo "Output: ${OUT_DIR}"
} 2>&1 | relpath_filter | tee "${CONSOLE_LOG}"
