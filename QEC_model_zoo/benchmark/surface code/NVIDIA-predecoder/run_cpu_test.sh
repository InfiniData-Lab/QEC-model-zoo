#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# CPU inference smoke-test for the pre-decoder pipeline (no GPU required).
#
# Why this exists:
#   The production launcher (benchmark/utils/predecoder/scripts/nvidia/local_run.sh) is GPU-only:
#   it requires nvidia-smi and torch.cuda. This script runs the public inference workflow
#   (workflows/run.py) directly on CPU -- the same scenario as QEC-PD-D9R9-CNN.sh -- so the
#   pipeline can be exercised on a laptop / CI box without CUDA.
#
#   Note: only INFERENCE is CPU-runnable. training/train.py hard-raises
#   "Training requires a GPU" when CUDA is absent.
#
# Usage (from QEC_model_zoo/benchmark/surface code/NVIDIA-predecoder):
#   bash run_cpu_test.sh                                   # Fast model (default)
#   MODEL=Accurate bash run_cpu_test.sh
#   MODEL="/abs/path/to/model.pt" EXPERIMENT_NAME=cpu_test bash run_cpu_test.sh
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Shared pre-decoder engine (import root): benchmark/utils/predecoder
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../utils/predecoder" && pwd)}"
cd "${REPO_ROOT}"

MODEL="${MODEL:-Fast}"
CONFIG_NAME="${CONFIG_NAME:-config_public}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cpu_test}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- Force CPU (also overrides GPU on a CUDA host, so the test is reproducible anywhere) ---
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PREDECODER_TORCH_COMPILE=0
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

case "${MODEL}" in
  Fast)     CKPT="saved predecoder models/Ising-Decoder-SurfaceCode-1-Fast.pt" ;;
  Accurate) CKPT="saved predecoder models/Ising-Decoder-SurfaceCode-1-Accurate.pt" ;;
  *)        CKPT="${MODEL}" ;;
esac

echo "=================================================================="
echo "Pre-decoder CPU inference test"
echo "  repo root  : ${REPO_ROOT}"
echo "  python     : $("${PYTHON_BIN}" --version 2>&1)"
echo "  checkpoint : ${CKPT}"
echo "=================================================================="

# --- Preflight: torch importable, confirm CPU-only ---
if ! "${PYTHON_BIN}" - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print(f"[cpu_test] ERROR: PyTorch not importable: {e}", file=sys.stderr); sys.exit(1)
print(f"[cpu_test] torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
PY
then
  exit 1
fi

# --- Preflight: checkpoint present and not an unpulled Git LFS pointer ---
if [ ! -f "${CKPT}" ]; then
  echo "[cpu_test] ERROR: checkpoint not found: ${CKPT}" >&2
  exit 2
fi
if head -c 64 "${CKPT}" | grep -q "git-lfs"; then
  echo "[cpu_test] ERROR: '${CKPT}' is a Git LFS pointer, not real weights." >&2
  echo "[cpu_test] Pull the binaries first:  git lfs install && git lfs pull" >&2
  exit 3
fi

# --- Run public inference workflow on CPU (QEC-PD-D9R9-CNN.sh scenario) ---
export PREDECODER_BASE_OUTPUT_DIR="${PREDECODER_BASE_OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
OUT_DIR="${PREDECODER_BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"
mkdir -p "${OUT_DIR}"
echo "[cpu_test] NOTE: the public inference path runs the full LER eval"
echo "[cpu_test]       (test.num_samples is fixed at 262144 and cannot be reduced via config),"
echo "[cpu_test]       so this can take a long time on CPU. Ctrl-C to stop early."

"${PYTHON_BIN}" -u workflows/run.py \
  --config-name "${CONFIG_NAME}" \
  workflow.task=inference \
  +exp_tag="${EXPERIMENT_NAME}" \
  ++load_checkpoint=True \
  +model_checkpoint_file="${CKPT}" \
  hydra.run.dir="${OUT_DIR}"
rc=$?

if [ "${rc}" -eq 0 ]; then
  echo "CPU TEST PASSED (inference)"
else
  echo "CPU TEST FAILED (inference, rc=${rc})"
fi
exit "${rc}"
