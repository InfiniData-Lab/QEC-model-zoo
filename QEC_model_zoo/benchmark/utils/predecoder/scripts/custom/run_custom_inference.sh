#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Custom model inference / LER runner for Ising-Decoding
#
# custom_model_id=90: MLPPreDecoder
# custom_model_id=91: LSTMPreDecoder
# custom_model_id=92: TabMStylePreDecoder
# custom_model_id=93: TabRStylePreDecoder
#
# Usually called by the code-family launchers in
# benchmark/{surface code,color code}/NVIDIA-predecoder/. Usage example
# (from QEC_model_zoo/benchmark):
#
#   CUSTOM_CONFIG="$PWD/surface code/NVIDIA-predecoder/configs/config_2080ti_inference_tabm_p003.yaml" \
#   RESULTS_BASE_DIR="$PWD/surface code/NVIDIA-predecoder/outputs" \
#   EXPERIMENT_NAME=bench_tabm_d9r9_e10_s1024_bs64 \
#   DISTANCE=9 \
#   N_ROUNDS=9 \
#   TEST_SAMPLES=1024 \
#   TEST_BATCH_SIZE=32 \
#   USE_CHECKPOINT=10 \
#   bash utils/predecoder/scripts/custom/run_custom_inference.sh
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
# Default to the conda env already activated in this shell ($CONDA_DEFAULT_ENV);
# fall back to base only if none is active. Set CONDA_ENV=<name> to force a specific env.
CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-base}}"

CUSTOM_CONFIG="${CUSTOM_CONFIG:?Set CUSTOM_CONFIG to a config yaml (see benchmark/*/NVIDIA-predecoder/configs)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-custom_mlp_smoke_bs16}"

DISTANCE="${DISTANCE:-9}"
N_ROUNDS="${N_ROUNDS:-9}"

TEST_SAMPLES="${TEST_SAMPLES:-1024}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-32}"

# Use checkpoint index, e.g. 10 means *.0.10.pt.
# -1 usually means latest / default in repo logic, but explicit is safer.
USE_CHECKPOINT="${USE_CHECKPOINT:-1}"

cd "$REPO_DIR"

# All run artifacts (logs, config_used, model checkpoints) land under
# RESULTS_BASE_DIR/EXPERIMENT_NAME. The code-family launchers set it to their own
# outputs/ folder; a relative path is resolved from this engine directory.
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-outputs}"
export PREDECODER_BASE_OUTPUT_DIR="${PREDECODER_BASE_OUTPUT_DIR:-$RESULTS_BASE_DIR}"
OUTPUT_DIR="$RESULTS_BASE_DIR/$EXPERIMENT_NAME"
mkdir -p "$OUTPUT_DIR/logs"
CONSOLE_LOG="$OUTPUT_DIR/logs/inference.console.log"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUTPUT_DIR/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

# shellcheck source=log_paths.sh
source "${SCRIPT_DIR}/log_paths.sh"

if [ -z "${PREDECODER_LOGGING_ACTIVE:-}" ]; then
  export PREDECODER_LOGGING_ACTIVE=1
  # Logs keep repo-relative paths, including the Hydra job log Python writes to OUTPUT_DIR.
  status=0
  bash "$SCRIPT_PATH" "$@" 2>&1 | relpath_filter | tee -a "$CONSOLE_LOG" || status=$?
  relpath_logs "$OUTPUT_DIR"
  exit "$status"
fi

echo "=============================================================================="
echo "Custom inference / LER runner"
echo "=============================================================================="
echo "Repo:             $REPO_DIR"
echo "Conda env:        $CONDA_ENV"
echo "Custom config:    $CUSTOM_CONFIG"
echo "Experiment name:  $EXPERIMENT_NAME"
echo "Distance:         $DISTANCE"
echo "N rounds:         $N_ROUNDS"
echo "Test samples:     $TEST_SAMPLES"
echo "Test batch size:  $TEST_BATCH_SIZE"
echo "Use checkpoint:   $USE_CHECKPOINT"
echo "Console log:      $CONSOLE_LOG"
echo "=============================================================================="

# ===== Activate conda =====
# Only switch envs when CONDA_ENV differs from the one already active in this shell,
# so an env activated in your terminal is respected instead of being reset to base.
if [ "$CONDA_ENV" = "${CONDA_DEFAULT_ENV:-}" ]; then
  echo "Using already-activated conda env: ${CONDA_DEFAULT_ENV:-<none>}"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
else
  echo "WARNING: conda.sh not found. Assuming the correct env is already activated."
fi

# ===== Basic checks =====
if [ ! -f "$CUSTOM_CONFIG" ]; then
  echo "ERROR: custom config not found: $CUSTOM_CONFIG"
  echo "Available configs:"
  find conf -maxdepth 1 -type f -name "*.yaml" -print || true
  exit 1
fi

if [ ! -f "workflows/run_custom_models.py" ]; then
  echo "ERROR: workflows/run_custom_models.py not found."
  exit 1
fi

echo "== Python =="
which python
python --version

# echo "== Check CUDA / GPU =="
# python - <<'PY'
# import torch
# print("torch:", torch.__version__)
# print("torch cuda:", torch.version.cuda)
# print("cuda available:", torch.cuda.is_available())
# print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
# PY

echo "== Check custom hook =="
grep -n "custom_model_id\|build_custom_model" model/factory.py || {
  echo "ERROR: custom hook not found in model/factory.py"
  exit 1
}

# echo "== Check unsupported fields in custom config =="
# if grep -n "^val:\|^train:\|^batch_schedule:" "$CUSTOM_CONFIG"; then
#   echo "ERROR: custom config contains unsupported public fields."
#   echo "Move val/train/batch_schedule to runtime env variables instead."
#   exit 1
# fi

# echo "== Check model/checkpoint files =="
# find "outputs/$EXPERIMENT_NAME" -type f | grep -Ei '\.pt$|checkpoint|model' | sort || true

# ===== Runtime environment =====
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# Use ${VAR-default} (no colon) so an explicit empty CUDA_VISIBLE_DEVICES="" is honored
# to force CPU; only an *unset* value falls back to GPU 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export GPUS="${GPUS:-1}"
export TORCH_COMPILE=0
export PREDECODER_INFERENCE_NUM_WORKERS="${PREDECODER_INFERENCE_NUM_WORKERS:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Runtime overrides consumed by code/workflows/run_custom_models.py
export PREDECODER_DISTANCE="$DISTANCE"
export PREDECODER_N_ROUNDS="$N_ROUNDS"

export PREDECODER_TEST_DISTANCE="$DISTANCE"
export PREDECODER_TEST_N_ROUNDS="$N_ROUNDS"
export PREDECODER_TEST_SAMPLES="$TEST_SAMPLES"
export PREDECODER_TEST_BATCH_SIZE="$TEST_BATCH_SIZE"
export PREDECODER_USE_CHECKPOINT="$USE_CHECKPOINT"

export PREDECODER_TORCH_COMPILE=false
export PREDECODER_VALIDATION_LER=false

echo "== Runtime env overrides =="
echo "PREDECODER_DISTANCE=$PREDECODER_DISTANCE"
echo "PREDECODER_N_ROUNDS=$PREDECODER_N_ROUNDS"
echo "PREDECODER_TEST_DISTANCE=$PREDECODER_TEST_DISTANCE"
echo "PREDECODER_TEST_N_ROUNDS=$PREDECODER_TEST_N_ROUNDS"
echo "PREDECODER_TEST_SAMPLES=$PREDECODER_TEST_SAMPLES"
echo "PREDECODER_TEST_BATCH_SIZE=$PREDECODER_TEST_BATCH_SIZE"
echo "PREDECODER_USE_CHECKPOINT=$PREDECODER_USE_CHECKPOINT"
echo "PREDECODER_TORCH_COMPILE=$PREDECODER_TORCH_COMPILE"

# ===== Temporarily replace public config =====
BACKUP_CONFIG="$(mktemp /tmp/config_public_backup.XXXXXX.yaml)"

restore_config() {
  if [ -f "$BACKUP_CONFIG" ]; then
    cp "$BACKUP_CONFIG" conf/config_public.yaml
    rm -f "$BACKUP_CONFIG"
    echo "Restored original conf/config_public.yaml"
  fi
}
trap restore_config EXIT

echo "== Temporarily replacing conf/config_public.yaml =="
cp conf/config_public.yaml "$BACKUP_CONFIG"
cp "$CUSTOM_CONFIG" conf/config_public.yaml

mkdir -p "$OUTPUT_DIR/config_used"
cp "$CUSTOM_CONFIG" "$OUTPUT_DIR/config_used/custom_config_for_inference.yaml"
cp conf/config_public.yaml "$OUTPUT_DIR/config_used/config_public_effective_for_inference.yaml"

echo "== Effective custom config =="
grep -n "model_id\|custom_model_id\|custom_in_channels\|custom_out_channels\|custom_hidden_dim\|custom_depth\|custom_model_width\|custom_num_heads\|custom_head_hidden_dim\|custom_embed_dim\|custom_memory_size\|custom_top_k\|custom_mlp_ratio\|workflow\|task\|distance\|n_rounds\|validation_ler" \
  conf/config_public.yaml || true

# echo "== nvidia-smi before inference =="
# nvidia-smi || true

# ===== Start inference / LER =====
echo "== Start custom inference / LER =="
HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}" \
python workflows/run_custom_models.py \
  --config-name config_public \
  workflow.task=inference \
  +exp_tag="$EXPERIMENT_NAME" \
  hydra.run.dir="$OUTPUT_DIR" \
  hydra.output_subdir=null \
  hydra.job.chdir=false \
  ++load_checkpoint=True

echo "== Done =="
echo "Output: $OUTPUT_DIR"
echo "Console log: $CONSOLE_LOG"

echo "== Important inference lines =="
grep -Ei "CustomModels|Loaded|checkpoint|model|LER|logical|logical error|SDR|syndrome|density|threshold|decode|decoder|error|traceback|exception|out of memory|shape" \
  "$CONSOLE_LOG" | tail -n 300 || true

# echo "== nvidia-smi after inference =="
# nvidia-smi || true
