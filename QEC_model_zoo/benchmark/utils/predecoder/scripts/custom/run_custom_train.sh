#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Custom model training runner for Ising-Decoding
#
# This runner is for custom models only:
#   custom_model_id=90: MLPPreDecoder
#   custom_model_id=91: LSTMPreDecoder
#   custom_model_id=92: TabMStylePreDecoder
#   custom_model_id=93: TabRStylePreDecoder
#
# It does NOT use code/scripts/local_run.sh.
# It uses code/workflows/run_custom_models.py.
#
# Usage example (from QEC_model_zoo/benchmark):
#
#   CUSTOM_CONFIG="$PWD/surface code/NVIDIA-predecoder/configs/config_2080ti_train_tabm_p003.yaml" \
#   RESULTS_BASE_DIR="$PWD/surface code/NVIDIA-predecoder/outputs" \
#   EXPERIMENT_NAME=bench_tabm_d9r9_e10_s1024_bs64 \
#   TRAIN_EPOCHS=10 \
#   TRAIN_SAMPLES=1024 \
#   VAL_SAMPLES=512 \
#   BATCH_SIZE=64 \
#   ACCUMULATE_STEPS=1 \
#   CLEAN_OUTPUT=1 \
#   bash utils/predecoder/scripts/custom/run_custom_train.sh
#
# Optional for LSTM:
#
#   PREDECODER_OPTIMIZER_TYPE=AdamW \
#   PREDECODER_LR=1e-4 \
#   ...
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
# Default to the conda env already activated in this shell ($CONDA_DEFAULT_ENV);
# fall back to base only if none is active. Set CONDA_ENV=<name> to force a specific env.
CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-base}}"

CUSTOM_CONFIG="${CUSTOM_CONFIG:?Set CUSTOM_CONFIG to a config yaml (see benchmark/*/NVIDIA-predecoder/configs)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-custom_mlp_smoke_bs16}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-1024}"
VAL_SAMPLES="${VAL_SAMPLES:-1024}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATE_STEPS="${ACCUMULATE_STEPS:-1}"

# Set CLEAN_OUTPUT=1 for benchmark runs to avoid old checkpoints / best loss.
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"

cd "$REPO_DIR"

# All run artifacts (logs, config_used, model checkpoints) land under
# RESULTS_BASE_DIR/EXPERIMENT_NAME. The code-family launchers set it to their own
# outputs/ folder; a relative path is resolved from this engine directory.
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-outputs}"
export PREDECODER_BASE_OUTPUT_DIR="${PREDECODER_BASE_OUTPUT_DIR:-$RESULTS_BASE_DIR}"
OUTPUT_DIR="$RESULTS_BASE_DIR/$EXPERIMENT_NAME"

# Clean before opening the console log, otherwise CLEAN_OUTPUT=1 deletes the active log.
if [ "$CLEAN_OUTPUT" = "1" ] && [ -z "${PREDECODER_LOGGING_ACTIVE:-}" ]; then
  echo "== Cleaning old output directory =="
  rm -rf "$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR/logs"
CONSOLE_LOG="$OUTPUT_DIR/logs/train.console.log"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUTPUT_DIR/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

# shellcheck source=log_paths.sh
source "${SCRIPT_DIR}/log_paths.sh"

# Save all terminal output to console log without Bash process substitution,
# which is not available in some restricted /dev/fd environments.
# Logs keep repo-relative paths, including the Hydra job log Python writes to OUTPUT_DIR.
if [ -z "${PREDECODER_LOGGING_ACTIVE:-}" ]; then
  export PREDECODER_LOGGING_ACTIVE=1
  status=0
  bash "$SCRIPT_PATH" "$@" 2>&1 | relpath_filter | tee -a "$CONSOLE_LOG" || status=$?
  relpath_logs "$OUTPUT_DIR"
  exit "$status"
fi

echo "=============================================================================="
echo "Custom training runner"
echo "=============================================================================="
echo "Repo:              $REPO_DIR"
echo "Conda env:         $CONDA_ENV"
echo "Custom config:     $CUSTOM_CONFIG"
echo "Experiment name:   $EXPERIMENT_NAME"
echo "Train epochs:      $TRAIN_EPOCHS"
echo "Train samples:     $TRAIN_SAMPLES"
echo "Val samples:       $VAL_SAMPLES"
echo "Batch size:        $BATCH_SIZE"
echo "Accumulate steps:  $ACCUMULATE_STEPS"
echo "Clean output:      $CLEAN_OUTPUT"
echo "Console log:       $CONSOLE_LOG"
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
  echo "Create the custom runner first."
  exit 1
fi

echo "== Python =="
which python
python --version

echo "== Check CUDA / GPU =="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

echo "== Check custom hook =="
grep -n "custom_model_id\|build_custom_model" model/factory.py || {
  echo "ERROR: custom hook not found in model/factory.py"
  exit 1
}

echo "== Check custom model files =="
# Custom tensor models (MLP/LSTM/TabM/TabR) live in model/tensor_models.py; builders in model/factory.py
find model/tensor_models.py model/factory.py -maxdepth 0 -type f -print || {
  echo "ERROR: custom model files not found (model/tensor_models.py, model/factory.py)."
  exit 1
}

echo "== Check unsupported fields in custom config =="
if grep -n "^val:\|^train:\|^batch_schedule:" "$CUSTOM_CONFIG"; then
  echo "ERROR: custom config contains unsupported public fields."
  echo "Move val/train/batch_schedule to runtime env variables instead."
  exit 1
fi

# ===== Runtime environment =====
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GPUS="${GPUS:-1}"
export TORCH_COMPILE=0
export PREDECODER_INFERENCE_NUM_WORKERS="${PREDECODER_INFERENCE_NUM_WORKERS:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Runtime overrides consumed by code/workflows/run_custom_models.py
export PREDECODER_TRAIN_EPOCHS="$TRAIN_EPOCHS"
export PREDECODER_TRAIN_SAMPLES="$TRAIN_SAMPLES"
export PREDECODER_VAL_SAMPLES="$VAL_SAMPLES"
export PREDECODER_BATCH_INITIAL="$BATCH_SIZE"
export PREDECODER_BATCH_FINAL="$BATCH_SIZE"
export PREDECODER_ACCUMULATE_STEPS="$ACCUMULATE_STEPS"
export PREDECODER_VALIDATION_LER=false
export PREDECODER_TORCH_COMPILE=false

# Optional overrides, useful for LSTM stability.
# These are only effective if run_custom_models.py reads them.
export PREDECODER_LR="${PREDECODER_LR:-}"
export PREDECODER_OPTIMIZER_TYPE="${PREDECODER_OPTIMIZER_TYPE:-}"

echo "== Runtime env overrides =="
echo "PREDECODER_TRAIN_EPOCHS=$PREDECODER_TRAIN_EPOCHS"
echo "PREDECODER_TRAIN_SAMPLES=$PREDECODER_TRAIN_SAMPLES"
echo "PREDECODER_VAL_SAMPLES=$PREDECODER_VAL_SAMPLES"
echo "PREDECODER_BATCH_INITIAL=$PREDECODER_BATCH_INITIAL"
echo "PREDECODER_BATCH_FINAL=$PREDECODER_BATCH_FINAL"
echo "PREDECODER_ACCUMULATE_STEPS=$PREDECODER_ACCUMULATE_STEPS"
echo "PREDECODER_VALIDATION_LER=$PREDECODER_VALIDATION_LER"
echo "PREDECODER_TORCH_COMPILE=$PREDECODER_TORCH_COMPILE"
echo "PREDECODER_LR=${PREDECODER_LR:-<unset>}"
echo "PREDECODER_OPTIMIZER_TYPE=${PREDECODER_OPTIMIZER_TYPE:-<unset>}"

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
cp "$CUSTOM_CONFIG" "$OUTPUT_DIR/config_used/custom_config.yaml"
cp conf/config_public.yaml "$OUTPUT_DIR/config_used/config_public_effective.yaml"

echo "== Effective custom config =="
grep -n "model_id\|custom_model_id\|custom_in_channels\|custom_out_channels\|custom_hidden_dim\|custom_depth\|custom_model_width\|custom_num_heads\|custom_head_hidden_dim\|custom_embed_dim\|custom_memory_size\|custom_top_k\|custom_mlp_ratio\|workflow\|task\|distance\|n_rounds\|validation_ler" \
  conf/config_public.yaml || true

echo "== nvidia-smi before training =="
nvidia-smi || true

# ===== Start training =====
echo "== Start custom training =="
HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}" \
python workflows/run_custom_models.py \
  --config-name config_public \
  workflow.task=train \
  +exp_tag="$EXPERIMENT_NAME" \
  hydra.run.dir="$OUTPUT_DIR" \
  hydra.output_subdir=null \
  hydra.job.chdir=false \
  ++load_checkpoint=False

echo "== Done =="
echo "Output: $OUTPUT_DIR"
echo "Console log: $CONSOLE_LOG"

echo "== Checkpoints / model files =="
find "$OUTPUT_DIR" -type f | grep -Ei '\.pt$|\.pth$|checkpoint|model|state|weight|pkl' | sort || true

echo "== Important log lines =="
if [ -f "$OUTPUT_DIR/run.log" ]; then
  LOG_FILE="$OUTPUT_DIR/run.log"
else
  LOG_FILE="$CONSOLE_LOG"
fi

echo "Using log file: $LOG_FILE"

grep -Ei "MLPPreDecoder|LSTMPreDecoder|TabMStylePreDecoder|TabRStylePreDecoder|CustomModels|Effective batch|epoch|loss|vloss|save|Saved|checkpoint|out of memory|error|traceback|NaN|Inf|DebugLion" \
  "$LOG_FILE" | tail -n 250 || true

echo "== nvidia-smi after training =="
nvidia-smi || true
