#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:?Please set EXPERIMENT_NAME}"
MODEL_NAME="${MODEL_NAME:?Please set MODEL_NAME: MLPPreDecoder | LSTMPreDecoder | TabMStylePreDecoder | TabRStylePreDecoder}"
CKPT_INDEX="${CKPT_INDEX:-1}"

cd "$REPO_DIR"

# Same base as run_custom_train.sh / run_custom_inference.sh.
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-outputs}"
BASE_DIR="$RESULTS_BASE_DIR/$EXPERIMENT_NAME/models"
BEST_DIR="$BASE_DIR/best_model"
LOG_DIR="$RESULTS_BASE_DIR/$EXPERIMENT_NAME/logs"
CONSOLE_LOG="$LOG_DIR/prepare_checkpoint.console.log"

# shellcheck source=log_paths.sh
source "${SCRIPT_DIR}/log_paths.sh"

mkdir -p "$LOG_DIR"
if [ -z "${PREDECODER_LOGGING_ACTIVE:-}" ]; then
  export PREDECODER_LOGGING_ACTIVE=1
  status=0
  bash "$SCRIPT_PATH" "$@" 2>&1 | relpath_filter | tee -a "$CONSOLE_LOG" || status=$?
  exit "$status"
fi

echo "=============================================================================="
echo "Prepare custom checkpoint for inference"
echo "=============================================================================="
echo "Experiment:   $EXPERIMENT_NAME"
echo "Model name:   $MODEL_NAME"
echo "CKPT index:   $CKPT_INDEX"
echo "Base dir:     $BASE_DIR"
echo "Best dir:     $BEST_DIR"
echo "Console log:  $CONSOLE_LOG"
echo "=============================================================================="

SRC=""

if [ -f "$BEST_DIR/${MODEL_NAME}.0.${CKPT_INDEX}.pt" ]; then
  SRC="$BEST_DIR/${MODEL_NAME}.0.${CKPT_INDEX}.pt"
elif [ -f "$BASE_DIR/${MODEL_NAME}.0.${CKPT_INDEX}.pt" ]; then
  SRC="$BASE_DIR/${MODEL_NAME}.0.${CKPT_INDEX}.pt"
else
  echo "ERROR: cannot find custom model checkpoint."
  echo ""
  echo "Expected one of:"
  echo "  $BEST_DIR/${MODEL_NAME}.0.${CKPT_INDEX}.pt"
  echo "  $BASE_DIR/${MODEL_NAME}.0.${CKPT_INDEX}.pt"
  echo ""
  echo "Available checkpoint/model files:"
  find "$RESULTS_BASE_DIR/$EXPERIMENT_NAME" -type f | grep -Ei '\.pt$|checkpoint|model' | sort || true
  exit 1
fi

mkdir -p "$BEST_DIR"

# Some inference code searches for PreDecoderModelMemory_v1.0.<idx>.pt.
# This copy keeps the state_dict content unchanged; only the filename is made compatible.
DST_BEST="$BEST_DIR/PreDecoderModelMemory_v1.0.${CKPT_INDEX}.pt"
DST_BASE="$BASE_DIR/PreDecoderModelMemory_v1.0.${CKPT_INDEX}.pt"

cp "$SRC" "$DST_BEST"
cp "$SRC" "$DST_BASE"

echo "Copied:"
echo "  $SRC"
echo "to:"
echo "  $DST_BEST"
echo "  $DST_BASE"

echo "== Final checkpoint files =="
find "$BASE_DIR" -type f | grep -Ei '\.pt$|checkpoint|model' | sort || true
