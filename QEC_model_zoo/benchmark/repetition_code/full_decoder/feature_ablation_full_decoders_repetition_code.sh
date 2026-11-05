#!/usr/bin/env bash

TRAIN_CONFIG_FOLDER="QEC_model_zoo/benchmark/repetition_code/full_decoder/configs/ablation"

# Execute all experiment configs in the folder
for file_path in "$TRAIN_CONFIG_FOLDER"/*; do
    # Skip directories
    [ -f "$file_path" ] || continue

    echo "Running experiment: $file_path"

    python QEC_model_zoo/main.py \
        --experiment "$file_path"
done
