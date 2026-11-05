#!/usr/bin/env bash

INFERENCE_CONFIG_FOLDER="QEC_model_zoo/benchmark/color code/full_decoder/configs/inference"

for file_path in "$INFERENCE_CONFIG_FOLDER"/*; do
    # Skip directories
    [ -f "$file_path" ] || continue

    echo "Running experiment: $file_path"

    python QEC_model_zoo/main.py \
        --experiment "$file_path"
done