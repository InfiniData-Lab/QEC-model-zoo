#!/usr/bin/env bash

TRAIN_CONFIG_FOLDER="QEC_model_zoo/benchmark/surface code/full_decoder/configs/train"

for file_path in "$TRAIN_CONFIG_FOLDER"/*; do
    # Skip directories
    [ -f "$file_path" ] || continue

    echo "Running experiment: $file_path"

    python QEC_model_zoo/main.py \
        --experiment "$file_path"
done