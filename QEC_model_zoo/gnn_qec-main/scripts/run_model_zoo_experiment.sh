#!/usr/bin/env bash
set -euo pipefail

# Global settings
ROUNDS=5
MAX_NODES=64
K=4
EPOCHS=30
BATCH_SIZE=128

mkdir -p data/generated outputs/models outputs/metrics

# Format: distance, p, filename suffix, test shots
configs=(
    "3 0.0005 00005 50000"
    "3 0.001  0001  50000"
    "3 0.005  0005  100000"
    "3 0.01   001   50000"
    "5 0.005  0005  50000"
    "7 0.005  0005  100000"
)

for config in "${configs[@]}"; do
    read -r d p p_tag test_shots <<< "$config"

    tag="d${d}_r${ROUNDS}_p${p_tag}"

    raw="data/generated/raw_${tag}.npz"
    graphs="data/generated/graphs_${tag}.npz"
    model="outputs/models/gnn_${tag}.keras"
    mwpm="outputs/metrics/mwpm_${tag}.json"
    gnn="outputs/metrics/gnn_graphs_${tag}.json"

    echo "========================================"
    echo "Running: d=$d, p=$p, rounds=$ROUNDS"
    echo "========================================"

    # 1. Generate training data
    python -m qec_gnn.generate_data \
        --distance "$d" \
        --rounds "$ROUNDS" \
        --p "$p" \
        --shots "$test_shots" \
        --out "$raw"

    # 2. Build graphs
    python -m qec_gnn.graph_builder \
        --input "$raw" \
        --max-nodes "$MAX_NODES" \
        --k "$K" \
        --out "$graphs"

    # 3. Optional debugging: tiny overfit
    if [[ "${RUN_TINY_OVERFIT:-0}" == "1" ]]; then
        python -m qec_gnn.train_gnn \
            --data "$graphs" \
            --tiny-overfit
    fi

    # 4. Train GNN
    python -m qec_gnn.train_gnn \
        --data "$graphs" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --out "$model"

    # 5. Evaluate MWPM
    python -m qec_gnn.eval_mwpm \
        --raw-data "$raw" \
        --out "$mwpm"

    # 6. Compare GNN and MWPM
    if [[ -f "$gnn" ]]; then
        python -m qec_gnn.compare \
            --gnn "$gnn" \
            --mwpm "$mwpm"
    else
        echo "Missing GNN metrics: $gnn"
        echo "Run GNN evaluation before comparing."
    fi

    echo "Finished: $tag"
done

echo "All experiments completed."
