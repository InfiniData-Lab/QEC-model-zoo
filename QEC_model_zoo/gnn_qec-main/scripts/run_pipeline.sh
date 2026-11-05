#!/usr/bin/env bash
# End-to-end pipeline: generate data -> build graphs -> tiny-overfit check ->
# full training -> MWPM baseline -> comparison. Mirrors the README's
# "End-to-End Run" commands, parameterized via env vars so it can be reused
# across configurations without editing the script.
set -euo pipefail

DISTANCE="${DISTANCE:-3}"
ROUNDS="${ROUNDS:-5}"
P="${P:-0.005}"
SHOTS="${SHOTS:-30000}"
MAX_NODES="${MAX_NODES:-64}"
K_NEIGHBORS="${K_NEIGHBORS:-4}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"

TAG="d${DISTANCE}_r${ROUNDS}_p$(echo "${P}" | tr -d '.')"

RAW="data/generated/raw_${TAG}.npz"
GRAPHS="data/generated/graphs_${TAG}.npz"
MODEL="outputs/models/gnn_${TAG}.keras"
GNN_METRICS="outputs/metrics/gnn_graphs_${TAG}.json"
MWPM_METRICS="outputs/metrics/mwpm_${TAG}.json"

echo "== generate_data =="
python -m qec_gnn.generate_data \
  --distance "${DISTANCE}" --rounds "${ROUNDS}" --p "${P}" --shots "${SHOTS}" \
  --out "${RAW}"

echo "== graph_builder =="
python -m qec_gnn.graph_builder \
  --input "${RAW}" --max-nodes "${MAX_NODES}" --k "${K_NEIGHBORS}" \
  --out "${GRAPHS}"

echo "== train_gnn (tiny-overfit sanity check) =="
python -m qec_gnn.train_gnn --data "${GRAPHS}" --tiny-overfit

echo "== train_gnn (full training) =="
python -m qec_gnn.train_gnn \
  --data "${GRAPHS}" --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" \
  --out "${MODEL}"

echo "== eval_mwpm =="
python -m qec_gnn.eval_mwpm --raw-data "${RAW}" --out "${MWPM_METRICS}"

echo "== compare =="
python -m qec_gnn.compare --gnn "${GNN_METRICS}" --mwpm "${MWPM_METRICS}"
