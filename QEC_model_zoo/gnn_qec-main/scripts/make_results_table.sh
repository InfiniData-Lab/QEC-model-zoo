#!/usr/bin/env bash
set -euo pipefail

# Navigate to the project root
cd "$(dirname "$0")/.."

# Create output directory
mkdir -p outputs/tables

# Generate CSV and LaTeX tables
python scripts/make_results_table.py \
    --input-dir outputs/metrics \
    --rounds 5 \
    --csv outputs/tables/table_e.csv \
    --latex outputs/tables/table_e.tex

echo "Results tables generated successfully!"