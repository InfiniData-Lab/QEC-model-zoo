#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Croissant (MLCommons 1.1) metadata files from datasets/dataset_catalog.yaml.

Usage:
    python QEC_model_zoo/scripts/generate_croissant.py
    python QEC_model_zoo/scripts/generate_croissant.py --dataset-id 4 --dataset-id 2
    python QEC_model_zoo/scripts/generate_croissant.py --output-dir datasets/croissant
"""

import argparse
import sys
from pathlib import Path

QEC_MODEL_ZOO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = QEC_MODEL_ZOO_DIR.parent
sys.path.insert(0, str(QEC_MODEL_ZOO_DIR))

from utils.croissant import generate_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "datasets" / "dataset_catalog.yaml",
        help="Path to dataset_catalog.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "croissant",
        help="Directory to write Croissant JSON files into",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        action="append",
        dest="dataset_ids",
        help="Restrict generation to this dataset id (repeatable). Default: all ids in the catalog.",
    )
    args = parser.parse_args()

    written = generate_all(args.catalog, args.output_dir, args.dataset_ids)
    for path in written:
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
