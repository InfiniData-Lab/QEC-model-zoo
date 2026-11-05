"""Generate raw Stim detector samples and logical-flip labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qec_gnn.config import DATA_DIR, DEFAULT_CONFIG
from qec_gnn.dataset import make_split_indices
from qec_gnn.stim_utils import default_raw_data_path, get_detector_coord_array, make_surface_code_circuit
from qec_gnn.utils import ensure_parent_dir


def generate_raw_dataset(
    *,
    distance: int,
    rounds: int,
    p: float,
    shots: int,
    seed: int,
    output_path: Path,
) -> Path:
    circuit = make_surface_code_circuit(distance=distance, rounds=rounds, p=p)
    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, observables = sampler.sample(shots=shots, separate_observables=True)

    if observables.shape[1] != 1:
        raise ValueError(f"Expected one logical observable, got {observables.shape[1]}")

    labels = observables[:, 0].astype(np.uint8)
    coords = get_detector_coord_array(circuit)
    train_idx, val_idx, test_idx = make_split_indices(shots, seed=seed)
    dem = circuit.detector_error_model(decompose_errors=True)

    ensure_parent_dir(output_path)
    np.savez_compressed(
        output_path,
        detectors=detectors.astype(np.uint8),
        labels=labels,
        detector_coords=coords.astype(np.float32),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        distance=np.array(distance, dtype=np.int64),
        rounds=np.array(rounds, dtype=np.int64),
        p=np.array(p, dtype=np.float32),
        shots=np.array(shots, dtype=np.int64),
        seed=np.array(seed, dtype=np.int64),
        circuit=np.array(str(circuit)),
        detector_error_model=np.array(str(dem)),
    )

    circuit_path = output_path.with_suffix(".stim")
    dem_path = output_path.with_suffix(".dem")
    circuit_path.write_text(str(circuit))
    dem_path.write_text(str(dem))

    print("Dataset:")
    print(f"  distance = {distance}")
    print(f"  rounds = {rounds}")
    print(f"  p = {p}")
    print(f"  shots = {shots}")
    print(f"  num_detectors = {circuit.num_detectors}")
    print(f"  positive label fraction = {labels.mean():.6f}")
    print(f"  output = {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=cfg.distance)
    parser.add_argument("--rounds", type=int, default=cfg.rounds)
    parser.add_argument("--p", type=float, default=cfg.p)
    parser.add_argument("--shots", type=int, default=cfg.shots)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.out or default_raw_data_path(DATA_DIR, args.distance, args.rounds, args.p)
    generate_raw_dataset(
        distance=args.distance,
        rounds=args.rounds,
        p=args.p,
        shots=args.shots,
        seed=args.seed,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
