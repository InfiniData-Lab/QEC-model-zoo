"""Stim helpers for generating surface-code detector-event data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import stim


def make_surface_code_circuit(
    distance: int = 3,
    rounds: int = 5,
    p: float = 0.005,
) -> stim.Circuit:
    """Create the rotated memory-Z surface-code circuit used by the baseline."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def get_detector_coord_array(circuit: stim.Circuit) -> np.ndarray:
    """Return detector coordinates as a dense normalized ``(num_detectors, 3)`` array."""
    coord_dict = circuit.get_detector_coordinates()
    coords = np.zeros((circuit.num_detectors, 3), dtype=np.float32)

    for detector_index in range(circuit.num_detectors):
        coord = coord_dict.get(detector_index, [])
        if len(coord) >= 1:
            coords[detector_index, 0] = coord[0]
        if len(coord) >= 2:
            coords[detector_index, 1] = coord[1]
        if len(coord) >= 3:
            coords[detector_index, 2] = coord[2]

    denom = coords.max(axis=0, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    return coords / denom


def probability_tag(p: float) -> str:
    """Format probabilities into compact filename tags, e.g. 0.005 -> p005."""
    text = f"{p:g}".replace(".", "")
    return f"p{text[1:]}" if text.startswith("0") else f"p{text}"


def default_raw_data_path(
    data_dir: Path,
    distance: int,
    rounds: int,
    p: float,
) -> Path:
    return data_dir / f"raw_d{distance}_r{rounds}_{probability_tag(p)}.npz"


def default_graph_data_path(
    data_dir: Path,
    distance: int,
    rounds: int,
    p: float,
) -> Path:
    return data_dir / f"graphs_d{distance}_r{rounds}_{probability_tag(p)}.npz"
