"""Build padded active-defect graph tensors from Stim detector samples."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import stim

from qec_gnn.config import DATA_DIR, DEFAULT_CONFIG
from qec_gnn.dataset import load_split_indices, make_split_indices
from qec_gnn.stim_utils import default_graph_data_path, get_detector_coord_array, make_surface_code_circuit
from qec_gnn.utils import ensure_parent_dir


def build_knn_adjacency(node_coords: np.ndarray, k: int = 4) -> np.ndarray:
    """Construct a symmetric KNN adjacency matrix with self-loops."""
    n = len(node_coords)
    adj = np.eye(n, dtype=np.float32)
    if n <= 1:
        return adj

    diff = node_coords[:, None, :] - node_coords[None, :, :]
    dist = np.sum(diff * diff, axis=-1)
    num_neighbors = min(k, n - 1)

    for i in range(n):
        nearest = np.argsort(dist[i])[1 : num_neighbors + 1]
        adj[i, nearest] = 1.0
        adj[nearest, i] = 1.0

    return adj


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """Apply symmetric GCN normalization."""
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    degree = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(degree + 1e-8, -0.5)
    return (d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]).astype(np.float32)


def detector_error_model_adjacency(
    dem: stim.DetectorErrorModel,
    *,
    num_detectors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build detector-detector topology from graphlike DEM error mechanisms.

    Single-detector error mechanisms are treated as boundary connections and
    recorded as a node feature instead of adding explicit boundary nodes.
    """
    adj = np.zeros((num_detectors, num_detectors), dtype=np.float32)
    edge_weights = np.zeros((num_detectors, num_detectors), dtype=np.float32)
    boundary_counts = np.zeros((num_detectors,), dtype=np.float32)

    for instruction in dem.flattened():
        if instruction.type != "error":
            continue

        probability = float(instruction.args_copy()[0])
        for target_group in instruction.target_groups():
            detector_ids = [
                target.val
                for target in target_group
                if target.is_relative_detector_id()
            ]

            if len(detector_ids) == 1:
                boundary_counts[detector_ids[0]] += probability
                continue

            for i, j in combinations(sorted(set(detector_ids)), 2):
                adj[i, j] = 1.0
                adj[j, i] = 1.0
                edge_weights[i, j] += probability
                edge_weights[j, i] += probability

    degrees = np.sum(adj, axis=1).astype(np.float32)
    return adj, degrees, boundary_counts


def build_graph_from_shot(
    detector_sample: np.ndarray,
    coords: np.ndarray,
    *,
    max_nodes: int = 64,
    k: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """Convert one shot into padded ``(X, A, mask)`` graph tensors."""
    num_detectors = len(detector_sample)
    active_indices = np.flatnonzero(detector_sample)
    truncated = len(active_indices) > max_nodes

    if len(active_indices) == 0:
        node_features = np.zeros((1, 4), dtype=np.float32)
        adjacency = np.eye(1, dtype=np.float32)
    else:
        active_indices = active_indices[:max_nodes]
        node_coords = coords[active_indices]
        node_features = np.concatenate(
            [
                node_coords.astype(np.float32),
                (active_indices[:, None] / max(num_detectors, 1)).astype(np.float32),
            ],
            axis=1,
        )
        adjacency = build_knn_adjacency(node_coords, k=k)

    adjacency = normalize_adjacency(adjacency)
    n = len(node_features)

    x = np.zeros((max_nodes, node_features.shape[1]), dtype=np.float32)
    a = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    mask = np.zeros((max_nodes,), dtype=np.float32)
    x[:n] = node_features
    a[:n, :n] = adjacency
    mask[:n] = 1.0
    return x, a, mask, n, truncated


def build_graph_dataset(
    detectors: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    *,
    max_nodes: int,
    k: int,
) -> dict[str, np.ndarray | float | int]:
    """Build graph tensors for all shots."""
    num_examples = len(detectors)
    feature_dim = 4
    x = np.zeros((num_examples, max_nodes, feature_dim), dtype=np.float32)
    a = np.zeros((num_examples, max_nodes, max_nodes), dtype=np.float32)
    mask = np.zeros((num_examples, max_nodes), dtype=np.float32)
    node_counts = np.zeros((num_examples,), dtype=np.int32)
    truncated = np.zeros((num_examples,), dtype=np.uint8)
    empty_graphs = np.zeros((num_examples,), dtype=np.uint8)

    for shot_id, detector_sample in enumerate(detectors):
        graph_x, graph_a, graph_mask, n, was_truncated = build_graph_from_shot(
            detector_sample,
            coords,
            max_nodes=max_nodes,
            k=k,
        )
        x[shot_id] = graph_x
        a[shot_id] = graph_a
        mask[shot_id] = graph_mask
        node_counts[shot_id] = n
        truncated[shot_id] = int(was_truncated)
        empty_graphs[shot_id] = int(np.sum(detector_sample) == 0)

    return {
        "X": x,
        "A": a,
        "mask": mask,
        "y": labels.astype(np.uint8).reshape(-1),
        "node_counts": node_counts,
        "truncated": truncated,
        "empty_graphs": empty_graphs,
        "feature_dim": feature_dim,
        "truncation_rate": float(np.mean(truncated)),
        "empty_graph_fraction": float(np.mean(empty_graphs)),
    }


def build_fixed_detector_dataset(
    detectors: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    dem: stim.DetectorErrorModel,
) -> dict[str, np.ndarray | float | int]:
    """Build one fixed detector-location graph per shot.

    Nodes are all detector locations. The per-shot detector bit is a node
    feature, and topology is shared across shots from the detector error model.
    """
    num_examples, num_detectors = detectors.shape
    adjacency, degrees, boundary_counts = detector_error_model_adjacency(
        dem,
        num_detectors=num_detectors,
    )
    normalized_adjacency = normalize_adjacency(adjacency)

    detector_index = (
        np.arange(num_detectors, dtype=np.float32)[:, None] / max(num_detectors - 1, 1)
    )
    max_degree = max(float(np.max(degrees)), 1.0)
    max_boundary_count = max(float(np.max(boundary_counts)), 1.0)
    static_features = np.concatenate(
        [
            coords.astype(np.float32),
            detector_index,
            (degrees[:, None] / max_degree).astype(np.float32),
            (boundary_counts[:, None] / max_boundary_count).astype(np.float32),
        ],
        axis=1,
    )

    feature_dim = 1 + static_features.shape[1]
    x = np.zeros((num_examples, num_detectors, feature_dim), dtype=np.float32)
    x[:, :, 0] = detectors.astype(np.float32)
    x[:, :, 1:] = static_features[None, :, :]

    a = np.broadcast_to(
        normalized_adjacency[None, :, :],
        (num_examples, num_detectors, num_detectors),
    ).copy()
    mask = np.ones((num_examples, num_detectors), dtype=np.float32)

    return {
        "X": x,
        "A": a.astype(np.float32),
        "mask": mask,
        "y": labels.astype(np.uint8).reshape(-1),
        "node_counts": np.full((num_examples,), num_detectors, dtype=np.int32),
        "truncated": np.zeros((num_examples,), dtype=np.uint8),
        "empty_graphs": np.zeros((num_examples,), dtype=np.uint8),
        "dem_degrees": degrees.astype(np.float32),
        "dem_boundary_counts": boundary_counts.astype(np.float32),
        "feature_dim": feature_dim,
        "truncation_rate": 0.0,
        "empty_graph_fraction": 0.0,
        "dem_edge_count": int(np.sum(adjacency) // 2),
        "boundary_detector_count": int(np.sum(boundary_counts > 0)),
    }


def build_graph_file(
    input_path: Path,
    output_path: Path,
    *,
    max_nodes: int,
    k: int,
    seed: int,
    graph_type: str,
) -> Path:
    with np.load(input_path, allow_pickle=False) as raw:
        detectors = raw["detectors"].astype(np.uint8)
        labels = raw["labels"].astype(np.uint8)
        distance = int(raw["distance"])
        rounds = int(raw["rounds"])
        p = float(raw["p"])
        if "detector_coords" in raw.files:
            coords = raw["detector_coords"].astype(np.float32)
        else:
            circuit = make_surface_code_circuit(distance=distance, rounds=rounds, p=p)
            coords = get_detector_coord_array(circuit)

        if {"train_idx", "val_idx", "test_idx"}.issubset(raw.files):
            train_idx, val_idx, test_idx = load_split_indices(raw, seed=seed)
        else:
            train_idx, val_idx, test_idx = make_split_indices(len(labels), seed=seed)

        if graph_type == "active":
            graph_data = build_graph_dataset(detectors, labels, coords, max_nodes=max_nodes, k=k)
            graph_type_name = "active_defect_knn"
            node_features = "x,y,t,detector_index_normalized"
            output_max_nodes = max_nodes
        elif graph_type == "fixed":
            circuit = make_surface_code_circuit(distance=distance, rounds=rounds, p=p)
            dem = circuit.detector_error_model(decompose_errors=True)
            graph_data = build_fixed_detector_dataset(detectors, labels, coords, dem)
            graph_type_name = "fixed_detector_dem"
            node_features = "detector_bit,x,y,t,detector_index_normalized,dem_degree,boundary_count"
            output_max_nodes = detectors.shape[1]
        else:
            raise ValueError(f"Unsupported graph_type: {graph_type}")

        metadata = {
            key: raw[key]
            for key in ("distance", "rounds", "p", "shots", "seed")
            if key in raw.files
        }

    ensure_parent_dir(output_path)
    np.savez_compressed(
        output_path,
        **graph_data,
        **metadata,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        max_nodes=np.array(output_max_nodes, dtype=np.int64),
        k_neighbors=np.array(k, dtype=np.int64),
        graph_type=np.array(graph_type_name),
        node_features=np.array(node_features),
    )

    print("Graphs:")
    print(f"  input = {input_path}")
    print(f"  output = {output_path}")
    print(f"  examples = {len(labels)}")
    print(f"  graph_type = {graph_type_name}")
    print(f"  max_nodes = {output_max_nodes}")
    print(f"  k_neighbors = {k}")
    if graph_type == "fixed":
        print(f"  dem_edge_count = {graph_data['dem_edge_count']}")
        print(f"  boundary_detector_count = {graph_data['boundary_detector_count']}")
    print(f"  truncation_rate = {graph_data['truncation_rate']:.6f}")
    print(f"  empty_graph_fraction = {graph_data['empty_graph_fraction']:.6f}")
    return output_path


def parse_args() -> argparse.Namespace:
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max-nodes", type=int, default=cfg.max_nodes)
    parser.add_argument("--k", type=int, default=cfg.k_neighbors)
    parser.add_argument("--graph-type", choices=("active", "fixed"), default="active")
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as raw:
        output_path = args.out or default_graph_data_path(
            DATA_DIR,
            int(raw["distance"]),
            int(raw["rounds"]),
            float(raw["p"]),
        )
    build_graph_file(
        args.input,
        output_path,
        max_nodes=args.max_nodes,
        k=args.k,
        seed=args.seed,
        graph_type=args.graph_type,
    )


if __name__ == "__main__":
    main()
