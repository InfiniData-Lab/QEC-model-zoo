"""Visualize raw Stim detector data and the graph built from one shot.

Run from the project root, for example:

    python qec_gnn/utils/visualize_input_data.py \
        --raw data/generated/raw_run2_d3_r5_p02_s50000.npz \
        --graph-type fixed \
        --shot 0 \
        --out outputs/plots/input_visualization

If you already have a graph tensor file, pass it with ``--graph``. Otherwise the
script builds the requested graph type in memory from the raw Stim data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# This file lives under qec_gnn/utils/, while the importable package root is two
# directories above it. Add that root so direct script execution works.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qec_gnn.graph_builder import build_fixed_detector_dataset, build_graph_from_shot
from qec_gnn.stim_utils import make_surface_code_circuit


FEATURE_NAMES = {
    "active_defect_knn": ["x", "y", "t", "detector_index"],
    "fixed_detector_dem": [
        "detector_bit",
        "x",
        "y",
        "t",
        "detector_index",
        "dem_degree",
        "boundary_count",
    ],
}


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.shape == () else array.tolist()


def _metadata(raw: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return {
        "distance": int(raw["distance"]),
        "rounds": int(raw["rounds"]),
        "p": float(raw["p"]),
        "shots": int(raw["shots"]) if "shots" in raw.files else len(raw["labels"]),
    }


def _load_or_build_graph(
    *,
    raw_path: Path,
    graph_path: Path | None,
    shot: int,
    graph_type: str,
    max_nodes: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, list[int]]:
    """Return one graph ``(X, A, mask, graph_name, detector_ids)``."""
    if graph_path is not None:
        with np.load(raw_path, allow_pickle=False) as raw:
            num_detectors = raw["detectors"].shape[1]

        with np.load(graph_path, allow_pickle=False) as graph:
            x = graph["X"][shot].astype(np.float32)
            a = graph["A"][shot].astype(np.float32)
            mask = graph["mask"][shot].astype(np.float32)
            graph_name = _scalar(graph["graph_type"]) if "graph_type" in graph.files else "loaded_graph"

        # Fixed graphs map node i -> detector i. Active graphs cannot recover the
        # original detector ids from X alone except via normalized detector_index.
        if graph_name == "fixed_detector_dem":
            detector_ids = list(range(int(mask.sum())))
        else:
            detector_ids = [int(round(v * num_detectors)) for v in x[: int(mask.sum()), -1]]
        return x, a, mask, str(graph_name), detector_ids

    with np.load(raw_path, allow_pickle=False) as raw:
        detectors = raw["detectors"].astype(np.uint8)
        labels = raw["labels"].astype(np.uint8)
        coords = raw["detector_coords"].astype(np.float32)
        meta = _metadata(raw)

    if graph_type == "active":
        x, a, mask, _, _ = build_graph_from_shot(
            detectors[shot],
            coords,
            max_nodes=max_nodes,
            k=k,
        )
        detector_ids = np.flatnonzero(detectors[shot]).astype(int).tolist()
        if not detector_ids:
            detector_ids = [-1]
        return x, a, mask, "active_defect_knn", detector_ids

    circuit = make_surface_code_circuit(
        distance=meta["distance"],
        rounds=meta["rounds"],
        p=meta["p"],
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    graph_data = build_fixed_detector_dataset(
        detectors[shot : shot + 1],
        labels[shot : shot + 1],
        coords,
        dem,
    )
    return (
        graph_data["X"][0],
        graph_data["A"][0],
        graph_data["mask"][0],
        "fixed_detector_dem",
        list(range(detectors.shape[1])),
    )


def print_summary(
    *,
    raw_path: Path,
    shot: int,
    detectors: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    x: np.ndarray,
    a: np.ndarray,
    mask: np.ndarray,
    graph_name: str,
    detector_ids: list[int],
) -> None:
    active_ids = np.flatnonzero(detectors[shot]).astype(int)
    real_nodes = int(mask.sum())
    edge_count = int(np.sum((a > 0) & ~np.eye(a.shape[0], dtype=bool)) // 2)
    feature_names = FEATURE_NAMES.get(graph_name, [f"f{i}" for i in range(x.shape[1])])

    print("Stim input data")
    print(f"  raw file = {raw_path}")
    print(f"  shot = {shot}")
    print(f"  detector vector shape = {detectors[shot].shape}")
    print(f"  active detector ids = {active_ids.tolist()}")
    print(f"  active detector count = {len(active_ids)}")
    print(f"  logical flip label = {int(labels[shot])}")
    print("")
    print("Graph built from this shot")
    print(f"  graph type = {graph_name}")
    print(f"  node feature matrix X shape = {x.shape}")
    print(f"  adjacency matrix A shape = {a.shape}")
    print(f"  real nodes = {real_nodes}")
    print(f"  visible non-self edges = {edge_count}")
    print(f"  feature names = {feature_names}")
    print("")
    print("First real node rows")
    limit = min(real_nodes, 10)
    for node_id in range(limit):
        detector_id = detector_ids[node_id] if node_id < len(detector_ids) else node_id
        values = ", ".join(
            f"{name}={x[node_id, i]:.3f}"
            for i, name in enumerate(feature_names[: x.shape[1]])
        )
        print(f"  node {node_id} detector {detector_id}: {values}")
    if real_nodes > limit:
        print(f"  ... {real_nodes - limit} more nodes")

    # Show coordinates for active detectors separately because they are the most
    # intuitive bridge from raw Stim bits to graph nodes.
    if len(active_ids):
        print("")
        print("Active detector coordinates")
        for detector_id in active_ids[:10]:
            cx, cy, ct = coords[detector_id]
            print(f"  detector {detector_id}: x={cx:.3f}, y={cy:.3f}, t={ct:.3f}")
        if len(active_ids) > 10:
            print(f"  ... {len(active_ids) - 10} more active detectors")


def plot_detector_vector(detector_bits: np.ndarray, label: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 2.8))
    values = detector_bits[None, :]
    ax.imshow(values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    for detector_id, bit in enumerate(detector_bits):
        text_color = "white" if bit else "black"
        ax.text(detector_id, 0, str(detector_id), ha="center", va="center", fontsize=7, color=text_color)
    active_ids = np.flatnonzero(detector_bits).astype(int).tolist()
    ax.set_title(
        f"Raw Stim detector vector for one shot (label={label}); "
        f"red tiles fired: {active_ids}"
    )
    ax.set_xlabel("Detector index")
    ax.set_yticks([])
    ax.set_xticks(np.arange(len(detector_bits)))
    ax.set_xticklabels([])
    ax.set_xlim(-0.5, len(detector_bits) - 0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_detector_coordinates(
    coords: np.ndarray,
    detector_bits: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    inactive = detector_bits == 0
    active = detector_bits == 1

    ax.scatter(coords[inactive, 0], coords[inactive, 1], c="lightgray", s=80, label="inactive detector")
    scatter = ax.scatter(
        coords[active, 0],
        coords[active, 1],
        c=coords[active, 2],
        cmap="viridis",
        s=150,
        edgecolors="black",
        label="active detector",
    )

    for detector_id, (x_coord, y_coord, _) in enumerate(coords):
        if detector_bits[detector_id]:
            ax.text(x_coord + 0.01, y_coord + 0.01, str(detector_id), fontsize=8)

    ax.set_title("Detector x/y locations; active detectors are highlighted")
    ax.set_xlabel("normalized x")
    ax.set_ylabel("normalized y")
    ax.legend(loc="best")
    if np.any(active):
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("normalized time coordinate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _node_positions(
    *,
    graph_name: str,
    x: np.ndarray,
    detector_ids: list[int],
    coords: np.ndarray,
    real_nodes: int,
) -> np.ndarray:
    if graph_name == "fixed_detector_dem":
        positions = np.zeros((real_nodes, 2), dtype=np.float32)
        # Use time as the vertical axis. Many detectors share the same x/y
        # spatial position across rounds, so an x/y plot hides most nodes.
        positions[:, 0] = coords[:real_nodes, 2]
        positions[:, 1] = coords[:real_nodes, 0] + 0.08 * coords[:real_nodes, 1]
        return positions

    positions = np.zeros((real_nodes, 2), dtype=np.float32)
    for node_id, detector_id in enumerate(detector_ids[:real_nodes]):
        if detector_id >= 0:
            positions[node_id, 0] = coords[detector_id, 2]
            positions[node_id, 1] = coords[detector_id, 0] + 0.08 * coords[detector_id, 1]
        else:
            positions[node_id] = x[node_id, :2]
    return positions


def plot_graph(
    *,
    x: np.ndarray,
    a: np.ndarray,
    mask: np.ndarray,
    coords: np.ndarray,
    detector_bits: np.ndarray,
    detector_ids: list[int],
    graph_name: str,
    output_path: Path,
) -> None:
    real_nodes = int(mask.sum())
    positions = _node_positions(
        graph_name=graph_name,
        x=x,
        detector_ids=detector_ids,
        coords=coords,
        real_nodes=real_nodes,
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    edge_threshold = 1e-8
    for i in range(real_nodes):
        for j in range(i + 1, real_nodes):
            if a[i, j] > edge_threshold:
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    color="gray",
                    alpha=0.18,
                    linewidth=0.8,
                    zorder=1,
                )

    node_colors = []
    for detector_id in detector_ids[:real_nodes]:
        if detector_id >= 0 and detector_bits[detector_id] == 1:
            node_colors.append("tab:red")
        else:
            node_colors.append("#f7f7f7")

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=node_colors,
        edgecolors="black",
        s=170,
        zorder=2,
    )

    for node_id, detector_id in enumerate(detector_ids[:real_nodes]):
        label = "dummy" if detector_id < 0 else str(detector_id)
        text_color = "tab:red" if detector_id >= 0 and detector_bits[detector_id] else "black"
        ax.text(positions[node_id, 0] + 0.006, positions[node_id, 1] + 0.006, label, fontsize=7, color=text_color)

    ax.set_title(f"Graph nodes and edges ({graph_name})")
    ax.set_xlabel("normalized time / measurement round")
    ax.set_ylabel("detector spatial lane: x + small y offset")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_active_subgraph(
    *,
    coords: np.ndarray,
    detector_bits: np.ndarray,
    output_path: Path,
) -> None:
    active_ids = np.flatnonzero(detector_bits).astype(int)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if len(active_ids) == 0:
        ax.text(0.5, 0.5, "No active detectors in this shot", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
    else:
        positions = np.zeros((len(active_ids), 2), dtype=np.float32)
        positions[:, 0] = coords[active_ids, 2]
        positions[:, 1] = coords[active_ids, 0] + 0.08 * coords[active_ids, 1]
        ax.scatter(positions[:, 0], positions[:, 1], c="tab:red", edgecolors="black", s=220)
        for row, detector_id in enumerate(active_ids):
            x_coord, y_coord = positions[row]
            ax.text(x_coord + 0.008, y_coord + 0.008, str(detector_id), fontsize=10)
        ax.set_title("Only the active/fired detectors in this shot")
        ax.set_xlabel("normalized time / measurement round")
        ax.set_ylabel("detector spatial lane: x + small y offset")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Raw Stim dataset .npz from qec_gnn.generate_data")
    parser.add_argument("--graph", type=Path, default=None, help="Optional graph tensor .npz from qec_gnn.graph_builder")
    parser.add_argument("--shot", type=int, default=0, help="Shot/example index to visualize")
    parser.add_argument("--graph-type", choices=("active", "fixed"), default="fixed")
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("outputs/plots/input_visualization"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with np.load(args.raw, allow_pickle=False) as raw:
        detectors = raw["detectors"].astype(np.uint8)
        labels = raw["labels"].astype(np.uint8).reshape(-1)
        coords = raw["detector_coords"].astype(np.float32)

    if args.shot < 0 or args.shot >= len(detectors):
        raise ValueError(f"shot must be in [0, {len(detectors) - 1}], got {args.shot}")

    x, a, mask, graph_name, detector_ids = _load_or_build_graph(
        raw_path=args.raw,
        graph_path=args.graph,
        shot=args.shot,
        graph_type=args.graph_type,
        max_nodes=args.max_nodes,
        k=args.k,
    )

    print_summary(
        raw_path=args.raw,
        shot=args.shot,
        detectors=detectors,
        labels=labels,
        coords=coords,
        x=x,
        a=a,
        mask=mask,
        graph_name=graph_name,
        detector_ids=detector_ids,
    )

    detector_path = args.out / f"shot_{args.shot:04d}_detector_vector.png"
    coords_path = args.out / f"shot_{args.shot:04d}_detector_coordinates.png"
    graph_path = args.out / f"shot_{args.shot:04d}_{graph_name}_graph.png"
    active_path = args.out / f"shot_{args.shot:04d}_active_detectors_only.png"

    plot_detector_vector(detectors[args.shot], int(labels[args.shot]), detector_path)
    plot_detector_coordinates(coords, detectors[args.shot], coords_path)
    plot_graph(
        x=x,
        a=a,
        mask=mask,
        coords=coords,
        detector_bits=detectors[args.shot],
        detector_ids=detector_ids,
        graph_name=graph_name,
        output_path=graph_path,
    )
    plot_active_subgraph(
        coords=coords,
        detector_bits=detectors[args.shot],
        output_path=active_path,
    )

    print("")
    print("Wrote plots")
    print(f"  detector vector = {detector_path}")
    print(f"  detector coordinates = {coords_path}")
    print(f"  graph = {graph_path}")
    print(f"  active detectors only = {active_path}")


if __name__ == "__main__":
    main()
