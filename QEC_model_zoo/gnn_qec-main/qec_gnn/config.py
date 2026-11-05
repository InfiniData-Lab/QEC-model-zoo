"""Default experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "generated"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


@dataclass(frozen=True)
class ExperimentConfig:
    distance: int = 3
    rounds: int = 5
    p: float = 0.005
    shots: int = 30_000
    max_nodes: int = 64
    k_neighbors: int = 4
    hidden_dim: int = 64
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    seed: int = 12345


DEFAULT_CONFIG = ExperimentConfig()
