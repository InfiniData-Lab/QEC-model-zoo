from typing import List

import numpy as np
from scipy.optimize import curve_fit


def ler_model(R, e):
    return 0.5 - 0.5 * (1 - 2 * e) ** R


def compute_ler(shot_ids: List[int], rounds_per_shot: List[int], y_gt: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the LER per round.

    :param shot_ids: List of shot IDs.
    :param rounds_per_shot: Amount of rounds per shot (aligned with shot_ids).
    :param y_gt: Ground truth labels for all shots.
    :param y_pred: Predicted labels for all shots.
    :return: LER per round.
    """

    assert len(shot_ids) == len(rounds_per_shot) == y_pred.shape[0] == y_gt.shape[0]

    # Boolean array: True where prediction is wrong
    is_error = y_pred != y_gt

    # Group by rounds without constructing Python lists
    unique_rounds, inverse = np.unique(rounds_per_shot, return_inverse=True)

    errors_per_round = np.bincount(inverse, weights=is_error)
    total_per_round = np.bincount(inverse)

    leps = errors_per_round / total_per_round

    # Fit curve to find LER
    ler, cov = curve_fit(
        ler_model,
        unique_rounds,
        leps,
        bounds=(0, 0.5),
    )

    return float(ler[0])
