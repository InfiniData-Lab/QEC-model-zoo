from typing import List

import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, log_loss

# todo: add more evaluation metrics for QEC as needed, and consider adding support if necessary.
from analysis.ler import compute_ler


def calculate_metric(metric: str, y_true: np.ndarray, y_pred: np.ndarray, shot_ids: List[int] = None, rounds_per_shot: List[int] = None) -> float:
    if metric == 'balanced_accuracy':
        return balanced_accuracy_score(y_true, y_pred)
    elif metric == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric == 'f1_score':
        return f1_score(y_true, y_pred)
    elif metric == 'log_loss':
        return log_loss(y_true, y_pred)
    elif metric == 'lep':
        return 1 - accuracy_score(y_true, y_pred)
    elif metric == 'ler':
        if shot_ids is None or rounds_per_shot is None:
            raise ValueError('shot_ids and rounds_per_shot are required to calculate metric ler')
        return compute_ler(shot_ids, rounds_per_shot, y_true, y_pred)
    else:
        raise ValueError(f'Unknown evaluation metric: {metric}')
