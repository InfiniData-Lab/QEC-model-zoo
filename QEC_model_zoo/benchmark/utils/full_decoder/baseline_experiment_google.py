import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.metric import calculate_metric
from benchmark.utils.full_decoder.base_experiment import BaseExperiment
from data.croissant_data_loader import CroissantDataLoader
from model.full_decoder.classical_baselines.MWPM import MWPM
from model.full_decoder.classical_baselines.chromobius import Chromobius


class BaselineExperimentGoogle(BaseExperiment):
    """
    Baseline experiment for surface code google datasets
    """

    def __init__(self, experiment_name: str, dataloader: CroissantDataLoader, seed: int, repetitions = 1):
        """
        Constructor for baseline experiment surface code.

        :param experiment_name: Name of experiment
        :param dataloader: DataLoader
        :param seed: Random seed
        :param repetitions: Number of repetitions for inference measurements
        """
        if not "google" in experiment_name:
            raise ValueError("Baseline experiment only supported for google datasets.")

        self.experiment_name = experiment_name
        self.dataloader = dataloader
        self.seed = seed
        self.repetitions = repetitions

    def run(self) -> None:
        """
        Run the baseline experiment.
        """
        # Calculate test predictions
        y_preds = []
        timings = []
        baseline_type = 'chromobius' if 'color' in self.experiment_name else 'mwpm' # Auto detect baseline to be used
        print('\n===========================================================')
        print(
            f'Calculating predictions for baseline {baseline_type} on dataset {self.dataloader.dataset_name} for {self.repetitions} repetitions...')
        print('===========================================================')
        all_equal_preds = True

        test_df = self.dataloader.get_test_df()
        y_test = test_df[self.dataloader.target_column].to_numpy()
        for i in range(self.repetitions):
            if baseline_type.lower() == 'mwpm':
                y_pred, time_seconds = MWPM.predict(test_df)
            else:
                y_pred, time_seconds = Chromobius.predict(self.dataloader)
            if i > 0 and not np.array_equal(y_pred,
                                            y_preds[i - 1]):  # Set flag so we can check consistency of predictions
                all_equal_preds = False

            timings.append(time_seconds)
            y_preds.append(y_pred)
            print(f'Repetition {i + 1}/{self.repetitions} completed.')

        print(
            f'Finished calculating predictions for for baseline {baseline_type} on dataset {self.dataloader.dataset_name}.')

        # Calculate evaluation metrics
        shots, rounds_per_shot = self.dataloader.get_test_shot_info()
        metrics = ['lep', 'accuracy', 'ler']
        metric_results = {}
        predictions = {}
        for i in range(1, self.repetitions + 1):
            repetition_key = f'run_{i}'
            metric_results[repetition_key] = {}
            predictions[repetition_key] = {}
            for metric in metrics:
                metric_value = calculate_metric(metric, y_test, y_preds[i - 1], shots, rounds_per_shot)
                metric_results[repetition_key][metric] = metric_value
                print(f'{metric}: {metric_value}.')
            metric_results[repetition_key]['inference_time'] = timings[i - 1]
            predictions[repetition_key]['y_pred'] = y_preds[i - 1].tolist()

        # Save results along with some metadata
        results_file_path = Path('experiments') / 'results' / self.experiment_name / f'{baseline_type}.json'
        predictions_file_path = Path('experiments') / 'predictions' / self.experiment_name / f'{baseline_type}.json'
        results = {
            'data_info': self.dataloader.get_info(),
            'metrics': metric_results,
            'config': {
                'model': baseline_type,
            },
            'predictions': {
                'shot_ids': shots,
                'rounds_per_shot': rounds_per_shot,
                'y_gt': y_test.tolist(),
                'all_equal_preds': all_equal_preds,
            }
        }
        results_file_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file_path, 'w') as f:
            json.dump(results, f, indent=4)
            print(f"Saved results to {results_file_path}")
        with open(predictions_file_path, 'w') as f:
            json.dump(predictions, f, indent=4)
            print(f"Saved raw predictions to {predictions_file_path}")
