import time
from pathlib import Path
from typing import List, Dict

import argparse

import numpy as np
import torch

from analysis.metric import calculate_metric
from data.croissant_data_loader import CroissantDataLoader
from data.real.base_data_loader import BaseDataLoader
from benchmark.utils.full_decoder.base_experiment import BaseExperiment
from model.full_decoder.classifier_factory import load_model_config
from utils.string_utils import StringUtils
import json

class EvalFullDecoderExperiment(BaseExperiment):
    """
    Evaluate full decoder models.
    """
    def __init__(self, experiment_name: str, dataloader: CroissantDataLoader, decoder_configs: List[Dict[str, str]], seed: int, retrain_model: bool = False, repetitions: int = 1, decoder_filter: List[str] | None = None):
        self.experiment_name = experiment_name
        self.dataloader = dataloader
        self.decoder_configs = decoder_configs
        self.seed = seed
        self.retrain_model = retrain_model
        self.repetitions = repetitions
        self.decoder_filter = decoder_filter

    def run(self) -> None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA is not available but required for training full decoders.')
        device = "cuda"

        # Load each model and calculate predictions
        for decoder_config in self.decoder_configs:
            decoder_config_path = Path(decoder_config["param_path"])
            checkpoint_path = decoder_config.get("checkpoint_path", None)
            # Load model
            ModelClass, model_parameters, config = load_model_config(Path(decoder_config_path))
            model_name = StringUtils.normalize_string(config['model'])
            if self.decoder_filter and model_name not in StringUtils.filter_by_match([model_name], self.decoder_filter):
                print(f'Skipping decoder model {model_name}.')
                continue
            uses_raw_data = 'lstm' in model_name.lower()  # so far lstm is the only sequence model to use raw data
            val_dataset_info = self.dataloader.get_info_for_split('val', raw=uses_raw_data)
            model_args = argparse.Namespace(
                objective="classification" if val_dataset_info["n_classes"] > 1 else "binary",
                device=device,
                use_gpu=True,
                gpu_ids=[0],
                logging_period=100,
                early_stopping_rounds=10,
                data_parallel=False,
                num_features=val_dataset_info["n_features"],
                num_classes=val_dataset_info["n_classes"],
                seed=self.seed,
                model_name=model_name,
            )

            if self.retrain_model or 'tabicl' in model_name or 'tabpfn' in model_name:
                X_train, y_train = self.dataloader.get_train()
                X_val, y_val = self.dataloader.get_val()
                model = ModelClass(model_parameters, model_args)

                # Retrain the model
                print(f'Retraining full decoder model {model_name}...')
                model.fit(X_train, y_train, X_val, y_val)
                print(f'Finished retraining full decoder model {model_name}.')
            else:
                assert checkpoint_path is not None
                model = ModelClass.load_model(checkpoint_path, model_parameters, model_args)

            # Calculate test predictions
            y_preds = []
            timings = []
            print('\n===========================================================')
            print(f'Calculating predictions for full decoder {model.model_name} on dataset {self.dataloader.dataset_name} for {self.repetitions} repetitions...')
            print('===========================================================')
            all_equal_preds = True
            for i in range(self.repetitions):
                X_test, y_test, shot_features_test = self.dataloader.get_test(raw=uses_raw_data, include_shot_features=True)
                batch_size_inference = getattr(model, 'batch_size_inference', None) # Some models use batching
                if getattr(model, "n_shot_features", 0) > 0:
                    y_pred, exec_time_seconds = model.predict_with_timing(X_test, shot_features=shot_features_test)
                else:
                    y_pred, exec_time_seconds = model.predict_with_timing(X_test, batch_size=batch_size_inference)
                if i > 0 and not np.array_equal(y_pred, y_preds[i - 1]): # Set flag so we can check consistency of predictions
                    all_equal_preds = False
                timings.append(exec_time_seconds)
                y_preds.append(y_pred)
                print(f'Repetition {i + 1}/{self.repetitions} completed.')

            print(f'Finished calculating predictions for full decoder {model.model_name} on dataset {self.dataloader.dataset_name}.')

            # Calculate evaluation metrics
            shots, rounds_per_shot = self.dataloader.get_test_shot_info()
            metrics = ['lep', 'accuracy', 'ler']
            metric_results = {}
            predictions = {}
            y_test = self.dataloader.get_test()[1]
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
            results_file_path = Path('experiments') / 'results' / self.experiment_name / f'{model_name}.json'
            predictions_file_path = Path('experiments') / 'predictions' / self.experiment_name / f'{model_name}.json'
            results = {
                'data_info': self.dataloader.get_info(raw=uses_raw_data),
                'metrics': metric_results,
                'config': {
                    'model': model.config_to_dict(),
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
