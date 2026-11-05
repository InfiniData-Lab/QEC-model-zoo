import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Any, Dict, Tuple, Optional

import yaml

from benchmark.utils.full_decoder.base_experiment import BaseExperiment
from data.real.base_data_loader import BaseDataLoader
from model.full_decoder.classifier_factory import load_model_config
from utils.log_utils import capture_output_to_file
from utils.string_utils import StringUtils


class TrainSequenceFullDecoderExperiment(BaseExperiment):
    """
    Train full-decoder models.
    """
    def __init__(self, experiment_name: str, dataloader: BaseDataLoader, decoder_configs: List[Dict[str, str]], seed: int, decoder_filter: List[str] | None = None):
        self.experiment_name = experiment_name
        self.dataloader = dataloader
        self.decoder_configs = decoder_configs
        self.seed = seed
        self.decoder_filter = decoder_filter

    def run(self) -> None:
        device = "cuda"
        # Train each model
        for decoder_config in self.decoder_configs:
            decoder_config_path = Path(decoder_config['params_path'])
            decoder_config_save_path = Path(decoder_config['params_save_path'])
            decoder_checkpoint_save_path = Path(decoder_config['checkpoint_save_path'])

            ModelClass, model_parameters, config = load_model_config(decoder_config_path)
            model_name = StringUtils.normalize_string(config['model'])
            if self.decoder_filter and model_name not in StringUtils.filter_by_match([model_name], self.decoder_filter):
                print(f'Skipping decoder model {model_name}.')
                continue
            if 'lstm' not in model_name.lower():
                raise ValueError('Only LSTM sequence model is supported.')
            uses_raw_data = True

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
            )
            self.model_args = model_args

            model_args.model_name = model_name
            model = ModelClass(model_parameters, model_args)

            time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_file_path = Path('experiments') / 'logs' / self.experiment_name / f'train_{model_name}_{time_str}.log'
            with capture_output_to_file(log_file_path):
                training_info = self.dataloader.get_info_for_split('train', raw=uses_raw_data)
                n_samples = training_info["n_samples"]
                n_features = training_info["n_features"]
                print("\n================================================================")
                print(f"Training full-decoder {model.model_name} on dataset {self.dataloader.dataset_name} with shape ({n_samples}, {n_features})")
                print("================================================================")
                start_time = time.perf_counter()
                model.fit(self.dataloader)
                end_time = time.perf_counter()
                training_time_seconds = end_time - start_time
                print(f"Finished training full-decoder {model_name} on dataset {self.dataloader.dataset_name} in {training_time_seconds} seconds.")
            print(f"Saved log file to {log_file_path}")
            decoder_config_save_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path = model.save_model(decoder_checkpoint_save_path)
            with open(str(decoder_config_save_path), 'w') as f:
                yaml.dump(config, f)
            if checkpoint_path is not None:
                print(f"Best model checkpoint saved to {checkpoint_path}")
                print(f"Model parameters saved to {decoder_config_save_path}")
