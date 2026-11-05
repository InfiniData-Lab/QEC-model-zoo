import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Any, Dict, Tuple, Optional

import numpy as np
import optuna
import yaml
from optuna.samplers import RandomSampler, TPESampler

from analysis.metric import calculate_metric
from benchmark.utils.full_decoder.base_experiment import BaseExperiment
from data.real.base_data_loader import BaseDataLoader
from model.full_decoder.base_classifier import BaseClassifier
from model.full_decoder.classifier_factory import load_model_config, find_model_class
from utils.log_utils import capture_output_to_file
from utils.string_utils import StringUtils


class TrainFullDecoderExperiment(BaseExperiment):
    """
    Train full-decoder models.
    """
    def __init__(self, experiment_name: str, dataloader: BaseDataLoader, decoder_configs: List[Dict[str, str]], seed: int, tuning_config: Dict[str, Any], decoder_filter: List[str] | None = None):
        self.experiment_name = experiment_name
        self.dataloader = dataloader
        self.decoder_configs = decoder_configs
        self.seed = seed
        self.tuning_config = tuning_config
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
            uses_raw_data = 'lstm' == model_name.lower() # so far lstm is the only sequence model to use raw data

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

            # Set checkpoint path for TFMs internally because they need it in .fit instead of .save_model
            if "tabiclv2_ft" in model_name or "tabpfnv25_ft" in model_name:
                model.checkpoint_path = decoder_checkpoint_save_path

            X_train, y_train, = self.dataloader.get_train()
            X_val, y_val = self.dataloader.get_val()

            time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_file_path = Path('experiments') / 'logs' / self.experiment_name / f'train_{model_name}_{time_str}.log'
            with capture_output_to_file(log_file_path):
                training_info = self.dataloader.get_info_for_split('train', raw=uses_raw_data)
                n_samples = training_info["n_samples"]
                n_features = training_info["n_features"]
                print("\n================================================================")
                print(f"Training full-decoder {model.model_name} on dataset {self.dataloader.dataset_name} with shape ({n_samples}, {n_features})")
                print("================================================================")
                if self.tuning_config is not None: # Train with HP tuning
                    max_trials = self.tuning_config['max_trials']
                    early_stopping = self.tuning_config['early_stopping']
                    patience = self.tuning_config['patience'] if early_stopping else None
                    tuning_method = self.tuning_config['tuning_method']
                    minimize_metric = self.tuning_config['minimize_metric']
                    model, model_parameters, tuning_info = self._train_model_with_tuning(model, config['hyperparameters'], X_train, y_train, X_val, y_val, max_trials, patience, minimize_metric, tuning_method)
                    training_time_seconds = tuning_info["tuning_seconds"]
                    config['hp_tuning_info'] = tuning_info
                    for hp, value in model_parameters.items():
                        if hp in  config['hyperparameters'].keys():
                            config['hyperparameters'][hp]['default'] = value
                        else:
                            config['hyperparameters'][hp] = {'default': value, 'value_type': 'ignore'}
                else:
                    training_time_seconds = model.fit_with_timing(X_train, y_train, X_val, y_val)
                print(f"Finished training full-decoder {model_name} on dataset {self.dataloader.dataset_name} in {training_time_seconds} seconds.")
            print(f"Saved log file to {log_file_path}")
            decoder_config_save_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path = model.save_model(decoder_checkpoint_save_path)
            with open(str(decoder_config_save_path), 'w') as f:
                yaml.dump(config, f)
            if checkpoint_path is not None:
                print(f"Best model checkpoint saved to {checkpoint_path}")
                print(f"Model parameters saved to {decoder_config_save_path}")

    def _train_model_with_tuning(
            self,
            model,
            params_config: Dict[str, Any],
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            max_trials: int,
            early_stopping_patience: Optional[int] = None,
            minimize_metric: str = 'ler',
            tuning_method: str = 'random',
    ) -> Tuple[BaseClassifier, Dict[str, Any], Dict[str, Any]]:
        """
        Train a model with hyperparameter tuning using the optuna library.

        :param params_config: Hyperparameter optimization configuration, dictionary with hyperparameter names as keys. Values are dictionaries with
        :param dataloader: DataLoader instance to use.
        :param max_trials: Maximum number of trials to run.
        :param early_stopping_patience: When set, amount of trials to wait before early stopping. When not set, max trials will be used.
        :param minimize_metric: Metric to minimize.
        :param tuning_method: Tuning method to use.
        :return: The best trained model, the best parameters, and a dictionary containing optimization information.
        """
        model_name = StringUtils.normalize_string(model.args.model_name)

        best_model, best_params, best_value, tuning_time, trials = self.get_best_tuned_model(
            model_name,
            params_config,
            X_train,
            y_train,
            X_val,
            y_val,
            self.model_args,
            max_trials=max_trials,
            early_stopping_patience=early_stopping_patience,
            minimize_metric=minimize_metric,
            tuning_method=tuning_method,
        )

        optimization_info = {
            "trials": len(trials),
            "max_trials": max_trials,
            "early_stopping_enabled": early_stopping_patience is not None,
            "early_stopping_patience": early_stopping_patience,
            "metric": minimize_metric,
            "validation_metric_value": best_value,
            "best_params": best_params,
            "data_fraction": self.dataloader.data_fraction,
            "train_samples": X_train.shape[0],
            "val_samples": X_val.shape[0],
            "tuning_seconds": tuning_time,
            "early_stopped": len(trials) < max_trials,
        }

        return best_model, best_params, optimization_info

    @staticmethod
    def get_best_tuned_model(
            model_name: str,
            params_config: Dict[str, Any],
            X_train,
            y_train,
            X_val,
            y_val,
            model_args: argparse.Namespace,
            max_trials: int = 500,
            early_stopping_patience: Optional[int] = 50,
            minimize_metric: str = 'lep',
            tuning_method: str = 'tpe',
    ):
        """
        Get best model using hyperparameter tuning.

        :param model_name: Name of model.
        :param params_config: Parameter tuning configuration.
        :param dataloader: DataLoader instance.
        :param model_args: Model arguments
        :param max_trials: Maximum number of trials to run.
        :param early_stopping_patience: Patience for early stopping.
        :param minimize_metric: Metric to minimize.
        :param tuning_method: Tuning method to use.
        :return: Tuple (best fitted model, best parameters, best value for metric, tuning time, amount of trials)
        """
        print(f"Optimising hyperparameters for model {model_name}...")
        model_class = find_model_class(StringUtils.normalize_string(model_name))

        def objective(trial: optuna.Trial) -> float:
            trial_params = {}

            for hp, config in params_config.items():
                value_type = config.get("value_type")
                hp_range = list(config.get("range", []))
                use_log_scale = config.get("log_scale", False)

                constraint = config.get("constraint", None)
                if constraint is not None:
                    other_param = constraint["other_param"]

                    if other_param not in trial_params:
                        raise ValueError(
                            f"Constraint for {hp} depends on {other_param}, "
                            f"but {other_param} has not been sampled yet. "
                            f"Make sure {other_param} appears earlier in params_config."
                        )

                    if constraint["type"] == ">":
                        hp_range[0] = trial_params[other_param]
                    elif constraint["type"] == "<":
                        hp_range[1] = trial_params[other_param]
                    else:
                        raise ValueError(
                            "Invalid constraint type: {}".format(constraint["type"])
                        )

                if value_type == 'int':
                    trial_params[hp] = trial.suggest_int(
                        hp,
                        int(hp_range[0]),
                        int(hp_range[1]),
                        log=use_log_scale,
                    )
                elif value_type == 'float':
                    trial_params[hp] = trial.suggest_float(
                        hp,
                        float(hp_range[0]),
                        float(hp_range[1]),
                        log=use_log_scale,
                    )
                elif value_type == 'categorical':
                    trial_params[hp] = trial.suggest_categorical(hp, hp_range)
                elif value_type == 'ignore':
                    trial_params[hp] = config.get("default")
                else:
                    raise NotImplementedError(
                        f"Value type {value_type} not implemented for hyperparameter optimization"
                    )

            instance = model_class(trial_params, model_args)
            instance.model_name = model_name
            instance.fit(X_train, y_train, X_val, y_val)

            y_pred = instance.predict(X_val)
            metric_score = calculate_metric(minimize_metric, y_val, y_pred)

            # Store useful trial-level data.
            trial.set_user_attr("model_instance", instance)
            trial.set_user_attr("params", trial_params)

            return metric_score

        min_delta = 1e-8
        best_score = float("inf")
        trials_without_improvement = 0

        def early_stopping_callback(
                study: optuna.Study,
                trial: optuna.trial.FrozenTrial,
        ) -> None:
            nonlocal best_score, trials_without_improvement

            if trial.value is None:
                return

            improvement = best_score - trial.value

            if improvement > min_delta:
                best_score = trial.value
                trials_without_improvement = 0
            else:
                trials_without_improvement += 1

            if (
                    early_stopping_patience is not None
                    and trials_without_improvement >= early_stopping_patience
            ):
                print(
                    f"Early stopping triggered after "
                    f"{trials_without_improvement} trials without significant improvement."
                )
                study.stop()

        if tuning_method.lower() == 'random':
            sampler = RandomSampler(seed=model_args.seed)
        elif tuning_method.lower() == 'tpe':
            sampler = TPESampler(seed=model_args.seed)
        else:
            raise ValueError(f'Invalid tuning method: {tuning_method}')
        study = optuna.create_study(direction='minimize', sampler=sampler)

        start_tuning = time.perf_counter()

        callbacks = []
        if early_stopping_patience is not None:
            callbacks.append(early_stopping_callback)

        study.optimize(
            objective,
            n_trials=max_trials,
            callbacks=callbacks,
        )

        end_tuning = time.perf_counter()

        best_trial = study.best_trial
        best_model = best_trial.user_attrs["model_instance"]

        best_params = dict(best_trial.user_attrs["params"])
        best_value = study.best_value

        for hp, config in params_config.items():
            if hp not in best_params:
                best_params[hp] = config["default"]

        print(f"Finished optimizing hyperparameters for model {model_name}.")

        tuning_time = end_tuning - start_tuning
        return best_model, best_params, best_value, tuning_time, study.trials
