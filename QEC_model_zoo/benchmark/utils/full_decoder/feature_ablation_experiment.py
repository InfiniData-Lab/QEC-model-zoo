import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from analysis.metric import calculate_metric
from benchmark.utils.full_decoder.base_experiment import BaseExperiment
from benchmark.utils.full_decoder.train_full_decoder_experiment import TrainFullDecoderExperiment
from data.croissant_data_loader import CroissantDataLoader
from model.full_decoder.classifier_factory import load_model_config
from utils.log_utils import _TqdmLoggingHandler
from utils.random_utils import set_seed
from utils.string_utils import StringUtils

MODEL_CONFIG_DIR = Path("experiments/configs/model/full-decoder")

class FeatureAblationExperiment(BaseExperiment):
    """
    Feature ablation experiment to find the feature groups leading to the best full decoder accuracy for a specific dataset.
    """
    dataloader: CroissantDataLoader
    model_keys: List[str]
    logger: logging.Logger
    out_dir: Path
    seed: int

    def __init__(self, dataloader: CroissantDataLoader, model_keys: List[str], out_dir: Path, seed: int = 42):
        """
        Constructor.

        :param dataloader: Data loader for croissant dataset.
        :param model_keys: List of model keys (names, e.g. xgboost, lightgbm, tabpfn, ...).
        :param out_dir: Output directory for ablation results and logs.
        """
        self.dataloader = dataloader
        self.model_keys = model_keys
        self.seed = seed

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_dir / self.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir

        self.logger = self._setup_logging(out_dir)

    def run(self) -> None:
        """
        Run the ablation experiment.
        """
        # Maps groups of features to the list of features in it.
        logger = self.logger
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available but is required (full-decoder models here train on GPU). "
                "Run this via local_gpu.sh / the Docker image with --gpus all, or on a CUDA machine."
            )

        set_seed(self.seed)

        selected_models = [m.strip() for m in self.model_keys]
        ablation_sets = self._get_feature_ablation_sets(self.dataloader)
        selected_ids = list(ablation_sets.keys())
        for aid in selected_ids:
            if aid not in ablation_sets:
                raise ValueError(f"Unknown ablation: {aid!r}. Available: {list(ablation_sets)}")

        logger.info("Run ID     : %s", self.run_id)
        logger.info("Out dir    : %s", self.out_dir)
        logger.info("Dataset id : %s", self.dataloader.dataset_name)
        logger.info("Ablations  : %s", selected_ids)
        logger.info("Models     : %s", selected_models)
        logger.info("Total runs : %d", len(selected_ids) * len(selected_models))

        pairs = [(a, m) for a in selected_ids for m in selected_models]
        results: List[Dict[str, Any]] = []
        with tqdm(pairs, desc="ablation sweep", unit="run") as pbar:
            for ablation_id, model_key in pbar:
                pbar.set_postfix_str(f"{ablation_id}/{model_key}")
                feature_groups = ablation_sets[ablation_id]
                logger.info("")
                logger.info("--- %s / %s ---", ablation_id, model_key)
                try:
                    result = self._run_one(ablation_id, feature_groups, model_key, self.seed)
                    results.append(result)
                    pbar.set_postfix_str(f"{ablation_id}/{model_key} ler={result['ler']:.4f}")
                except Exception as e:
                    logger.error("[%s/%s] FAILED: %s", ablation_id, model_key, e, exc_info=True)

        if not results:
            logger.warning("No results produced.")
            return

        summary_csv = self.out_dir / "ablation_summary.csv"
        pd.DataFrame(results).to_csv(summary_csv, index=False)
        summary_json = self.out_dir / "ablation_summary.json"
        summary_json.write_text(json.dumps(results, indent=2))

        logger.info("Summary CSV  -> %s", summary_csv)
        logger.info("Summary JSON -> %s", summary_json)

        self._log_leaderboard(results, logger)
        logger.info("Done.")

    def _run_one(
        self,
        ablation_id: str,
        feature_groups: List[str],
        model_key: str,
        seed: int,
    ) -> Dict[str, Any]:
        train_df = self.dataloader.get_train_df()
        val_df = self.dataloader.get_val_df()
        test_df = self.dataloader.get_test_df()

        X_train, y_train = self.dataloader.df_to_features_and_labels(train_df, feature_groups)
        X_val, y_val = self.dataloader.df_to_features_and_labels(val_df, feature_groups)
        X_test, y_test = self.dataloader.df_to_features_and_labels(test_df, feature_groups)

        if X_train.shape[1] == 0:
            raise ValueError(
                f"Ablation {ablation_id!r} resolved to 0 feature columns for this dataset "
                f"-- check that {feature_groups} are valid group names for the selected dataset."
            )

        n_classes = int(len(np.unique(y_train)))
        model_args = argparse.Namespace(
            objective="classification" if n_classes > 2 else "binary",
            device="cuda",
            use_gpu=True,
            gpu_ids=[0],
            logging_period=100,
            early_stopping_rounds=10,
            data_parallel=False,
            num_features=int(X_train.shape[1]),
            num_classes=n_classes,
            seed=seed,
        )

        config_path = MODEL_CONFIG_DIR / f"{model_key}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No model config for {model_key!r} at {config_path}. "
                f"Available: {sorted(p.stem for p in MODEL_CONFIG_DIR.glob('*.yaml'))}"
            )
        ModelClass, model_params, config = load_model_config(config_path)
        model_args.model_name = StringUtils.normalize_string(config["model"])

        model = ModelClass(model_params, model_args)

        t0 = time.perf_counter()
        if model_key in ['catboost', 'lightgbm', 'xgboost']:
            model, _, _, train_time, _ = TrainFullDecoderExperiment.get_best_tuned_model(model_key, config["hyperparameters"], X_train, y_train, X_val, y_val, model_args)
        else:
            model.fit(X_train, y_train, X_val, y_val)
            train_time = time.perf_counter() - t0
        y_pred = model.predict(X_test)
        shots, rounds_per_shot = self.dataloader.get_test_shot_info()

        result = {
            "ablation_id": ablation_id,
            "model": model_key,
            "feature_groups": feature_groups,
            "n_features": int(X_train.shape[1]),
            "train_time_s": round(train_time, 2),
            "accuracy": calculate_metric("accuracy", y_test, y_pred, shots, rounds_per_shot),
            "lep": calculate_metric("lep", y_test, y_pred, shots, rounds_per_shot),
            "ler": calculate_metric("ler", y_test, y_pred, shots, rounds_per_shot),
        }

        self.logger.info(
            "[%s/%s] ler=%.4f  lep=%.4f  acc=%.4f  n_feat=%d  time=%.1fs",
            ablation_id, model_key,
            result["ler"], result["lep"], result["accuracy"],
            result["n_features"], train_time,
        )
        return result

    def _log_leaderboard(self, results: List[Dict[str, Any]], logger: logging.Logger) -> None:
        self.logger.info("")
        self.logger.info("=" * 78)
        self.logger.info("LEADERBOARD  (sorted by LER asc)")
        self.logger.info("  %-22s %-10s %8s %8s %8s %7s", "ablation", "model", "ler", "lep", "acc", "n_feat")
        self.logger.info("-" * 78)
        for r in sorted(results, key=lambda r: r["ler"]):
            self.logger.info(
                "  %-22s %-10s %8.4f %8.4f %8.4f %7d",
                r["ablation_id"], r["model"], r["ler"], r["lep"], r["accuracy"], r["n_features"],
            )
        self.logger.info("=" * 78)

    def _setup_logging(self, out_dir: Path) -> logging.Logger:
        """
        Setup logging so it does not mess with progress bars.

        :param out_dir: Output directory to save logs.
        :return: Logger object.
        """
        logger = logging.getLogger("run_ablations")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass

        ch = _TqdmLoggingHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        return logger

    def _get_feature_ablation_sets(self, dataloader: CroissantDataLoader) -> Dict[str, List[str]]:
        """
        Get feature ablation sets for a specific dataset.

        :param dataloader: Croissant dataloader.
        :return: Map of ablation experiment names to list of feature groups.
        """
        feature_group_mapping = dataloader.get_feature_group_mapping()
        feature_groups = list(feature_group_mapping.keys())
        ablation_sets = {
            'full': feature_groups,
            'minimal': ['code'],
        }
        is_lstm = lambda f: 'lstm' in f.lower()
        contains_lstm = any(is_lstm(f) for f in feature_groups)

        # Additional groups for datasets with LSTM encoding columns
        base_groups = feature_groups
        if contains_lstm:
            crafted = [f for f in feature_groups if not is_lstm(f)]
            base_groups = crafted
            ablation_sets['crafted'] = crafted

            lstm_feature_group = [f for f in feature_groups if is_lstm(f)][0]
            ablation_sets['lstm_only'] = [lstm_feature_group]
            ablation_sets["lstm_no_temporal"] = ["lstm"] + [g for g in base_groups if g != "temporal"]

        # Leave one out of the base groups
        for group in base_groups:
            leave_out = [f for f in base_groups if f != group]
            ablation_sets[f'no_{group}'] = leave_out

        print('Ablation sets:', ablation_sets)
        return ablation_sets
