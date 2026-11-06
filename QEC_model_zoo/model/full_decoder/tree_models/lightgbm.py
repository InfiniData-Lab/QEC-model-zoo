import gzip
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np

from model.full_decoder.base_classifier import BaseClassifier
import lightgbm as lgb

class LightGBM(BaseClassifier):
    """
    LightGBM (https://lightgbm.readthedocs.io/en/latest/) wrapper class
    """
    # TabZilla: add default number of boosting rounds
    # default_epochs = 500
    def __init__(self, params, args):
        if lgb is None:
            raise ImportError("LightGBM is not installed.")
        super().__init__(params, args)

        self.params["verbosity"] = -1

        if args.objective == "regression":
            self.params["objective"] = "regression"
            self.params["metric"] = "mse"
        elif args.objective == "classification":
            self.params["objective"] = "multiclass"
            self.params["num_class"] = args.num_classes
            self.params["metric"] = "multiclass"
        elif args.objective == "binary":
            self.params["objective"] = "binary"
            self.params["metric"] = "auc"

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None,
    ) -> None:
        # Handle categorical features if present
        cat_features = "auto"
        if hasattr(self.args, 'cat_idx') and self.args.cat_idx is not None and len(self.args.cat_idx) > 0:
            cat_features = self.args.cat_idx

        train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)

        valid_sets = []
        valid_names = []

        if X_val is not None and y_val is not None:
            val = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, reference=train)
            valid_sets.append(val)
            valid_names.append("eval")

        self.model = lgb.train(
            self.params,
            train,
            num_boost_round=self.params['epochs'],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(self.args.early_stopping_rounds),
                lgb.log_evaluation(self.args.logging_period),
            ],
        )

        super().fit(X_train, y_train, X_val, y_val)  # Simply sets is_trained to true

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.

        :param X_test: Feature matrix (n_samples, n_features)
        :return: Class probabilities (n_samples, n_classes)
        """
        probabilities = self.model.predict(X_test)

        if self.args.objective == "binary":
            probabilities = probabilities.reshape(-1, 1)
            probabilities = np.concatenate((1 - probabilities, probabilities), 1)

        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        # For LightGBM train object, predict returns probabilities, not classes
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    @classmethod
    def define_trial_parameters(cls, trial: Any, args: Any) -> Dict[str, Any]:
        """
        Define trial parameters.

        :param trial: Trial object
        :param args: Arguments
        :return: Trial parameters
        """
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 2, 4096, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
        return params

    # TabZilla: add function for seeded random params and default params
    @classmethod
    def get_random_parameters(cls, seed: int) -> Dict[str, Any]:
        """
        Get random parameters.

        :param seed: Random seed
        :return: Random parameters
        """
        rs = np.random.RandomState(seed)
        params = {
            "num_leaves": int(np.round(np.power(2, rs.uniform(1, 12)))),
            "lambda_l1": np.power(10, rs.uniform(-8, 1)),
            "lambda_l2": np.power(10, rs.uniform(-8, 1)),
            "learning_rate": 3.0 * np.power(10, rs.uniform(-2, 1)),
        }
        return params

    @classmethod
    def default_parameters(cls) -> Dict[str, Any]:
        """
        Get default parameters.

        :return: Default parameters
        """
        params = {
            "num_leaves": 512,
            "lambda_l1": 1e-3,
            "lambda_l2": 1e-3,
            "learning_rate": 0.08,
        }
        return params

    def save_model(self, path: Path) -> Path:
        path = Path(path)
        compressed_path = Path(str(path) + ".gz")

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "model.txt"

            # Save the exact same LightGBM representation
            self.model.save_model(str(raw_path))

            # Compress it losslessly
            with raw_path.open("rb") as src:
                with gzip.open(compressed_path, "wb", compresslevel=9) as dst:
                    shutil.copyfileobj(src, dst)

        return compressed_path

    @classmethod
    def load_model(
            cls,
            path: Path,
            params: Dict[str, Any],
            model_args: Any
    ) -> "LightGBM":
        model = cls(params=params, args=model_args)
        path = Path(path)

        # Try both compressed and uncompressed checkpoints
        if path.suffix == ".gz":
            alternative_path = path.with_suffix("")
        else:
            alternative_path = Path(str(path) + ".gz")

        if not path.is_file():
            if alternative_path.is_file():
                path = alternative_path
            else:
                raise FileNotFoundError(
                    f"Neither checkpoint exists:\n"
                    f"  {path}\n"
                    f"  {alternative_path}"
                )

        if path.suffix == ".gz":
            with tempfile.TemporaryDirectory() as tmpdir:
                raw_path = Path(tmpdir) / "model.txt"

                with gzip.open(path, "rb") as src:
                    with raw_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)

                model.model = lgb.Booster(model_file=str(raw_path))
        else:
            model.model = lgb.Booster(model_file=str(path))

        model.is_trained = True
        return model
