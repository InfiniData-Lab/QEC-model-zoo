from pathlib import Path
from typing import Dict, Any

import numpy as np
import optuna
import xgboost as xgb
from sklearn.exceptions import NotFittedError

from model.full_decoder.base_classifier import BaseClassifier


class XGBoost(BaseClassifier):
    """
    Wrapper class for XGBoost classifier (https://xgboost.readthedocs.io/en/stable/)
    """
    # TabZilla: add default number of boosting rounds
    # default_epochs = 500

    def __init__(self, params, args):
        if xgb is None:
            raise ImportError("XGBoost is not installed.")

        # Filter out non-XGBoost parameters
        self.epochs = params['epochs']
        xgb_params = {k: v for k, v in params.items() if k in [
            'max_depth', 'alpha', 'lambda', 'eta', 'learning_rate',
            'verbosity', 'device', 'tree_method', 'objective', 'eval_metric',
        ]}
        super().__init__(xgb_params, args)

        if hasattr(args, 'use_gpu') and args.use_gpu:
            xgb_params.update({
                "device": "cuda",
                "tree_method": "hist"
            })

        if args.objective == "regression":
            self.params["objective"] = "reg:squarederror"
            self.params["eval_metric"] = "rmse"
        elif args.objective == "classification":
            self.params["objective"] = "multi:softprob"
            self.params["num_class"] = args.num_classes
            self.params["eval_metric"] = "mlogloss"
            self.n_classes_ = args.num_classes  # Add this line
        elif args.objective == "binary":
            self.params["objective"] = "binary:logistic"
            self.params["eval_metric"] = "auc"
            self.n_classes_ = 2  # Add this line for binary classification

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None,
    ):
        train = xgb.DMatrix(X_train, label=y_train)

        eval_list = []
        if X_val is not None and y_val is not None:
            val = xgb.DMatrix(X_val, label=y_val)
            eval_list = [(val, "eval")]

        self.model = xgb.train(
            self.params,
            train,
            num_boost_round=self.epochs,
            evals=eval_list,
            early_stopping_rounds=self.args.early_stopping_rounds,
            verbose_eval=self.args.logging_period,
        )
        self.classes_ = np.arange(self.n_classes_)
        super().fit(X_train, y_train, X_val, y_val)  # Simply sets is_trained to true
        return [], []

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        :param X_test: Test data
        :return: Predicted class probabilities for test data.
        """
        if not self.model:
            raise NotFittedError("This XGBoost instance is not fitted yet.")

        # Convert NumPy array to DMatrix before prediction
        if isinstance(X_test, np.ndarray):
            X_dmatrix = xgb.DMatrix(X_test)
        elif isinstance(X_test, xgb.DMatrix):
            X_dmatrix = X_test  # Already a DMatrix
        else:
            # Handle other potential types or raise an error
            raise TypeError(
                f"Unsupported data type for XGBoost predict_proba: {type(X_test)}. Expected numpy.ndarray or xgb.DMatrix.")

        raw_probabilities = self.model.predict(X_dmatrix)

        # Assuming raw_probabilities are P(Y=1) for binary classification
        # and have shape (n_samples,)
        if raw_probabilities.ndim == 1:
            probabilities = np.vstack([1 - raw_probabilities, raw_probabilities]).T
        elif raw_probabilities.ndim == 2 and raw_probabilities.shape[1] == 1:  # if it returns [[p1], [p2], ...]
            raw_probabilities = raw_probabilities.flatten()
            probabilities = np.vstack([1 - raw_probabilities, raw_probabilities]).T
        elif raw_probabilities.ndim == 2 and raw_probabilities.shape[1] == self.n_classes_:  # For multi-class
            probabilities = raw_probabilities
        else:
            raise ValueError(f"Unexpected shape for XGBoost probabilities: {raw_probabilities.shape}")

        return probabilities

    @classmethod
    def define_trial_parameters(cls, trial: optuna.Trial, args: Any) -> Dict[str, Any]:
        """
        Define trial parameters for XGBoost HP tuning.

        :param trial: Tuning trial
        :param args: Arguments (unused)
        :return: Trial parameters
        """
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 12, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        }
        return params

    # TabZilla: add function for seeded random params and default params
    @classmethod
    def get_random_parameters(cls, seed: int) -> Dict[str, Any]:
        """
        Get random parameters for XGBoost.

        :param seed: Random seed.
        :return: Dictionary of random parameters.
        """
        rs = np.random.RandomState(seed)
        params = {
            "max_depth": int(np.round(np.power(2, rs.uniform(1, np.log2(12))))),
            "alpha": np.power(10, rs.uniform(-8, 0)),
            "lambda": np.power(10, rs.uniform(-8, 0)),
            "eta": 3.0 * np.power(10, rs.uniform(-2, -1)),
        }
        return params

    @classmethod
    def default_parameters(cls) -> Dict[str, Any]:
        """
        Get default parameters for XGBoost.

        :return: Dictionary of default parameters for XGBoost.
        """
        return {
            "max_depth": 5,
            "alpha": 1e-4,
            "lambda": 1e-4,
            "eta": 0.08,
        }

    def save_model(self, path: Path) -> Path:
        self.model.save_model(str(path))

        return path

    @classmethod
    def load_model(cls, path: Path, params: Dict[str, Any], model_args: Any) -> "XGBoost":
        model = cls(params=params, args=model_args)

        booster = xgb.Booster()
        booster.load_model(str(path))

        model.model = booster
        model.classes_ = np.arange(model.n_classes_)
        model.is_trained = True

        return model
