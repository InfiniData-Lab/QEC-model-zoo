import inspect
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.exceptions import NotFittedError

from data.croissant_data_loader import CroissantDataLoader
from model.full_decoder.base_classifier import BaseClassifier
import catboost as cat

class CatBoost(BaseClassifier):
    """
    Wrapper clas for CatBoost (https://catboost.ai/)
    """
    # TabZilla: add default number of boosting rounds
    # default_epochs = 500
    def __init__(self, params, args):
        if cat is None:
            raise ImportError("CatBoost is not installed.")
        super()
        super().__init__(params, args)

        # Get valid parameter names for CatBoostClassifier's constructor
        valid_constructor_params = set(inspect.signature(cat.CatBoostClassifier).parameters.keys())

        # Prepare a dictionary for parameters to be passed to CatBoostClassifier
        catboost_final_params = {}

        # Make a copy of input params to modify
        current_params = self.params.copy()

        # 1. Handle specific name mappings
        if 'epochs' in current_params:
            if 'iterations' in valid_constructor_params:
                catboost_final_params['iterations'] = current_params.pop('epochs')
            else:  # Should not happen if 'iterations' is standard
                current_params.pop('epochs')

        if 'patience' in current_params:
            if 'early_stopping_rounds' in valid_constructor_params:
                catboost_final_params['early_stopping_rounds'] = current_params.pop('patience')
            else:  # Should not happen if 'early_stopping_rounds' is standard
                current_params.pop('patience')

        # 2. Filter remaining params against valid CatBoost constructor params
        for param_name, param_value in current_params.items():
            if param_name in valid_constructor_params:
                catboost_final_params[param_name] = param_value

        # 3. Set/override specific arguments from self.args or defaults

        # train_dir
        if hasattr(self.args, 'dataset') and self.args.dataset:
            dataset_identifier = self.args.dataset
        elif hasattr(self.args, 'task') and self.args.task is not None:
            dataset_identifier = "task_" + str(self.args.task)
        else:
            dataset_identifier = "unknown_run"

        # # Ensure output directory exists
        # catboost_train_dir = os.path.join("output", "CatBoost", dataset_identifier, "catboost_info")
        # os.makedirs(catboost_train_dir, exist_ok=True)
        # catboost_final_params['train_dir'] = catboost_train_dir

        # allow_writing_files
        catboost_final_params['allow_writing_files'] = False

        # verbose
        if 'verbose' not in catboost_final_params:  # CatBoost default is 1, often 0 is preferred for less log spam
            catboost_final_params['verbose'] = 1

            # random_seed
        if 'random_seed' not in catboost_final_params:
            catboost_final_params['random_seed'] = self.args.seed if hasattr(self.args, 'seed') else 42

        # cat_features (from self.args.cat_idx)
        if 'cat_features' in valid_constructor_params:
            if hasattr(self.args, 'cat_idx') and self.args.cat_idx is not None and len(self.args.cat_idx) > 0:
                catboost_final_params["cat_features"] = self.args.cat_idx
            elif 'cat_features' not in catboost_final_params:  # Only default to None if not already provided
                catboost_final_params["cat_features"] = None

        self.model = cat.CatBoostClassifier(**catboost_final_params)
        self.n_classes_ = None  # Will be set in fit

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> None:
        # Store n_classes_
        self.n_classes_ = len(np.unique(y_train))

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        cat_features_for_fit = self.model.get_params().get('cat_features')

        self.model.fit(X_train, y_train, eval_set=eval_set, cat_features=cat_features_for_fit)
        super().fit(X_train, y_train, X_val, y_val)  # Simply sets is_trained to true

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.model, 'classes_'):  # or some other check to ensure model is fitted
            raise NotFittedError("This CatBoost instance is not fitted yet.")

        return self.model.predict(X)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for test data.

        :param X_test: Test feature matrix.
        :return: Predicted class probabilities.
        """
        if not hasattr(self.model, 'classes_'):  # or some other check to ensure model is fitted
            raise NotFittedError("This CatBoost instance is not fitted yet.")

        raw_probabilities = self.model.predict_proba(X_test)

        # Ensure probabilities are in the shape (n_samples, n_classes)
        if raw_probabilities.ndim == 1 and self.n_classes_ == 2:
            probabilities = np.vstack([1 - raw_probabilities, raw_probabilities]).T
        else:
            probabilities = raw_probabilities

        return probabilities

    @classmethod
    def define_trial_parameters(cls, trial: Any, args: Any) -> Dict[str, Any]:
        """
        Define trial parameters.

        :param trial: Trial object.
        :param args: Argument list.
        :return: Dictionary of trial parameters.
        """
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 12, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
        }
        return params

    # TabZilla: add function for seeded random params and default params
    @classmethod
    def get_random_parameters(cls, seed: int) -> Dict[str, Any]:
        """
        Get random parameters.

        :param seed: Random seed.
        :return: Dictionary of random parameters.
        """
        rs = np.random.RandomState(seed)
        params = {
            "learning_rate": 3.0 * np.power(10, rs.uniform(-2, -1)),
            "max_depth": int(np.round(np.power(2, rs.uniform(1, np.log2(12))))),
            "l2_leaf_reg": 0.5 * np.power(60, rs.uniform(0, 1)),
        }
        return params

    @classmethod
    def default_parameters(cls) -> Dict[str, Any]:
        """
        Default parameters.

        :return: Default parameters.
        """
        params = {
            "learning_rate": 0.08,
            "max_depth": 5,
            "l2_leaf_reg": 5,
        }
        return params

    def save_model(self, path: Path) -> Path:
        self.model.save_model(str(path))

        return path

    @classmethod
    def load_model(cls, path: Path, params: Dict[str, Any], model_args: Any) -> "CatBoost":
        model = cls(params=params, args=model_args)

        model.model.load_model(str(path))

        if hasattr(model.model, "classes_"):
            model.n_classes_ = len(model.model.classes_)
        else:
            model.n_classes_ = getattr(model_args, "num_classes", None)

        model.is_trained = True

        return model

