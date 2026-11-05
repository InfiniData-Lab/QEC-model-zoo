import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np


class BaseClassifier(ABC):
    """
    Base class for classification models.
    Defines common functionality and some abstract methods subclasses should implement
    """
    params: Dict[str, Any]
    args: Any
    model: Any
    model_name: str
    is_trained: bool

    def __init__(self, params: Dict[str, Any], args: Any):
        self.params = params
        self.args = args
        self.is_trained = False
        self.model_name = args.model_name

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> None:
        """
        Fit the model to the given training data.

        X can be a standard 2D tabular matrix or a higher-dimensional tensor for sequence models.
        """
        self.is_trained = True # Make sure to call this in subclasses

    def fit_with_timing(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None,
    ) -> float:
        """
        Fit the model and give execution time.

        X can be a standard 2D tabular matrix or a higher-dimensional tensor for sequence models.
        """
        time_start = time.perf_counter()
        self.fit(X_train, y_train, X_val, y_val)
        time_end = time.perf_counter()
        return time_end - time_start

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for the provided features.

        X can be a standard 2D tabular matrix or a higher-dimensional tensor for sequence models.
        """
        pass

    def predict_with_timing(self, X: np.ndarray, batch_size = None) -> Tuple[np.ndarray, float]:
        """
        Predict labels for the provided features with execution time (seconds).
        If batch size is set, do predictions in batches, execution time is summed up over batches.

        :param X: Feature matrix to predict labels for.
        :param batch_size: Batch size, defaults to None (all predictions at once).
        :return: Tuple of predicted labels and execution time.
        """
        if batch_size is None: # Default: all predictions done in single batch
            time_start = time.perf_counter()
            predictions = self.predict(X)
            time_end = time.perf_counter()
            return predictions, time_end - time_start

        # Prediction in batches, timing is summed up over all batches
        chunks = []
        timings = 0.0

        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            chunk, chunk_timing = self.predict_with_timing(X[start:end])
            chunks.append(chunk)
            timings += chunk_timing

        predictions = np.concatenate(chunks, axis=0)
        return predictions, timings

    def config_to_dict(self) -> Dict[str, Any]:
        """
        Get a dictionary of the model configuration (params and args).

        :return: Dictionary of the model configuration (params and args).
        """
        return {
            'params': self.params,
            'args': self._convert_to_serializable(self.args),
        }

    def _convert_to_serializable(self, obj: Any) -> Any:
        """
        Convert an object to a serializable object (for model args).

        :param obj: Object to be serialized.
        :return: Serialized object (primitive or dictionary).
        """
        if hasattr(obj, "__dict__"):
            return dict([(k, self._convert_to_serializable(v)) for k, v in vars(obj).items()])
        if isinstance(obj, list):
            return [self._convert_to_serializable(el) for el in obj]
        return obj  # Default for primitive types

    def save_model(self, path: Path) -> Path:
        """
        Save the trained model to the given path.
        Should be overridden by subclasses, otherwise it will skip this step.

        :param path: Path to directory to save model to.
        :return: Path to file the model is saved to.
        """
        pass

    @classmethod
    def load_model(cls, path: Path, params: Dict[str, Any], model_args: Any) -> "BaseClassifier":
        """
        Load the model from the given path.
        If not overridden, exception will be raised.

        :param path: Path to load model from.
        :return: Instance of the model
        """
        raise NotImplemented('Loading model not implemented for this model.')
