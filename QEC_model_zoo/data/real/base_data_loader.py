from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd


class BaseDataLoader(ABC):
    """
    Generic abstract class for a dataset loader.
    """
    dataset_metadata: Dict[str, str]
    data_fraction: float
    n_classes: int
    n_features: int
    seed: int
    target_column: str

    def __init__(self, dataset_metadata: Dict[str, Any], data_fraction: float, seed: int) -> None:
        """
        Constructor.

        :param dataset_metadata: Metadata of the dataset
        :param data_fraction: Fraction of data to use
        :param seed: Random seed
        """
        self.dataset_metadata = dataset_metadata
        self.dataset_name = dataset_metadata['dataset_name']
        self.data_fraction = data_fraction
        self.seed = seed
        self.target_column = dataset_metadata['target_col']
        self.feature_sets = dataset_metadata['feature_sets']

    @abstractmethod
    def get_train(self, raw: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training dataset as a tuple of feature matrix and labels.

        :param raw: If true, return raw feature matrix, otherwise feature engineered features.
        :return: Tuple of feature matrix and labels.
        """
        pass

    @abstractmethod
    def get_test(self, raw: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get testing dataset as a tuple of feature matrix and labels.

        :param raw: If true, return raw feature matrix, otherwise feature engineered features.
        :return: Tuple of feature matrix and labels.
        """
        pass

    @abstractmethod
    def get_val(self, raw: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get validation dataset as a tuple of feature matrix and labels.

        :param raw: If true, return raw feature matrix, otherwise feature engineered features.
        :return: Tuple of feature matrix and labels.
        """
        pass

    def get_test_shot_info(self) -> Tuple[List[int], List[int]]:
        """
        Get test shot information (tuple of list of shot ids and corresponding number of rounds in that shot).

        :return: Test shot info.
        """
        pass
