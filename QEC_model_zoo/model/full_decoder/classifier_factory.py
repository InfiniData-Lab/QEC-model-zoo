from pathlib import Path
from typing import Any, Dict, Type, Tuple

from model.full_decoder.base_classifier import BaseClassifier
from model.full_decoder.sequence.lstm_sequence_classifier import LSTMSequenceClassifier
from model.full_decoder.neural.mlp import MLP
from model.full_decoder.neural.tabm import TabMClassifier
from model.full_decoder.neural.tabr import TabRClassifier
from model.full_decoder.tfms.tabiclv2 import TabICLV2Wrapper
from model.full_decoder.tfms.tabpfn import TabPFNWrapper
from model.full_decoder.tree_models.catboost import CatBoost
from model.full_decoder.tree_models.lightgbm import LightGBM
from model.full_decoder.tree_models.xgboost import XGBoost
from utils.file_utils import load_yaml
from utils.string_utils import StringUtils


def find_model_class(model_name: str) -> Type[BaseClassifier]:
    """
    Find the appropriate model class based on its name.

    :param model_name: Normalized name of the model
    :return: Class corresponding to the model name.
    """
    # Tree models
    if model_name == 'xgboost':
        return XGBoost
    if model_name == 'lightgbm':
        return LightGBM
    if model_name == 'catboost':
        return CatBoost
    # Neural models
    if model_name == 'tabm':
        return TabMClassifier
    if model_name == 'tabr':
        return TabRClassifier
    if model_name == 'mlp':
        return MLP
    # Foundation models
    if model_name == 'tabpfnv2' or model_name == 'tabpfnv3':
        return TabPFNWrapper
    if model_name == 'tabiclv2':
        return TabICLV2Wrapper
    # Sequence models
    if model_name == 'lstm':
        return LSTMSequenceClassifier
    else:
        raise NotImplementedError(f"Classifier {model_name} not implemented")


def load_model_config(config_path: Path, override_hp: Dict[str, Any] | None = None) -> Tuple[Type[BaseClassifier], Dict[str, Any], Any]:
    """
    Load a model configuration (class to use, parameter configuration, entire config file).

    :param config_path: Path to model configuration file.
    :param override_hp: Optional overrides for hyperparameters.
    :return: Tuple of (model_class, parameters, configuration file contents).
    """
    if override_hp is None:
        override_hp = {}

    file_contents = load_yaml(config_path)
    params_config = file_contents["hyperparameters"]

    model_name = StringUtils.normalize_string(file_contents["model"])
    model_class = find_model_class(model_name)
    model_params = {}
    # Use default values instead of optimizing
    for hp, config in params_config.items():
        model_params[hp] = config.get("default")
    # Apply hyperparameter overrides
    for key, value in override_hp.items():
        model_params[key] = value

    return model_class, model_params, file_contents
