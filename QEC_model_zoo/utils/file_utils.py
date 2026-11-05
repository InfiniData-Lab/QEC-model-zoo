from pathlib import Path
from typing import Any

import yaml


def load_yaml(config_path: Path) -> Any:
    """
    Load configuration yaml file.
    
    :param config_path: Path to yaml file.
    :return: Object representing the configuration.
    """
    return yaml.load(open(config_path), Loader=yaml.SafeLoader)
    