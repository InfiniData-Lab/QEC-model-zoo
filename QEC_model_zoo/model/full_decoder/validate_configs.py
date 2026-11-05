import os
from typing import List


def validate_configs(decoder_configs: List[dict]) -> None:
    """
    Validates decoder configs. Raises error if validation fails.

    :param decoder_configs: List of decoder configs.
    """
    possible_decoder_names = [value.lower() for decoder_config in decoder_configs for value in decoder_config.values()]

    validate_decoders(possible_decoder_names)

def validate_decoders(decoders: List[str]) -> None:
    """
    Validates whether decoders are ready to use. Raises error if validation fails.
    Validation checks:
    1. TabPFN requires a token to be set in the environment, if not set, advise user on how to do this.

    :param decoders: List of decoders.
    """
    # 1. Check for valid tabpfn token
    contains_tabpfn = 'tabpfn' in [d.lower() for d in decoders]
    if contains_tabpfn and not validate_tabpfn_token_present():
        raise(ValueError('Usage of TabPFN requires a token to be set in the environment, follow instructions from README in this codebase or refer to https://priorlabs.ai/tabpfn-2'))

def validate_tabpfn_token_present() -> bool:
    """
    Validate whether tabpfn token is present in environment.

    :return: True iff tabpfn token is present in environment.
    """
    return bool(os.environ.get("TABPFN_TOKEN", "").strip())
