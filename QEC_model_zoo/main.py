import argparse
from pathlib import Path

from benchmark.utils.full_decoder.baseline_experiment_google import BaselineExperimentGoogle
from benchmark.utils.full_decoder.eval_full_decoder_experiment import EvalFullDecoderExperiment
from benchmark.utils.full_decoder.feature_ablation_experiment import FeatureAblationExperiment
from benchmark.utils.full_decoder.train_full_decoder_experiment import TrainFullDecoderExperiment
from benchmark.utils.full_decoder.train_sequence_full_decoder_experiment import TrainSequenceFullDecoderExperiment
from data.croissant_data_loader import CroissantDataLoader
from model.full_decoder.validate_configs import validate_configs, validate_decoders
from utils.file_utils import load_yaml
from utils.random_utils import set_seed
from utils.string_utils import StringUtils


def get_args() -> argparse.Namespace:
    """
    Get arguments from command line.

    :return: Namespace containing CLI arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", type=str, required=True, default='experiments/configs/experiment/train/train_tree_models_dicarlo.yaml', help="Path to experiment configuration file.")
    parser.add_argument('--decoders', type=str, required=False, nargs="*", help="Optionally filter for decoders matching specific name, defaults to all defined in experiment.yaml.")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of dataset used, defaults to 1.0.")

    return parser.parse_args()

def main() -> None:
    """
    Entrypoint for QEC model zoo.
    """
    cli_args = get_args()

    # 1. Load experiment configuration
    experiment_config_path = Path(cli_args.experiment)
    experiment_config = load_yaml(experiment_config_path)
    croissant_datapath = experiment_config['dataset_id']
    experiment_type = experiment_config['experiment_type']
    decoder_type = experiment_config['decoder_type']
    random_seed = experiment_config['random_seed']
    decoder_configs= experiment_config.get('decoder_configs', None)

    # Validation checks for certain decoders
    if decoder_configs is not None and len(decoder_configs) > 0:
        validate_configs(decoder_configs)

    experiment_name = experiment_config['experiment_name']
    tuning_config = experiment_config.get('tuning_config', None)
    data_frac = experiment_config.get('data_fraction', 1.0)
    max_shots = experiment_config.get('max_shots', None)

    # Optional filter for decoders, defaults to all defined in experiment
    decoder_filter = getattr(cli_args, 'decoders', None)
    decoder_filter = decoder_filter if decoder_filter and len(decoder_filter) > 0 else None

    if cli_args.data_fraction < data_frac:
        data_frac = cli_args.data_fraction
    experiment_repetitions = experiment_config.get('repetitions', 1)

    # 2. Set random seed for reproducibility
    set_seed(random_seed)

    # 3. Get dataset loader (also pre-processes data)
    if decoder_type == 'full-decoder':
        dataloader = CroissantDataLoader(croissant_datapath, seed=random_seed, data_fraction=data_frac, max_shots=max_shots)

    # 4. Get experiment type
    if experiment_type == 'train' and decoder_type == 'full-decoder': # Training full decoder
        if experiment_repetitions > 1:
            raise ValueError('Cannot do multiple repetitions of training')
        experiment = TrainFullDecoderExperiment(experiment_name, dataloader, decoder_configs, random_seed, tuning_config, decoder_filter=decoder_filter)
    elif experiment_type == 'train-sequence' and decoder_type == 'full-decoder': # Simplified training flow for sequence models.
        if experiment_repetitions > 1:
            raise ValueError('Cannot do multiple repetitions of training')
        experiment = TrainSequenceFullDecoderExperiment(experiment_name, dataloader, decoder_configs, random_seed, decoder_filter=decoder_filter)
    elif 'inference' in experiment_type and decoder_type == 'full-decoder': # Evaluating full decoder
        retrain = 'tfm' in experiment_type # Tabular foundation models don't have a checkpoint so must be retrained
        experiment = EvalFullDecoderExperiment(experiment_name, dataloader, decoder_configs, random_seed, retrain_model=retrain, repetitions=experiment_repetitions, decoder_filter=decoder_filter)
    elif 'baseline' in experiment_type and decoder_type == 'full-decoder':
        if decoder_filter:
            raise ValueError('Decoder filter cannot be applied to this experiment type as it is run for a single decoder.')
        experiment = BaselineExperimentGoogle(experiment_name, dataloader, random_seed, repetitions=experiment_repetitions)
    elif 'ablation' in experiment_type and decoder_type == 'full-decoder':
        default_decoders = ['mlp', 'tabm', 'tabr', 'xgboost', 'lightgbm', 'catboost', 'tabiclv2', 'tabpfnv2']
        decoders = experiment_config.get('decoders', default_decoders)
        if decoder_filter:
            decoders = StringUtils.filter_by_match(decoders, decoder_filter)
        validate_decoders(decoders)
        out_path = Path("experiments/results/ablations") / experiment_name / dataloader.dataset_name
        experiment = FeatureAblationExperiment(dataloader, decoders, out_path, seed=random_seed)
    else:
        raise ValueError('Invalid experiment configuration.')

    # 5. Run experiment
    experiment.run()

if __name__ == '__main__':
    main()
