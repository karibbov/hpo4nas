"""
This package gathers the main features that are offered by the HPO4NAS project.
"""
import os
from pathlib import Path

import numpy as np

from constants import APP_ROOT_DIR
from src.optimizers.dehb_optimizer import run_dehb
from src.optimizers.nas_opt2 import run_naslib
from src.utils.run_config import load_yaml_config
from src.optimizers.random_search_optimizer import run_rs
from src.utils.output_generator import generate_only_outputs_for_deepcave


def run_optimizer(run_config: dict):
    """
    Reads the configuration dictionary and runs an optimizer on the search space with the settings given in the
    configuration. The run is performed for each dataset N times. This gets us an estimate for the variance of the
    performance for the selected optimizer.

    :param: run_config: the configuration for the optimization as a dictionary
    """
    optimizer = run_config['optimizer']
    search_space = run_config['search_space']
    number_of_runs = run_config['n_runs']
    datasets = run_config['datasets']
    seed = run_config["seed"]
    output_root_dir = run_config['output_root_dir']
    np.random.seed(run_config['seed'])

    for dataset in datasets:
        output_path = Path(os.path.join(APP_ROOT_DIR,
                                        output_root_dir,
                                        optimizer,
                                        search_space,
                                        dataset,
                                        f'seed-{seed}'))

        for _ in range(number_of_runs):
            if optimizer == 'rs':
                run_rs(run_config=run_config, output_path=output_path, dataset=dataset)
            elif optimizer == 'dehb':
                run_dehb(run_config=run_config, output_path=output_path, dataset=dataset)
            elif optimizer == 're':
                run_naslib(run_config, optimizer)
            elif optimizer == 'npenas':
                run_naslib(run_config, optimizer)
            elif optimizer == 'bananas':
                run_naslib(run_config, optimizer)
            elif optimizer == 'smac':
                raise NotImplementedError("SMAC has not yet been implemented.")
            else:
                raise NameError('Invalid optimizer name "{}"'.format(optimizer))


def convert_outputs(run_config: dict):
    """
    Generate DeepCAVE run files under output_path for the optimizer defined in the config file. This method will
    NOT run optimization again, just uses the previously generated run_history files under output_path that will
    be used to generate files in the DeepCAVE format.

    :param: run_config: the configuration for the optimization as a dictionary
    """
    generate_only_outputs_for_deepcave(run_config=run_config)


def main(path_to_run_config: Path):
    """
    Reads configuration file, creates the output path for the optimizer and runs an optimization based on these.

    If convert_output_only_mode in the config file is set to True, then instead of running the optimizer, it will
    convert every output run of that optimizer into the deepcave format.

    :param: path_to_config: path to the configuration file. (ex.: config.yaml)
    """
    run_config: dict = load_yaml_config(path_to_yaml=path_to_run_config)

    if not run_config['convert_output_only_mode']:
        run_optimizer(run_config=run_config)
    else:
        convert_outputs(run_config=run_config)


# TODO define a way to run main with command line arguments that can overwrite the config file entries
if __name__ == '__main__':
    main(path_to_run_config=Path(APP_ROOT_DIR) / 'configs' / 'run_config.yaml')
