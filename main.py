"""
This package defines an organized way of accessing the different features of the HPO for NAS project.
"""
from pathlib import Path

import numpy as np

from src.optimizers.dehb_optimizer import run_dehb
from src.optimizers.nas_opt2 import run_naslib
from src.utils.config import load_yaml_config
from src.utils.nasbench201_configspace import run_rs
from src.utils.output_generator import generate_only_outputs_for_deepcave


def run_optimizer(config: dict, output_path="./"):
    """
    Reads the configuration dictionary and runs an optimizer on the given search space.

    :param config: the configuration for the optimization as a dictionary
    :param output_path: the path for the output files generated by the optimizer
    """
    optimizer = config['optimizer']
    number_of_runs = config['n_runs']
    np.random.seed(config['seed'])

    for _ in range(number_of_runs):
        if optimizer == 'rs':
            run_rs(config, output_path)
        elif optimizer == 'dehb':
            run_dehb(config, output_path)
        if optimizer == 're':
            run_naslib(config, optimizer)
        elif optimizer == 'bananas':
            run_naslib(config, optimizer)
        elif optimizer == 'smac':
            raise NotImplementedError("SMAC has not yet been implemented.")
        else:
            raise NameError('Invalid optimizer name "{}"'.format(optimizer))


def convert_outputs(config: dict, output_path:str):
    """
    Generate DeepCAVE run files under output_path for the optimizer defined in the config file. This method will
    NOT run optimization again, just uses the previously generated run_history files under output_path that will
    be used to generate files in the DeepCAVE format.

    :param config: the configuration for the optimization as a dictionary
    :param output_path: the path for the output files generated by the optimizer
    """
    generate_only_outputs_for_deepcave(config, output_path)


def main(path_to_config: str):
    """
    Reads configuration file, creates the output path for the optimizer and runs an optimization based on these.

    If convert_output_only_mode in the config file is set to True, then instead of running the optimizer, it will
    convert every output run of that optimizer into the deepcave format.
    """
    config: dict = load_yaml_config(path_to_config)
    output_path = f"results/{config['optimizer']}/{config['search_space']}/{config['dataset']}/seed-{config['seed']}"

    # Create directory structure in advance to avoid possible errors
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if not config['convert_output_only_mode']:
        run_optimizer(config=config, output_path=output_path)
    else:
        convert_outputs(config, output_path)


# TODO define a way to run main with command line arguments that can overwrite the config file entries
if __name__ == '__main__':
    main("configs/config.yaml")

