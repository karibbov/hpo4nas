"""
This package defines a DEHB optimizer for the selected search space and uses DeepCAVE to analyze the results of the
optimization.
"""
import fnmatch
import os
from pathlib import Path

import numpy as np
from ConfigSpace import Configuration

from dehb import DEHB
from src.utils.nasbench201_configspace import configure_nasbench201, query_nasbench201
from src.utils.output_generator import generate_deepcave_output


def _target_function(cs_config: Configuration, budget: float, **kwargs):
    """
    Interface for target function that DEHB optimizes. It is the problem that needs to be solved,
    or the function to be optimized

    :param: cs_config: the architecture to query defined as a ConfigSpace object
    :param: budget: the current epoch to query
    :return: regret, training time and some additional information
    """
    budget = int(budget)

    train_loss, valid_loss, test_loss, train_regret, valid_regret, test_regret, train_time = query_nasbench201(
        cs_config, kwargs['dataset'], budget)

    result = {
        "fitness": valid_regret,  # this is what DEHB minimizes
        "cost": train_time,
        "info": {
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "test_loss": test_loss,
            "train_regret": train_regret,
            "test_regret": test_regret,
        }
    }
    return result


def _create_run_id(output_path: Path, add_prefix=False, budget=None):
    """
    This makes sure that each DEHB run gets a separate run folder with a unique name.

    Warning: Only a single underscore '_' is allowed in the run folder's name, so do not use one in the given budget!

    :param: add_prefix: whether to prefix each run's folder name with 'DEHB-'
    :param: budget: the maximum budget for the optimization (ex.: 2h, for two hours of wallclock time)
    :param: output_path: the folder to output the results
    :return: a new path having an id to distinguish this run from earlier ones
    """
    run_name = 'run_1'
    runs = fnmatch.filter(os.listdir(output_path), '*run*')
    if len(runs) > 0:
        next_id = np.max(np.array([run.split('_') for run in runs])[:, 1].astype(np.int)) + 1
        run_name = f'run_{next_id}'
    if add_prefix:
        run_name = f'DEHB-{run_name}'
    if budget is not None:
        run_name = f'{budget}-{run_name}'

    return output_path / run_name


def run_dehb(run_config: dict, output_path: Path, dataset: str, format_for_deepcave=True):
    """
    Run DEHB on NAS-Bench-201 with the settings defined in the config dictionary

    :param: dataset: the dataset as a string
    :param: run_config: the configuration for the optimization as a dictionary
    :param: output_path: the directory to store the outputs in
    :param: format_for_deepcave: Whether to generate DeepCAVE run files besides the final output of this DEHB run
    """
    output_path = _create_run_id(output_path=output_path,
                                 add_prefix=True,
                                 budget=f'{int(run_config["dehb"]["runtime_limit"] / 60)}min')

    cs = configure_nasbench201()

    dehb = DEHB(
        f=_target_function,
        cs=cs,
        dimensions=len(cs.get_hyperparameters()),
        min_budget=run_config['dehb']['min_budget'],
        max_budget=run_config['dehb']['max_budget'],
        n_workers=run_config['dehb']['n_workers'],
        output_path=output_path
    )

    trajectory, runtime, history = dehb.run(
        total_cost=run_config['dehb']['runtime_limit'],
        verbose=True,
        save_intermediate=False,
        dataset=dataset,
        name=run_config['dehb']['name'],
        max_budget=run_config['dehb']['max_budget']
    )

    if format_for_deepcave:
        generate_deepcave_output(run_config=run_config, output_path=output_path)

    print(f'Output of the run saved under:\n{output_path}')
