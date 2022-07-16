"""
This package defines a DEHB optimizer for the selected search space and uses DeepCAVE to analyze the results of the
optimization.
"""
from pathlib import Path
import numpy as np
from ConfigSpace import Configuration

from dehb import DEHB
from src.utils.nasbench201_configspace import configure_nasbench201, query_nasbench201
from src.utils.output_converter import DEHBRun


def analyze_run(output_path: str):
    """
    Use deep cave to visually interpret the data.

    :param cs: the search space as a configspace object
    :param output_path: the path to the output folder
    """
    run = DEHBRun.from_path(Path(output_path))
    run.save(output_path)


def _target_function(cs_config: Configuration, budget: float, **kwargs):
    """
    Interface for target function that DEHB optimizes. It is the problem that needs to be solved,
    or the function to be optimized

    :param cs: the architecture to query defined as a ConfigSpace object
    :param budget: the current epoch to query
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


def run_dehb(config: dict, output_path: str):
    """
    Run DEHB on NAS-Bench-201 using the DATASET defined in this package

    :param config: the configuration for the optimization as a dictionary
    :param output_path: the directory to store the outputs in
    """

    np.random.seed(config['seed'])
    cs = configure_nasbench201()

    dehb = DEHB(
        f=_target_function,
        cs=cs,
        dimensions=len(cs.get_hyperparameters()),
        min_budget=config['dehb']['min_budget'],
        max_budget=config['dehb']['max_budget'],
        n_workers=config['dehb']['n_workers'],
        output_path=output_path
    )

    trajectory, runtime, history = dehb.run(
        total_cost=config['dehb']['wallclock'],
        verbose=True,
        save_intermediate=False,
        dataset=config['dataset'],
        name=config['dehb']['name'],
        max_budget=config['dehb']['max_budget']
    )

    analyze_run(output_path)
