"""
This package defines a DEHB optimizer for the selected search space and uses DeepCAVE to analyze the results of the
optimization.
"""
from pathlib import Path
import ConfigSpace
import numpy as np

from naslib.search_spaces import NasBench201SearchSpace
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils import get_dataset_api
from dehb import DEHB
from src.utils.nasbench201_configspace import configure_nasbench201, configuration2op_indices, \
    nasbenc201_optimal_results as optimal
from src.utils.output_converter import DEHBRun


def analyze_run(output_path: Path, cs: ConfigSpace):
    """
    Use deep cave to visually interpret the data.

    :param cs: the search space as a configspace object
    :param output_path: the path to the output folder
    """
    run = DEHBRun(name="dehb", configspace=cs).from_path(output_path)
    # TODO DEHBRun.from_path(output_path) ?
    run.save(output_path)


def _target_function(cs: ConfigSpace, budget: float, **kwargs):
    """
    Interface for target function that DEHB optimizes. It is the problem that needs to be solved,
    or the function to be optimized

    :param cs: the architecture to query defined as a ConfigSpace object
    :param budget: the current epoch to query
    :return: regret, training time and some additional information
    """
    budget = int(budget)
    op_indices = configuration2op_indices(cs)
    # model represents the sampled architecture
    model = NasBench201SearchSpace()
    model.set_op_indices(op_indices)
    dataset_api = get_dataset_api(search_space='nasbench201', dataset=kwargs['dataset'])
    train_acc = model.query(Metric.TRAIN_ACCURACY, dataset=kwargs['dataset'], dataset_api=dataset_api, epoch=budget)
    valid_acc = model.query(Metric.VAL_ACCURACY, dataset=kwargs['dataset'], dataset_api=dataset_api, epoch=budget)
    test_acc = model.query(Metric.TEST_ACCURACY, dataset=kwargs['dataset'], dataset_api=dataset_api, epoch=budget)
    train_time = model.query(Metric.TRAIN_TIME, dataset=kwargs['dataset'], dataset_api=dataset_api, epoch=budget)

    regret = optimal["cifar10_test_acc"] - test_acc

    # TODO possibly move this to a config file to make the dehb run conversion for deep cave more generic
    result = {
        "fitness": regret,  # this is what DE/DEHB minimizes
        "cost": train_time,
        "info": {
            "train_accuracy": train_acc,
            "validation_accuracy": valid_acc,
            "test_accuracy": test_acc,
            "budget": budget,
        }
    }
    return result


def run_dehb(config: dict, output_path: Path):
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
        brackets=config['dehb']['brackets'],
        verbose=True,
        save_intermediate=False,
        dataset=config['dataset'],
    )

    analyze_run(output_path, cs)

