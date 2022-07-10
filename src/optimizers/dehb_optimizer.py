"""
This package defines a DEHB optimizer for the selected search space and uses DeepCAVE to analyze the results of the
optimization.
"""

import ConfigSpace
import numpy as np

from naslib.search_spaces import NasBench201SearchSpace
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils import get_dataset_api
from dehb import DEHB
from src.utils.nasbench201_configspace import configure_nasbench201, configuration2op_indices, \
    nasbenc201_optimal_results as optimal
from src.utils.output_converter import DEHBRun


def _target_function(config: ConfigSpace, budget: float, **kwargs):
    """
    Interface for target function that DEHB optimizes. It is the problem that needs to be solved,
    or the function to be optimized

    :param config: the architecture to query defined as a ConfigSpace object
    :param budget: the current epoch to query
    :return: regret, training time and some additional information
    """
    budget = int(budget)
    op_indices = configuration2op_indices(config)
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


def analyze_run(output_path, configspace):
    """
    Use deep cave to visually interpret the data.

    :param configspace: the search space as a configspace object
    :param output_path: the path to the output folder
    """
    run = DEHBRun(name="dehb", configspace=configspace).from_path(output_path)
    run.save(output_path)


def run_dehb(search_space, dataset, seed, output_path, min_budget=3, max_budget=199, brackets=4):
    """
    Run DEHB on NAS-Bench-201 using the DATASET defined in this package

    :param seed: the seed for the optimizer
    :param search_space: string representation of the search space
    :param output_path: the directory to store the outputs in
    :param dataset: a string specifying the dataset to train/validate/test the architectures on
    :param min_budget: minimum epoch dehb should use as the lowest fidelity
    :param max_budget: maximum epoch dehb should use as the highest fidelity
    :param brackets: the number of brackets to use with dehb
    """
    np.random.seed(seed)
    cs = configure_nasbench201()
    n_dimension = len(cs.get_hyperparameters())
    dehb = DEHB(
        f=_target_function,
        cs=cs,
        dimensions=n_dimension,
        min_budget=min_budget,
        max_budget=max_budget,
        n_workers=1,
        output_path=output_path
    )

    trajectory, runtime, history = dehb.run(
        brackets=brackets,
        verbose=True,
        save_intermediate=False,
        dataset=dataset,
    )

    analyze_run(output_path, cs)

    last_eval = history[-1]
    config, score, cost, budget, _info = last_eval

    print("Last evaluated configuration:")
    print(dehb.vector_to_configspace(config), end="")
    print(f"got a score of {score}, was evaluated at a budget of {budget:.2f} and took {cost:.3f} seconds to run.")
    print(f"The additional info attached: {_info}")

