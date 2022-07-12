"""
This package defines basic ConfigSpace related utility functions.
"""

from pathlib import Path
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import Configuration
from deepcave import Objective, Recorder
from naslib.search_spaces import NasBench201SearchSpace
from naslib.utils import get_dataset_api
from naslib.search_spaces.core.query_metrics import Metric
from typing import Iterable, List, Union
from ConfigSpace.read_and_write import json as cs_json


# TODO: Convert this to a ConfigurationAdapter Class
OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool3x3"]
nasbench201_params = ['op_0', 'op_1', 'op_2', 'op_3', 'op_4', 'op_5']
nasbenc201_optimal_results = {
    "cifar10_val_acc": 91.61, "cifar10_test_acc": 94.37,
    "cifar100_val_acc": 73.49, "cifar100_test_acc": 73.51,
    "imgnet_val_acc": 46.77, "imgnet_test_acc": 47.31
}


def save_configspace(output_path: str, file_name: str, formatting="json"):
    """
    Saves configspace into a file under the directory specified by output path.

    :param file_name: name of the file
    :param formatting: the format to save configspace in
    :param output_path: the path to the output
    """
    configspace = configure_nasbench201()
    if formatting == "json":
        Path(output_path, f"{file_name}.json").write_text(cs_json.write(configspace))
    else:
        raise NotImplementedError(f"'{formatting}' is not supported. Try json.")


def configure_nasbench201():
    """
    Creates the ConfigSpace for NAS-Bench-201

    :return: ConfigSpace object for the NAS-Bench-201 search space
    """
    cs = CS.ConfigurationSpace()
    op_0 = CSH.CategoricalHyperparameter(nasbench201_params[0], choices=OP_NAMES)
    op_1 = CSH.CategoricalHyperparameter(nasbench201_params[1], choices=OP_NAMES)
    op_2 = CSH.CategoricalHyperparameter(nasbench201_params[2], choices=OP_NAMES)
    op_3 = CSH.CategoricalHyperparameter(nasbench201_params[3], choices=OP_NAMES)
    op_4 = CSH.CategoricalHyperparameter(nasbench201_params[4], choices=OP_NAMES)
    op_5 = CSH.CategoricalHyperparameter(nasbench201_params[5], choices=OP_NAMES)

    cs.add_hyperparameters([op_0, op_1, op_2, op_3, op_4, op_5])
    return cs


def configuration2op_indices(config):
    """
    Given a NAS-Bench-201 configuration return operation indices for search space

    :param config: a sample NAS-Bench-201 configuration sampled from the ConfigSpace
    :return: operation indices
    """
    print(config)
    op_indices = np.ones(len(nasbench201_params)) * -1
    for idx, param in enumerate(nasbench201_params):
        op_indices[idx] = OP_NAMES.index(config[param])
    return op_indices.astype(int)


def sample_random_architecture(search_space, cs):
    """
    Given a ConfigSpace, it samples a random architecture from the search space

    :param search_space: Graph object (e.g. NasBench201SearchSpace)
    :param cs: ConfigurationSpace object for the given search space
    :return: Queryable SearchSpace object
    """
    config = cs.sample_configuration()
    op_indices = configuration2op_indices(config)
    search_space.set_op_indices(op_indices)

    return search_space


def op_indices2config(op_indices: Union[List[Union[int, str]], str]) -> Configuration:
    """
    Returns a configuration for nasbech201 configuration space, given operation indices

    :param op_indices: Iterable of operation indices
    :return: The configuration object corresponding to the op_indices
    """
    if isinstance(op_indices, str):
        op_indices = list(op_indices)

    cs = configure_nasbench201()

    values = {nasbench201_params[idx]: OP_NAMES[int(value)] for idx, value in enumerate(op_indices)}
    print(values)
    config = Configuration(configuration_space=cs, values=values)
    config.is_valid_configuration()

    return config


def nasbench201_random_query(search_space, configspace, dataset):
    """
    Samples a random configuration from NAS-Bench-201 and queries the evaluation results from the benchmark

    :param search_space: NasBench201SearchSpace object
    :param configspace: ConfigSpace object corresponding to the search space
    :param dataset: dataset the sample is evaluated on
    :return: a tuple containing the validation accuracy and the training time
    """
    sample_random_architecture(search_space, configspace)
    dataset_api = get_dataset_api(search_space='nasbench201', dataset=dataset)
    accuracy = search_space.query(Metric.VAL_ACCURACY, dataset=dataset, dataset_api=dataset_api)
    cost = search_space.query(Metric.TRAIN_TIME, dataset=dataset, dataset_api=dataset_api)

    return accuracy, cost


def nasbench201_query(search_space, cs_config: Configuration, dataset):
    op_indices = configuration2op_indices(cs_config)
    search_space.set_op_indices(op_indices)
    dataset_api = get_dataset_api(search_space='nasbench201', dataset=dataset)
    accuracy = search_space.query(Metric.VAL_ACCURACY, dataset=dataset, dataset_api=dataset_api)
    cost = search_space.query(Metric.TRAIN_TIME, dataset=dataset, dataset_api=dataset_api)

    return accuracy, cost


def run_rs(config: dict, output_path: str):
    """
    Run random search on the search space where a specified number of models are trained on different budgets and
    output the results in the DeepCAVE format.

    :param config: the configuration storing the run's parameters
    :param output_path: the path to the output produced by deepcave
    """
    configspace = configure_nasbench201()

    if output_path is None:
        raise ValueError("Output path has to be given when deepcave is enabled.")

    regret = Objective("regret", lower=0, upper=1, optimize="upper")
    train_time = Objective("training_time")

    with Recorder(configspace, objectives=[regret, train_time], save_path=output_path) as r:
        for cs_config in configspace.sample_configuration(config['rs']['n_models_per_budget']):
            for budget in config['rs']['budgets']:
                r.start(cs_config, budget)
                # The same nasbench201 object can't be queried more than once, so reinitialize it always
                # TODO Do you know why this is the case?
                nasbench201 = NasBench201SearchSpace()
                accuracy, train_time = nasbench201_query(nasbench201, cs_config, config['dataset'])
                regret = nasbenc201_optimal_results["cifar10_test_acc"] - accuracy
                r.end(costs=[regret, train_time])

