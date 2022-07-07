import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import Configuration

from naslib.search_spaces import NasBench201SearchSpace
from naslib.utils import get_dataset_api
from naslib.search_spaces.core.query_metrics import Metric

from typing import Iterable, List, Union, Dict

# TODO: Convert this to a ConfigurationAdapter Class
OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool3x3"]
nasbench201_params = ['op_0', 'op_1', 'op_2', 'op_3', 'op_4', 'op_5']
nasbenc201_optimal_results = {
    "cifar10_val_acc": 91.61, "cifar10_test_acc": 94.37,
    "cifar100_val_acc": 73.49, "cifar100_test_acc": 73.51,
    "imgnet_val_acc": 46.77, "imgnet_test_acc": 47.31
}


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


def nasbench201_random_query(search_space, nasbench201_space, dataset):
    """
    Samples a random configuration from NAS-Bench-201 and queries the evaluation results from the benchmark

    :param search_space: NasBench201SearchSpace object
    :param nasbench201_space: ConfigSpace object corresponding to NAS-Bench-201
    :param dataset: dataset the sample is evaluated on
    :return: a tuple containing the validation accuracy and the training time
    """
    sample_random_architecture(search_space, nasbench201_space)
    dataset_api = get_dataset_api(search_space='nasbench201', dataset=dataset)
    accuracy = search_space.query(Metric.VAL_ACCURACY, dataset=dataset, dataset_api=dataset_api)
    cost = search_space.query(Metric.TRAIN_TIME, dataset=dataset, dataset_api=dataset_api)

    return accuracy, cost


def run_rs_on_nasbench201(dataset='cifar10'):
    """
    Randomly select an architecture from NASBench201 and query the accuracy & cost of it. Print the results to
    console.

    :param dataset: the name of the dataset as a string (cifar10 by default)
    """
    nasbench201_configspace = configure_nasbench201()
    naslib_space = NasBench201SearchSpace()
    accuracy, cost = nasbench201_random_query(naslib_space, nasbench201_configspace, dataset)
    print(f"Val accuracy: {accuracy}")
    print(f"Training cost: {cost}")
