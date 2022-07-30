"""
This package defines basic ConfigSpace related utility functions.
"""
import time
from pathlib import Path

import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from naslib.search_spaces import NasBench201SearchSpace
from naslib.search_spaces.core.graph import Graph
from naslib.utils import get_dataset_api
from naslib.search_spaces.core.query_metrics import Metric
from typing import List, Union
from ConfigSpace.read_and_write import json as cs_json


def _nasbench201_parameters():
    """
    Get nasbench201 specific parameters. Returns the parameters needed to convert between ConfigSpace and nasbench201.

    :return: OP_NAMES: list, nasbench201_params: list
    """
    OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool3x3"]
    nasbench201_params = ['op_0', 'op_1', 'op_2', 'op_3', 'op_4', 'op_5']
    return OP_NAMES, nasbench201_params


def optimal_nasbench201_performance():
    """
    Returns the best possible performance that can be reached with the trained architectures on nasbench201.
    This includes validation, and test accuracy for all datasets that are defined for this benchmark. Namely, cifar10,
    cifar100, and Imagenet.

    Example:
        To get the best possible validation accuracy on cifar10 with nasbench201 use the key 'cifar10_val_acc' with the
        dictionary returned by this function

    :return: dictionary of optimal results
    """
    # The following optimal results are the mean optimal results, so it is not a good upper limit for accuracy
    # nasbenc201_optimal_results = {
    #     "cifar10_val_acc": 91.61, "cifar10_test_acc": 94.37,
    #     "cifar100_val_acc": 73.49, "cifar100_test_acc": 73.51,
    #     "imgnet_val_acc": 46.77, "imgnet_test_acc": 47.31,
    # }
    nasbenc201_optimal_results = {
        "cifar10_val_acc": 100, "cifar10_test_acc": 100,
        "cifar100_val_acc": 100, "cifar100_test_acc": 100,
        "imgnet_val_acc": 100, "imgnet_test_acc": 100,
    }
    return nasbenc201_optimal_results


def save_configspace(output_path: Path, file_name: str, formatting="json"):
    """
    Saves configspace into a file under the directory specified by output path.

    :param:file_name: name of the file
    :param:formatting: the format to save configspace in
    :param:output_path: the path to the output
    """
    cs = configure_nasbench201()
    if formatting == "json":
        Path(output_path, f"{file_name}.json").write_text(cs_json.write(cs))
    else:
        raise NotImplementedError(f"'{formatting}' is not supported. Try json.")


def configure_nasbench201():
    """
    Creates the ConfigSpace for NAS-Bench-201

    :return: ConfigSpace object for the NAS-Bench-201 search space
    """
    OP_NAMES, nasbench201_params = _nasbench201_parameters()
    cs = CS.ConfigurationSpace()
    op_0 = CSH.CategoricalHyperparameter(nasbench201_params[0], choices=OP_NAMES)
    op_1 = CSH.CategoricalHyperparameter(nasbench201_params[1], choices=OP_NAMES)
    op_2 = CSH.CategoricalHyperparameter(nasbench201_params[2], choices=OP_NAMES)
    op_3 = CSH.CategoricalHyperparameter(nasbench201_params[3], choices=OP_NAMES)
    op_4 = CSH.CategoricalHyperparameter(nasbench201_params[4], choices=OP_NAMES)
    op_5 = CSH.CategoricalHyperparameter(nasbench201_params[5], choices=OP_NAMES)

    cs.add_hyperparameters([op_0, op_1, op_2, op_3, op_4, op_5])
    return cs


def configuration2op_indices(config: Configuration):
    """
    Given a NAS-Bench-201 configuration return operation indices for search space

    :param:config: a sample NAS-Bench-201 configuration sampled from the ConfigSpace
    :return: operation indices
    """
    OP_NAMES, nasbench201_params = _nasbench201_parameters()

    op_indices = np.ones(len(nasbench201_params)) * -1
    for idx, param in enumerate(nasbench201_params):
        op_indices[idx] = OP_NAMES.index(config[param])
    return op_indices.astype(int)


def sample_random_architecture(search_space: Graph, cs: ConfigurationSpace):
    """
    Given a ConfigSpace, it samples a random architecture from the search space

    :param:search_space: Graph object (e.g. NasBench201SearchSpace)
    :param:cs: ConfigurationSpace object for the given search space
    :return: Queryable SearchSpace object
    """
    config = cs.sample_configuration()
    op_indices = configuration2op_indices(config)
    search_space.set_op_indices(op_indices)

    return search_space


def op_indices2config(op_indices: Union[List[Union[int, str]], str]) -> Configuration:
    """
    Returns a configuration for nasbech201 configuration space, given operation indices

    :param:op_indices: Iterable of operation indices
    :return: The configuration object corresponding to the op_indices
    """
    OP_NAMES, nasbench201_params = _nasbench201_parameters()

    if isinstance(op_indices, str):
        op_indices = list(op_indices)

    cs = configure_nasbench201()

    values = {nasbench201_params[idx]: OP_NAMES[int(value)] for idx, value in enumerate(op_indices)}
    config = Configuration(configuration_space=cs, values=values)
    config.is_valid_configuration()

    return config


def cherry_pick(op_names: List[str]):
    """
    Pick a specific architecture from the configuration space of nasbench201. Given a list of operation names, it sets
    each edge of the architecture to the respective operation name.

    :param:op_names: A list of 6 values from: ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool3x3"]
    :return: The Configuration object with the given operations
    """
    cs = configure_nasbench201()
    _, ops = _nasbench201_parameters()
    arch = cs.sample_configuration()
    for op, op_value in zip(ops, op_names):
        arch[op] = op_value
    return arch


def get_arch_performance(op_names: List[str], run_config: dict, epoch=199):
    """
    Cherry-pick an architecture and query its performance on the benchmark

    :param:epoch: The amount of epochs to train the architecture for on the dataset
    :param:run_config: the configuration as dictionary containing the dataset to evaluate the architecture on
    :param:op_names: A list of 6 values from: ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool3x3"]
    :return: Performance of the architecture defined by the op_names on the dataset, trained up to the given epochs
    """
    arch = cherry_pick(op_names)
    return query_nasbench201(arch, run_config['dataset'], epoch)


def nasbench201_random_query(search_space: NasBench201SearchSpace, cs: ConfigurationSpace, dataset: str):
    """
    Samples a random configuration from NAS-Bench-201 and queries the evaluation results from the benchmark

    :param:search_space: NasBench201SearchSpace object
    :param:cs: ConfigSpace object corresponding to the search space
    :param:dataset: dataset the sample is evaluated on
    :return: a tuple containing the validation accuracy and the training time
    """
    sample_random_architecture(search_space, cs)
    dataset_api = get_dataset_api(search_space='nasbench201', dataset=dataset)
    accuracy = search_space.query(Metric.VAL_ACCURACY, dataset=dataset, dataset_api=dataset_api)
    cost = search_space.query(Metric.TRAIN_TIME, dataset=dataset, dataset_api=dataset_api)

    return accuracy, cost


def query(model: NasBench201SearchSpace, dataset: str, dataset_api, epoch: int):
    """
    Train the model on the dataset up to a certain epoch budget, and query its performance metrics on NAS-Bench-201
    benchmark.

    :param:model: the model whose performance to query
    :param:dataset: the dataset, a string representation thereof, choices: 'cifar10' 'cifar100', 'ImageNet16-120'
    :param:dataset_api: the dataset_api for nasbench201, as defined in NASlib
    :param:epoch: the epoch to query
    :return: train_loss, valid_loss, test_loss, train_regret, valid_regret, test_regret, train_time
    """

    test_epoch = -1 if epoch == 199 else epoch

    train_loss = model.query(Metric.TRAIN_LOSS, dataset=dataset, dataset_api=dataset_api, epoch=epoch)
    valid_loss = model.query(Metric.VAL_LOSS, dataset=dataset, dataset_api=dataset_api, epoch=epoch)
    train_acc = model.query(Metric.TRAIN_ACCURACY, dataset=dataset, dataset_api=dataset_api, epoch=epoch)
    valid_acc = model.query(Metric.VAL_ACCURACY, dataset=dataset, dataset_api=dataset_api, epoch=epoch)
    train_time = model.query(Metric.TRAIN_TIME, dataset=dataset, dataset_api=dataset_api, epoch=epoch)
    test_loss = model.query(Metric.TEST_LOSS, dataset=dataset, dataset_api=dataset_api, epoch=test_epoch)
    test_acc = model.query(Metric.TEST_ACCURACY, dataset=dataset, dataset_api=dataset_api, epoch=test_epoch)

    train_regret = 100 - train_acc
    valid_regret = 100 - valid_acc
    test_regret = 100 - test_acc
    # simulate training time for the specific epoch
    train_time *= (epoch+1)/200

    return train_loss, valid_loss, test_loss, train_regret, valid_regret, test_regret, train_time


def query_nasbench201(config: Configuration, dataset: str, epoch: int):
    """
    Based on the given configuration, create a model, train it on the dataset up to a certain epoch budget, and query
    its performance metrics on NAS-Bench-201 benchmark.

    :param:epoch: the epoch to query the architecture's performance at
    :param:config: The architecture to query for
    :param:dataset: Query the architectures performance on this dataset
    :return: train_loss, valid_loss, test_loss, train_regret, valid_regret, test_regret, train_time
    """
    op_indices = configuration2op_indices(config)
    # model represents the sampled architecture
    model = NasBench201SearchSpace()
    model.set_op_indices(op_indices)
    dataset_api = get_dataset_api(search_space='nasbench201', dataset=dataset)

    return query(model, dataset, dataset_api, epoch)


if __name__ == '__main__':
    # A simple test case
    config_space = configure_nasbench201()
    sample = config_space.sample_configuration()
    results = query_nasbench201(sample, 'cifar10', 199)
    print(results)
