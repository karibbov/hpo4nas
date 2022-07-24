"""
This package defines basic ConfigSpace related utility functions.
"""
import time
from pathlib import Path
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from deepcave import Objective, Recorder
from naslib.search_spaces import NasBench201SearchSpace
from naslib.utils import get_dataset_api
from naslib.search_spaces.core.query_metrics import Metric
from typing import Iterable, List, Union
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


def configuration2op_indices(config):
    """
    Given a NAS-Bench-201 configuration return operation indices for search space

    :param config: a sample NAS-Bench-201 configuration sampled from the ConfigSpace
    :return: operation indices
    """
    OP_NAMES, nasbench201_params = _nasbench201_parameters()

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
    OP_NAMES, nasbench201_params = _nasbench201_parameters()

    if isinstance(op_indices, str):
        op_indices = list(op_indices)

    cs = configure_nasbench201()

    values = {nasbench201_params[idx]: OP_NAMES[int(value)] for idx, value in enumerate(op_indices)}
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


def _query(model: NasBench201SearchSpace, dataset: str, dataset_api, epoch: int):
    """
    Train the model on the dataset up to a certain epoch budget, and query its performance metrics on NAS-Bench-201
    benchmark.

    :param model: the model whose performance to query
    :param dataset: the dataset, a string representation thereof, e.g.: 'cifar10'
    :param dataset_api: the dataset_api for nasbench201
    :param epoch: the epoch to query
    :return: train_loss, valid_loss, test_loss, train_regret, valid_regret, test_regret, train_time
    """
    train_loss = model.query(Metric.TRAIN_LOSS, dataset=dataset, dataset_api=dataset_api, epoch=epoch)
    valid_loss = model.query(Metric.VAL_LOSS, dataset=dataset, dataset_api=dataset_api, epoch=epoch)
    if epoch == 199:
        test_loss = model.query(Metric.TEST_LOSS, dataset=dataset, dataset_api=dataset_api, epoch=-1)
    else:
        # The true test loss/regret is only defined for the architectures that have been fully trained for 200 epochs
        test_loss = valid_loss

    train_acc = model.query(Metric.TRAIN_ACCURACY, dataset=dataset, dataset_api=dataset_api, epoch=epoch)
    train_regret = 100 - train_acc
    valid_acc = model.query(Metric.VAL_ACCURACY, dataset=dataset, dataset_api=dataset_api, epoch=epoch)
    valid_regret = optimal_nasbench201_performance()[dataset + "_val_acc"] - valid_acc
    if epoch == 199:
        test_acc = model.query(Metric.TEST_ACCURACY, dataset=dataset, dataset_api=dataset_api, epoch=-1)
        test_regret = optimal_nasbench201_performance()[dataset + "_test_acc"] - test_acc
    else:
        test_regret = valid_regret

    train_time = model.query(Metric.TRAIN_TIME, dataset=dataset, dataset_api=dataset_api, epoch=epoch)
    # simulate training time for the specific epoch
    train_time *= (epoch+1)/200

    return train_loss, valid_loss, test_loss, train_regret, valid_regret, test_regret, train_time


def query_nasbench201(cs_config: Configuration, dataset: str, epoch: int):
    """
    Based on the given configuration, create a model, train it on the dataset up to a certain epoch budget, and query
    its performance metrics on NAS-Bench-201 benchmark.

    :param epoch: the epoch to query the architecture's performance at
    :param cs_config: The architecture to query for
    :param dataset: Query the architectures performance on this dataset
    :return: train_loss, valid_loss, test_loss, train_regret, valid_regret, test_regret, train_time
    """
    op_indices = configuration2op_indices(cs_config)
    # model represents the sampled architecture
    model = NasBench201SearchSpace()
    model.set_op_indices(op_indices)
    dataset_api = get_dataset_api(search_space='nasbench201', dataset=dataset)

    return _query(model, dataset, dataset_api, epoch)


def run_rs(config: dict, output_path: str):
    """
    Run random search on the search space where a specified number of models are trained on different budgets and
    output the results in the DeepCAVE format.

    :param config: the configuration storing the run's parameters
    :param output_path: the path to the output produced by deepcave
    """
    print(f"Configurations for random search: {config['rs']}")
    print("Random search is running, 1 dot = 1 sampled architecture")
    configspace = configure_nasbench201()

    optimal_train_acc = 100
    optimal_val_acc = optimal_nasbench201_performance()["cifar10_val_acc"]
    optimal_test_acc = optimal_nasbench201_performance()["cifar10_test_acc"]

    obj1 = Objective("Train loss", lower=0, upper=100)
    obj2 = Objective("Validation loss", lower=0, upper=100)
    obj3 = Objective("Test loss", lower=0, upper=100)
    obj4 = Objective("Train regret", lower=0, upper=100)
    obj5 = Objective("Validation regret", lower=0, upper=100)
    obj6 = Objective("Test regret", lower=0, upper=100)
    obj7 = Objective("Train time", lower=0)
    objectives = [obj1, obj2, obj3, obj4, obj5, obj6, obj7]

    budgets = config['rs']['budgets']
    runtime_limit = config['rs']['runtime_limit']
    wc_start_time = time.time()
    wc_current_time = time.time()
    start_time = 0

    with Recorder(configspace, objectives=objectives, save_path=output_path) as r:
        sampled_configs = configspace.sample_configuration(config['rs']['n_models_per_budget'])
        # in case we sampled a single configuration only
        if isinstance(sampled_configs, Configuration):
            # list() will automatically convert the configspace to op indices for some reason
            temp = []
            temp.append(sampled_configs)
            sampled_configs = temp
        idx = 0
        while wc_current_time - wc_start_time < runtime_limit and idx < len(sampled_configs):
            for budget in budgets:
                r.start(sampled_configs[idx], budget, start_time=start_time)
                train_loss, val_loss, test_loss, train_regret, val_regret, test_regret, train_time = query_nasbench201(
                    sampled_configs[idx], config['dataset'], budget)
                # Simulate train time
                end_time = start_time + train_time
                r.end(costs=[train_loss, val_loss, test_loss, train_regret, val_regret, test_regret, train_time],
                      end_time=end_time)
                wc_current_time = time.time()
                start_time = end_time
                print('.', end='')
            idx = idx + 1
    print(f"Completed random search run")
    print(f"Total runtime: {wc_current_time - wc_start_time}")
    print(f"Number of completed function evaluations: {(idx+1) * len(budgets)}")
