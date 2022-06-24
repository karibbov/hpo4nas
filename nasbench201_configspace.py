import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
# from NASLib import naslib
from naslib.search_spaces import NasBench201SearchSpace
from naslib.utils import get_dataset_api
from naslib.search_spaces.core.query_metrics import Metric


OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool3x3"]
nasbench201_params = ['op_0', 'op_1', 'op_2', 'op_3', 'op_4', 'op_5']


def configure_nasbench201():
    """
    Configures the configSpace for Nas-Bench-201
    :return: configSpace object for Nas-Bench-201 search space
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
    Given a NasBench201 configuration return operation indices for search space
    :param config: a sample NasBench201 configuration
    :return: operation indices
    """
    print(config)
    op_indices = np.ones(len(nasbench201_params)) * -1
    for idx, param in enumerate(nasbench201_params):
        op_indices[idx] = OP_NAMES.index(config[param])
    return op_indices.astype(int)


def sample_random_architecture(search_space, cs):
    """

    :param graph: SearchSpace object (e.g. NasBench201SearchSpace)
    :param cs: ConfigurationSpace object for the given search space
    :return: Queryable SearchSpace object
    """
    config = cs.sample_configuration()
    op_indices = configuration2op_indices(config)
    search_space.set_op_indices(op_indices)

    return search_space


# print(op_indices)

def nasbench201_random_query(search_space, nasbench201_space, dataset):
    """
    Samples a random configuration from Nas-Bench-201 and queries the evaluation results from the benchmark
    :param search_space: NasBench201SearchSpace object
    :param nasbench201_space: ConfigurationSpace object corresponding to nasbench201
    :param dataset: dataset the sample is evaluated on
    :return: validation_accuracy, training_time
    """

    sample_random_architecture(search_space, nasbench201_space)
    dataset_api = get_dataset_api(search_space='nasbench201', dataset=dataset)
    accuracy = search_space.query(Metric.VAL_ACCURACY, dataset=dataset, dataset_api=dataset_api)
    cost = search_space.query(Metric.TRAIN_TIME, dataset=dataset, dataset_api=dataset_api)

    return accuracy, cost


nasbench201_configspace = configure_nasbench201()
naslib_space = NasBench201SearchSpace()

accuracy, cost = nasbench201_random_query(naslib_space, nasbench201_configspace, 'cifar10')
print(f"Val accuracy: {accuracy}")
print(f"Training cost: {cost}")
