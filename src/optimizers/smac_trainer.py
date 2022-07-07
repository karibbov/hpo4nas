import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nas201/'))

from deepcave import Recorder, Objective
from naslib.utils import get_dataset_api, utils
from nasbench201_configspace import configuration2op_indices

from naslib.search_spaces.core.query_metrics import Metric

from naslib.search_spaces import (NasBench201SearchSpace)

search_spaces = {
    'nasbench201': NasBench201SearchSpace,
}


def build_eval_dataloaders(config):
    train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
        config, mode="val"
    )
    return train_queue, valid_queue, test_queue


class SmacTrainer(object):
    """
    Default implementation that handles dataloading and preparing batches, the
    train loop, gathering statistics, checkpointing and doing the final evaluation.

    """

    def __init__(self, lightweight_output=False):
        """
        Initializes the trainer.

        Args:
            optimizer: A NASLib optimizer
            config (AttrDict): The configuration loaded from a yaml file, e.g
                via  `utils.get_config_from_args()`
        """
        self.lightweight_output = lightweight_output

    def query(self, arch):
        """
        Evaluate the architecture as given from the optimizer.

        If the search space has an interface to a benchmark then query that.
        Otherwise train as defined in the config.

        """
        op_indicies = configuration2op_indices(arch)
        # Loading NAS-201
        api = get_dataset_api(search_space='nasbench201', dataset='cifar10')
        graph = search_spaces['nasbench201']()
        graph.set_op_indices(op_indicies)
        acc = 1 - graph.query(Metric.VAL_ACCURACY, dataset='cifar10', dataset_api=api)
        print("Validation Accuracy: %.4f" % acc)

        with Recorder(configspace, objectives=[accuracy, mse]) as r:
            for config in configspace.sample_configuration(100):
                for budget in [20, 40, 60]:
                    r.start(config, budget)
                    # Your code goes here
                    r.end(costs=[0.5, 0.5])

        return acc