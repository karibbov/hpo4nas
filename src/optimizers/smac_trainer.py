import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nas201/'))

import json
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

    def query(self, arch, budget):
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
        val_acc = graph.query(Metric.VAL_ACCURACY, epoch=round(budget), dataset='cifar10', dataset_api=api)
        print("Validation Accuracy: %.4f" % val_acc)

        train_acc = graph.query(Metric.TRAIN_ACCURACY, epoch=round(budget), dataset='cifar10', dataset_api=api)
        test_acc = graph.query(Metric.TEST_ACCURACY, epoch=round(budget), dataset='cifar10', dataset_api=api)

        train_loss = graph.query(Metric.TRAIN_LOSS, epoch=round(budget), dataset='cifar10', dataset_api=api)
        val_loss = graph.query(Metric.VAL_LOSS, epoch=round(budget), dataset='cifar10', dataset_api=api)
        test_loss = graph.query(Metric.TEST_LOSS, epoch=round(budget), dataset='cifar10', dataset_api=api)

        train_time = graph.query(Metric.TRAIN_TIME, epoch=round(budget), dataset='cifar10', dataset_api=api)

        dictionary = {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "train_time": train_time,
            "budget": round(budget)
        }

        # Read JSON file
        with open("run_history.json") as fp:
            listObj = json.load(fp)

        listObj.append(dictionary)

        with open('run_history.json', 'w') as f:
            json.dump(listObj, f, indent=4, separators=(',', ': '))

        return 100 - val_acc
