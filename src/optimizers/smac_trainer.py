import os
import sys

from src.utils.nasbench201_configspace import query_nasbench201

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
        train_loss, val_loss, test_loss, train_regret, val_regret, test_regret, train_time = query_nasbench201(
            arch, 'cifar10', round(budget))

        print("Validation Regret: %.4f" % val_regret)

        dictionary = {
            "train_acc": train_regret,
            "val_acc": val_regret,
            "test_acc": test_regret,
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

        return val_regret
