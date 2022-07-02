import ConfigSpace

from naslib.search_spaces import NasBench201SearchSpace
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils import get_dataset_api
from dehb import DEHB
from src.utils.nasbench201_configspace import configure_nasbench201, configuration2op_indices, \
    nasbenc201_optimal_results as optimal

DATASET = 'cifar10'

def _target_function(config: ConfigSpace, budget: int):
    """
    Interface for target function that DEHB optimizes. It is the problem that needs to be solved,
    or the function to be optimized

    :param config: the architecture to query defined as a ConfigSpace object
    :param budget: the current epoch to query
    :return: regret, training time and some additional information
    """
    # convert from ConfigSpace object to another one that is recognized by NasBench201
    op_indices = configuration2op_indices(config)
    # represents the sampled architecture
    model = NasBench201SearchSpace()
    model.set_op_indices(op_indices)
    dataset_api = get_dataset_api(search_space='nasbench201', dataset=DATASET)
    # query the benchmark with the given dataset at a certain epoch
    train_acc = model.query(Metric.TRAIN_ACCURACY, dataset=DATASET, dataset_api=dataset_api)
    valid_acc = model.query(Metric.VAL_ACCURACY, dataset=DATASET, dataset_api=dataset_api)
    test_acc = model.query(Metric.TEST_ACCURACY, dataset=DATASET, dataset_api=dataset_api)
    train_time = model.query(Metric.TRAIN_TIME, dataset=DATASET, dataset_api=dataset_api)
    regret = optimal["cifar10_val_acc"] - valid_acc
    result = {
        "fitness": regret,  # this is what DE/DEHB minimizes
        "cost": train_time,
        "info": {
            "train accuracy": train_acc,
            "test accuracy": test_acc,
            "budget": budget
        }
    }
    return result


def run_dehb_on_nasbench201(min_budget=6, max_budget=200, brackets=4):
    """
    Run DEHB on NAS-Bench-201 using the DATASET defined in this package

    :param dataset: a string specifying the dataset to train/validate/test the architectures on
    :param min_budget: minimum epoch dehb should use as the lowest fidelity
    :param max_budget: maximum epoch dehb should use as the highest fidelity
    :param brackets: the number of brackets to use with dehb
    """
    cs = configure_nasbench201()
    n_dimension = len(cs.get_hyperparameters())
    dehb = DEHB(
        f=_target_function,
        cs=cs,
        dimensions=n_dimension,
        min_budget=min_budget,
        max_budget=max_budget,
        n_workers=1,
        output_path="../../results/{}/dehbbbbbbbbbbbbbbbb".format(DATASET)
    )

    trajectory, runtime, history = dehb.run(
        brackets=4,
        verbose=True,
        save_intermediate=False,
    )

    print(len(trajectory), len(runtime), len(history), end="\n\n")

    # Last recorded function evaluation
    last_eval = history[-1]
    config, score, cost, budget, _info = last_eval

    print("Last evaluated configuration, ")
    print(dehb.vector_to_configspace(config), end="")
    print("got a score of {}, was evaluated at a budget of {:.2f} and "
          "took {:.3f} seconds to run.".format(score, budget, cost))
    print("The additional info attached: {}".format(_info))
