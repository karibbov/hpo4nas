import numpy as np

# make sure that this import line comes before all naslib imports
from nasbench201_configspace import configure_nasbench201, configuration2op_indices
from naslib.search_spaces import NasBench201SearchSpace
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils import get_dataset_api

from dehb import DEHB


def create_configspace_for_nasbench201():
    # define the config space that dehb will search over
    return configure_nasbench201()


def target_function(config, budget):
    # convert from ConfigSpace object to another one that is recognized by NasBench201
    op_indices = configuration2op_indices(config)
    # Building model
    model = NasBench201SearchSpace()
    model.set_op_indices(op_indices)
    dataset_api = get_dataset_api(search_space='nasbench201', dataset=dataset)
    # query the benchmark with the given dataset at a certain epoch, this will return all kinds of metric info
    train_cost = model.query(Metric.TRAIN_ACCURACY, dataset=dataset, dataset_api=dataset_api)
    valid_accuracy = model.query(Metric.VAL_ACCURACY, dataset=dataset, dataset_api=dataset_api)
    cost = model.query(Metric.TRAIN_TIME, dataset=dataset, dataset_api=dataset_api)
    cs.sample_configuration()
    valid_accuracy = np.random.uniform()
    result = {
        "fitness": -valid_accuracy,  # DE/DEHB minimizes
        "cost": 1,
        "info": {
            "train cost": train_cost,
            "budget": budget
        }
    }
    return result


# specify what is the minimum and the maximum epoch dehb should use as the lowest and highest fidelity
min_budget = 6
max_budget = 200
# prepare the dataset
dataset = 'cifar10'
cs = create_configspace_for_nasbench201()
# get the dimension of the config space for dehb
dimensions = len(cs.get_hyperparameters())

dehb = DEHB(
    f=target_function,
    cs=cs,
    dimensions=dimensions,
    min_budget=min_budget,
    max_budget=max_budget,
    n_workers=1,
    output_path="../../results/dehb_runs/cifar10"
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
