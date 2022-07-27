import json
# import smac_trainer
from functools import partial
import numpy as np

from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario
from src.utils.nasbench201_configspace import configure_nasbench201 as CS
from src.utils.nasbench201_configspace import query_nasbench201

# Select dataset: [ "cifar10", "cifar100". "ImageNet16-120"]
DATASET = "cifar10"

def query(arch, seed, budget):
    """
    Evaluate the architecture as given from the optimizer.

    If the search space has an interface to a benchmark then query that.
    Otherwise train as defined in the config.

    """
    # print(f"arch: {arch}, seed: {seed}, budget: {budget}")
    train_loss, val_loss, test_loss, train_regret, val_regret, test_regret, train_time = query_nasbench201(
        arch, DATASET, round(budget))

    print(f"Validation Regret: {val_regret}")
    dictionary = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "train_regret": train_regret,
        "val_regret": val_regret,
        "test_regret": test_regret,
        "train_time": train_time,
        "budget": round(budget)
    }

    print(dictionary)
    # Read JSON file
    with open("run_history.json") as fp:
        listObj = json.load(fp)

    listObj.append(dictionary)

    with open('run_history.json', 'w') as f:
        json.dump(listObj, f, indent=4, separators=(',', ': '))

    return val_regret


cs = CS()
# SMAC scenario object
scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative to runtime)
                     "wallclock-limit": 10,  # max duration to run the optimization (in seconds)
                     "cs": cs,  # configuration space
                     "deterministic": "True",
                     "limit_resources": True,  # Uses pynisher to limit memory and runtime
                     # Alternatively, you can also disable this.
                     # Then you should handle runtime and memory yourself in the TA
                     "memory_limit": 3072,  # adapt this to reasonable value for your hardware
                     })

# max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
max_iters = 199
# intensifier parameters (Budget parameters for BOHB)
intensifier_kwargs = {'initial_budget': 3, 'max_budget': max_iters, 'eta': 3}
# To optimize, we pass the function to the SMAC-object
# Select dataset: [ "cifar10", "cifar100". "ImageNet16-120"]
# nn = smac_trainer.SmacTrainer(dataset="cifar10")
smac = SMAC4MF(scenario=scenario, rng=np.random.RandomState(42),
               tae_runner=query,
               intensifier_kwargs=intensifier_kwargs,  # all arguments related to intensifier can be passed like this
               initial_design_kwargs={'n_configs_x_params': 1,  # how many initial configs to sample per parameter
                                      'max_config_fracs': .2})
# def_value = smac.get_tae_runner().run(config=cs.get_default_configuration(),
#                                       instance='1', budget=max_iters, seed=0)[1]
# Start optimization
try:  # try finally used to catch any interrupt
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

inc_value = smac.get_tae_runner().run(config=incumbent, instance='1',
                                      budget=max_iters, seed=0)[1]

print("Optimized Value: %.4f" % inc_value)

# score your optimal configuration to disk
opt_config = incumbent.get_dictionary()
with open('opt_cfg.json', 'w') as f:
    json.dump(opt_config, f)
