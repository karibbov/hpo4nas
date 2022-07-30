import time
from pathlib import Path
from deepcave import Objective, Recorder
from ConfigSpace.configuration_space import Configuration
from src.utils.nasbench201_configspace import configure_nasbench201


def run_rs(run_config: dict, output_path: Path, dataset: str):
    """
    Run random search on the search space where a specified number of models are trained on different budgets and
    output the results in the DeepCAVE format.

    :param: dataset: the dataset as a string
    :param: run_config: the configuration storing the run's parameters
    :param: output_path: the path to the output produced by deepcave
    """
    print(f'Configurations for random search: {run_config["rs"]}')
    print('Random search is running, 1 dot = 1 sampled architecture')
    cs = configure_nasbench201()

    obj1 = Objective('Train loss', lower=0)
    obj2 = Objective('Validation loss', lower=0)
    obj3 = Objective('Test loss', lower=0)
    obj4 = Objective('Train regret', lower=0, upper=100)
    obj5 = Objective('Validation regret', lower=0, upper=100)
    obj6 = Objective('Test regret', lower=0, upper=100)
    obj7 = Objective('Train time', lower=0)
    objectives = [obj1, obj2, obj3, obj4, obj5, obj6, obj7]

    budgets = run_config['rs']['budgets']
    runtime_limit = run_config['rs']['runtime_limit']
    wc_start_time = time.time()
    wc_current_time = time.time()
    start_time = 0

    with Recorder(cs, objectives=objectives, save_path=str(output_path)) as r:
        sampled_configs = cs.sample_configuration(run_config['rs']['n_models_per_budget'])
        # in case we sampled a single configuration only
        if isinstance(sampled_configs, Configuration):
            # list() will automatically convert the cs to op indices for some reason
            temp = []
            temp.append(sampled_configs)
            sampled_configs = temp
        idx = 0
        while wc_current_time - wc_start_time < runtime_limit and idx < len(sampled_configs):
            for budget in budgets:
                r.start(sampled_configs[idx], budget, start_time=start_time)
                train_loss, val_loss, test_loss, train_regret, val_regret, test_regret, train_time = query_nasbench201(
                    sampled_configs[idx], dataset, budget)
                # Simulate train time
                end_time = start_time + train_time
                r.end(costs=[train_loss,
                             val_loss,
                             test_loss,
                             train_regret,
                             val_regret,
                             test_regret,
                             train_time],
                      end_time=end_time)
                wc_current_time = time.time()
                start_time = end_time
                print('.', end='')
            idx = idx + 1
    print(f'Completed random search run')
    print(f'Total runtime: {wc_current_time - wc_start_time}')
    print(f'Number of completed function evaluations: {(idx+1) * len(budgets)}')
    print(f'Output of the run saved under:\n{output_path}')