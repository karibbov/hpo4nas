"""
This package defines an arbitrary NASLib optimizer for the selected search space and uses DeepCAVE to analyze the
results of running the optimizer.
"""

from NASLib import naslib
from naslib.optimizers.discrete.bananas.optimizer import Bananas
from naslib.search_spaces import NasBench201SearchSpace as NB201
from naslib.utils import utils, setup_logger, get_dataset_api
# import some utilities and parse the configuration file
import logging

# TODO clean things up
# To avoid automatically running the code I put it in a function

def run_naslib_opt(search_space, dataset, seed, output_path):
    """
    Runs an arbitrary NASLib optimizer.

    :param search_space: the search space for the optimization
    :param dataset: the dataset to train the picked architectures on
    :param seed: the seed to use for the optimizer
    :param output_path: the directory to output the resulting files of the optimization
    """
    print("Running NASLib optimizer...")

if __name__ == "__main__":
    # instantiate the search space object
    search_space = NB201()

    # This will read the parameters from the default yaml configuration file, which in this
    # case is located in NASLib/naslib/benchmarks/nas_predictors/discrete_config.yaml.
    # You do not have to change this but you can play around with its parameters.
    # config = utils.get_config_from_args(config_type="nas_predictor")
    config = utils.get_config_from_args(config_type="bbo-bs")
    utils.set_seed(config.seed)
    utils.log_args(config)

    logger = setup_logger(config.save + "/log.log")
    # logger.setLevel(logging.INFO)

    from naslib.optimizers import RegularizedEvolution as RE
    from naslib.optimizers import RandomSearch as RS
    from naslib.optimizers import Bananas

    # instantiate the optimizer object using the configuration file parameters
    optimizer = Bananas(config)

    from naslib.defaults.trainer import Trainer
    from src.trainers.extended_Trainer import ExtendedTrainer

    # since the optimizer has parsed the information of the search space, we do not need to pass the search
    # space object to the trainer when instantiating it.
    trainer = ExtendedTrainer(optimizer, config, lightweight_output=True)

    # this will load the NAS-Bench-201 data (architectures and their accuracy, runtime, etc).
    dataset_api = get_dataset_api(config.search_space, config.dataset)

    # adapt the search space to the optimizer type
    optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

    trainer.search(report_incumbent=False)

