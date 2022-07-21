from naslib.optimizers import Bananas
from naslib.search_spaces import NasBench201SearchSpace as NB201
from naslib.defaults.trainer import Trainer
from src.utils.extended_Trainer import ExtendedTrainer
import logging

from naslib.utils import utils, setup_logger, get_dataset_api
from naslib.utils.utils import parse_args


def run_optimizer(config_file="../configs/config_bananas_none_0.yaml",
                  nas_optimizer=Bananas) -> None:
    # TODO: add all the utilities, such as config file reading, logging as before.
    # afterwards instantiate the search space, optimizer, trainer and run the search + evaluation

    # This will read the parameters from the default yaml configuration file, which in this
    # case is located in NASLib/naslib/benchmarks/nas_predictors/discrete_config.yaml.
    # You do not have to change this but you can play around with its parameters.
    args = parse_args(args=["--config-file", config_file])
    config = utils.get_config_from_args(args=args, config_type="nas_predictor")
    utils.set_seed(config.seed)
    utils.log_args(config)

    logger = setup_logger(config.save + "/log.log")
    logger.setLevel(logging.INFO)

    # instantiate the search space object
    search_space = NB201()
    optimizer = nas_optimizer(config)

    # this will load the NAS-Bench-201 data (architectures and their accuracy, runtime, etc).
    dataset_api = get_dataset_api(config.search_space, config.dataset)

    # adapt the search space to the optimizer type
    optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

    # since the optimizer has parsed the information of the search space, we do not need to pass the search
    # space object to the trainer when instantiating it.
    trainer = ExtendedTrainer(optimizer, config, lightweight_output=True)
    trainer.search(report_incumbent=False)
    trainer.evaluate(dataset_api=dataset_api)

    pass

if __name__ == "__main__":
    config_path = "/home/samir/Desktop/F/Uni-Freiburg/DL lab/hpo4nas/configs/config_bananas_rf_0.yaml"
    run_optimizer(config_path, Bananas)