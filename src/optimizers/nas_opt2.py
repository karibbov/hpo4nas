from naslib.optimizers import Bananas, RegularizedEvolution
from naslib.search_spaces import NasBench201SearchSpace as NB201
from naslib.defaults.trainer import Trainer
from src.utils.extended_Trainer import ExtendedTrainer
import logging

from naslib.utils import utils, setup_logger, get_dataset_api
from naslib.utils.utils import parse_args
from naslib.search_spaces.core.query_metrics import Metric


def train_statistics_extended(self, report_incumbent=True):
    if report_incumbent:
        best_arch = self.get_final_architecture()
    else:
        try:
            best_arch = self.sampled_archs[-1].arch
        except AttributeError:
            try:
                best_arch = self.population[-1].arch
            except AttributeError:
                best_arch = self.train_data[-1].arch

    return (
        best_arch.query(
            Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api
        ),
        best_arch.query(
            Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api
        ),
        best_arch.query(
            Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api
        ),
        best_arch.query(
            Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api
        ),
        best_arch.query(
            Metric.TRAIN_LOSS, self.dataset, dataset_api=self.dataset_api
        ),
        best_arch.query(
            Metric.VAL_LOSS, self.dataset, dataset_api=self.dataset_api
        ),
        best_arch.query(
            Metric.TEST_LOSS, self.dataset, dataset_api=self.dataset_api
        ),
    )


def run_optimizer(config_file="../configs/config_bananas_none_0.yaml",
                  nas_optimizer=Bananas) -> None:
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
    nas_optimizer.train_statistics = train_statistics_extended
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
    config_path = "/home/samir/Desktop/F/Uni-Freiburg/DL lab/hpo4nas/configs/config_re_none_0.yaml"
    run_optimizer(config_path, RegularizedEvolution)
    config_path = "/home/samir/Desktop/F/Uni-Freiburg/DL lab/hpo4nas/configs/config_re_none_1.yaml"
    run_optimizer(config_path, RegularizedEvolution)
    config_path = "/home/samir/Desktop/F/Uni-Freiburg/DL lab/hpo4nas/configs/config_re_none_2.yaml"
    run_optimizer(config_path, RegularizedEvolution)
    config_path = "/home/samir/Desktop/F/Uni-Freiburg/DL lab/hpo4nas/configs/config_bananas_bayes_lin_reg_0.yaml"
    run_optimizer(config_path, Bananas)
    config_path = "/home/samir/Desktop/F/Uni-Freiburg/DL lab/hpo4nas/configs/config_bananas_gp_0.yaml"
    run_optimizer(config_path, Bananas)
    config_path = "/home/samir/Desktop/F/Uni-Freiburg/DL lab/hpo4nas/configs/config_bananas_rf_0.yaml"
    run_optimizer(config_path, Bananas)