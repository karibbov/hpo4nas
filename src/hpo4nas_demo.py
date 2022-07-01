import os

from fvcore.common.config import CfgNode

from naslib import utils
from naslib.utils import setup_logger, logging, get_dataset_api
from naslib.search_spaces import NasBench201SearchSpace
from naslib.optimizers import RandomSearch, DARTSOptimizer, GDASOptimizer, DrNASOptimizer, \
    RegularizedEvolution, LocalSearch, Bananas, BasePredictor
from naslib.defaults.trainer import Trainer

# define the search space, which will hold a collection of architectures
from naslib.utils.utils import get_project_root
search_space = NasBench201SearchSpace()

config = utils.get_config_from_args()

supported_optimizers = {
    'darts': DARTSOptimizer(config),
    'gdas': GDASOptimizer(config),
    'drnas': DrNASOptimizer(config),
    'rs': RandomSearch(config),
    're': RegularizedEvolution(config),
    'ls': LocalSearch(config),
    'bananas': Bananas(config),
    'bp': BasePredictor(config)
}


utils.set_seed(config.seed)
utils.log_args(config)

logger = setup_logger(config.save + "/log.log")

# make sure to change optimizer in config file as well (run/cifar10/*rs*/0/search/model_final.pth)
optimizer = supported_optimizers['rs']

# this will load the NAS-Bench-201 data (architectures and their accuracy, runtime, etc).
dataset_api = get_dataset_api(config.search_space, config.dataset)
# adapt the search space to the optimizer type
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

trainer = Trainer(optimizer, config, lightweight_output=True)

trainer.search()

trainer.evaluate(dataset_api=dataset_api)
