# coding=utf-8
# HPO for NAS project

from src.optimizers.dehb_optimizer import run_dehb_on_nasbench201
from src.utils.nasbench201_configspace import run_rs_on_nasbench201


def run(optimizer):
    """
    Runs an optimizer on NAS-Bench-201

    :param optimizer: the name of the optimizer as a string
    """
    if optimizer == 'rs':
        run_rs_on_nasbench201()
    elif optimizer == 'dehb':
        run_dehb_on_nasbench201()
    elif optimizer == 're':
        # TODO implement RE under src/optimizers
        raise NotImplementedError('regularized evolution has not yet been implemented')
    elif optimizer == 'smac':
        # TODO implement SMAC under src/optimizers
        raise NotImplementedError('SMAC has not yet been implemented')
    else:
        raise NameError('invalid optimizer name "{}"'.format(optimizer))


if __name__ == '__main__':
    run('rs')
