import fnmatch
import os
from pathlib import Path

from deepcave.runs.converters.deepcave import DeepCAVERun
from src.utils.dehb_converter import DEHBRun
from src.utils.nasbench201_configspace import save_configspace


def generate_deepcave_output(config: dict, output_path: str):
    """
    Generate output from the results under output_path that will be converted into the DeepCAVE format and saved under
    output_path.

    :param: config: the configuration file as a dictionary containing the optimizer for which to generate the outputs
    :param: output_path: the path to the output folder
    """
    optimizer = config['optimizer']
    save_configspace(output_path=output_path, file_name="configspace")
    if optimizer == 'rs':
        raise NotImplementedError("Generation of DeepCAVE outputs for RS has not yet been implemented.")
        # run = DeepCAVERun.from_path(Path(output_path)) this does not work, because I use Recorder with RS
    elif optimizer == 'dehb':
        run = DEHBRun.from_path(Path(output_path))
    elif optimizer == 're':
        raise NotImplementedError("Generation of DeepCAVE outputs for RE has not yet been implemented.")
    elif optimizer == 'smac':
        raise NotImplementedError("Generation of DeepCAVE outputs for SMAC has not yet been implemented.")
    else:
        raise NameError('Invalid optimizer name "{}"'.format(optimizer))
    run.save(output_path)


def generate_only_outputs_for_deepcave(config: dict, output_path: str):
    """
    Generates DeepCAVE run files for all previous optimizer runs under output_path.

    :param: config: the configuration file as a dictionary containing the optimizer for which to generate the outputs
    :param: output_path: the path to the output folder where the runs are
    """
    runs = fnmatch.filter(os.listdir(output_path), '*run*')
    if len(runs) > 0:
        for run in runs:
            generate_deepcave_output(config, f'{output_path}/{run}')
    else:
        raise FileNotFoundError(f'No runs were found under {output_path}')

