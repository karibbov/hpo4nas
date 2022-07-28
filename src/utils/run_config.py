"""
This package is stores all functions dealing with configuration files or related.
"""
from pathlib import Path

import yaml


# TODO Using dict is not a scalable, think of another way, maybe load the config into a singleton or some object
def load_yaml_config(path_to_yaml: Path):
    """
    Read and return the contents of a yaml file as a python dictionary.

    :param path_to_yaml: the path to the file
    :return: dict representation of the yaml file
    """
    with open(path_to_yaml, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
