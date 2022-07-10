"""
This package is stores all functions dealing with configuration files or related.
"""
import yaml


def load_yaml_config(path_to_config: str):
    """
    Read and return the contents of a yaml file as a python dictionary.

    :param path_to_config: the path to the configuration file
    :return: dict representation of the yaml file
    """
    with open(path_to_config, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
