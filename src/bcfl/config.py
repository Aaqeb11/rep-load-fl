"""Load an experiment config (YAML) into a plain Python dictionary."""

import copy

import yaml


def load_config(path):
    """Read a YAML file and return it as a dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(config, overrides):
    """Change config values from the command line.

    Each override looks like "data.alpha=0.1" or "seed=2".
    The part before "=" is the path through the config, separated by dots.
    """
    config = copy.deepcopy(config)

    for override in overrides:
        key_path, raw_value = override.split("=", 1)
        keys = key_path.split(".")

        # Walk down to the dictionary that holds the last key.
        section = config
        for key in keys[:-1]:
            section = section[key]

        # yaml.safe_load turns "0.1" into 0.1, "true" into True, and so on.
        section[keys[-1]] = yaml.safe_load(raw_value)

    return config
