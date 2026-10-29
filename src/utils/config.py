"""Small helpers used everywhere: load a config, set the random seed, pick the device."""

import os
import random

import numpy as np
import torch
import yaml


def load_config(path):
    """Read a YAML config file into a dict.

    If the file has a line `base: base.yaml`, that file is loaded first and the
    values in this file replace the ones from base. This way cls_scratch.yaml
    only needs to list what is different from base.yaml.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "base" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg.pop("base"))
        cfg = merge_configs(load_config(base_path), cfg)
    return cfg


def merge_configs(base, override):
    """Copy of `base` with the values of `override` on top.

    Sections like `data:` are merged key by key, so overriding
    `data.normalization` keeps the rest of `data` from base.
    """
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def set_seed(seed):
    """Seed python, numpy and torch so runs can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the GPU


def get_device():
    """Use the GPU if there is one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
