"""Config loading and reproducibility helpers shared by every step."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load a YAML config.

    A config may set ``base: other.yaml`` (relative to its own folder) to inherit
    from another config; keys in the child override the parent. This lets
    ``cls_scratch.yaml`` hold only what differs from ``base.yaml``.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    parent = cfg.pop("base", None)
    if parent is not None:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    return cfg


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (CPU and CUDA).

    ``deterministic=True`` also forces deterministic cuDNN kernels, which is
    slower; leave it off for normal training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` so each worker's random augmentations are reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(preferred: str | None = None) -> torch.device:
    """Return ``preferred`` if given, otherwise CUDA, then Apple MPS, then CPU."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
