"""Train / eval image transforms. Settings live under ``data.augment`` in the config.

Train: ``RandomResizedCrop`` + horizontal flip + small rotation + mild color jitter.
Eval:  ``Resize(256)`` + ``CenterCrop(224)``.

No vertical flips (birds are never upside down in CUB) and hue jitter stays
small because plumage color is what separates many species.
"""

from __future__ import annotations

from typing import Callable

import torch
from torchvision import transforms as T


def get_norm_stats(cfg: dict) -> tuple[list[float], list[float]]:
    """``(mean, std)`` picked by ``data.normalization`` (``imagenet`` or ``cub``)."""
    data = cfg["data"]
    name = data.get("normalization", "imagenet")
    if name not in ("imagenet", "cub"):
        raise ValueError(f"data.normalization must be 'imagenet' or 'cub', got {name!r}")
    return data[f"{name}_mean"], data[f"{name}_std"]


def get_train_transforms(cfg: dict) -> Callable:
    size = cfg["data"]["image_size"]
    aug = cfg["data"]["augment"]
    mean, std = get_norm_stats(cfg)
    return T.Compose([
        T.RandomResizedCrop(size, scale=tuple(aug["crop_scale"])),
        T.RandomHorizontalFlip(aug["hflip_p"]),
        T.RandomRotation(aug["rotation_deg"]),
        T.ColorJitter(**aug["color_jitter"]),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def get_eval_transforms(cfg: dict) -> Callable:
    size = cfg["data"]["image_size"]
    mean, std = get_norm_stats(cfg)
    return T.Compose([
        T.Resize(cfg["data"]["augment"]["eval_resize"]),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def denormalize(images: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    """Undo ``Normalize`` for plotting; works on ``[3, H, W]`` or ``[B, 3, H, W]``."""
    shape = (3, 1, 1)
    mean_t = torch.tensor(mean, dtype=images.dtype, device=images.device).view(shape)
    std_t = torch.tensor(std, dtype=images.dtype, device=images.device).view(shape)
    return (images * std_t + mean_t).clamp(0, 1)
