"""Train / val / test DataLoaders over CUB."""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.data.cub_dataset import CUBDataset, load_hf_split
from src.data.splits import get_train_val_indices
from src.utils.config import seed_worker


def default_transform(cfg: dict) -> Callable:
    """Resize + center crop + normalize. Placeholder until ``src/data/transforms.py`` (Step 2)."""
    size = cfg["data"]["image_size"]
    return T.Compose([
        T.Resize(int(size * 256 / 224)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(cfg["data"]["mean"], cfg["data"]["std"]),
    ])


def build_datasets(
    cfg: dict,
    train_tf: Callable | None = None,
    eval_tf: Callable | None = None,
) -> tuple[CUBDataset, CUBDataset, CUBDataset]:
    data_cfg = cfg["data"]
    train_tf = train_tf or default_transform(cfg)
    eval_tf = eval_tf or default_transform(cfg)

    hf_train = load_hf_split("train", data_cfg["train_dataset"], data_cfg.get("cache_dir"))
    hf_test = load_hf_split("test", data_cfg["test_dataset"], data_cfg.get("cache_dir"))

    train_idx, val_idx = get_train_val_indices(cfg, hf_train["label"])
    train_ds = CUBDataset("train", train_tf, indices=train_idx, hf_dataset=hf_train)
    val_ds = CUBDataset("train", eval_tf, indices=val_idx, hf_dataset=hf_train)
    test_ds = CUBDataset("test", eval_tf, hf_dataset=hf_test)
    return train_ds, val_ds, test_ds


def build_dataloaders(
    cfg: dict,
    train_tf: Callable | None = None,
    eval_tf: Callable | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader, test_loader)``.

    Validation comes from the committed stratified split of the train set and
    uses ``eval_tf`` (no augmentation). Only train is shuffled.
    """
    data_cfg = cfg["data"]
    train_ds, val_ds, test_ds = build_datasets(cfg, train_tf, eval_tf)

    num_workers = data_cfg.get("num_workers", 0)
    common = dict(
        batch_size=data_cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
    )
    generator = torch.Generator().manual_seed(cfg.get("seed", 0))

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, generator=generator, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader
