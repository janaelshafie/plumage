"""Fixed, stratified train/validation split of the CUB train set.

The validation indices are generated once and committed to
``configs/splits/val_indices.json`` so every teammate tunes on the same images.
The test set is never used for tuning.

Regenerate (only if the split must change, and tell the team):
    python -m src.data.splits --config configs/base.yaml --force
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


def stratified_val_indices(labels: Sequence[int], val_fraction: float, seed: int) -> list[int]:
    """Pick ``round(val_fraction * n_c)`` indices (at least 1) from every class ``c``."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[int(label)].append(idx)

    rng = np.random.default_rng(seed)
    val: list[int] = []
    for label in sorted(by_class):
        idxs = by_class[label]
        n_val = max(1, int(round(val_fraction * len(idxs))))
        if n_val >= len(idxs):
            raise ValueError(f"class {label} has only {len(idxs)} images; cannot hold out {n_val}")
        val.extend(int(i) for i in rng.choice(idxs, size=n_val, replace=False))
    return sorted(val)


def save_split(path: str | Path, val_indices: Sequence[int], n_train_total: int, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**meta, "n_train_total": n_train_total, "n_val": len(val_indices), "val_indices": list(val_indices)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def load_split(path: str | Path, n_train_total: int) -> tuple[list[int], list[int]]:
    """Return ``(train_indices, val_indices)`` into the HF train split."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload["n_train_total"] != n_train_total:
        raise ValueError(
            f"{path} was made for a train set of {payload['n_train_total']} images, "
            f"but the loaded one has {n_train_total}"
        )
    val = payload["val_indices"]
    val_set = set(val)
    train = [i for i in range(n_train_total) if i not in val_set]
    return train, val


def get_train_val_indices(cfg: dict, labels: Sequence[int]) -> tuple[list[int], list[int]]:
    """Load the committed split, creating it on first use."""
    data_cfg = cfg["data"]
    path = Path(data_cfg["val_indices_path"])
    if not path.exists():
        val = stratified_val_indices(labels, data_cfg["val_fraction"], data_cfg["split_seed"])
        save_split(
            path, val, len(labels),
            dataset=data_cfg["train_dataset"],
            val_fraction=data_cfg["val_fraction"],
            seed=data_cfg["split_seed"],
        )
    return load_split(path, len(labels))


def main() -> None:
    from src.data.cub_dataset import load_hf_split
    from src.utils.config import load_config

    parser = argparse.ArgumentParser(description="Create the stratified validation split.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--force", action="store_true", help="overwrite an existing split file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    path = Path(data_cfg["val_indices_path"])
    if path.exists() and not args.force:
        print(f"{path} already exists; pass --force to regenerate it.")
        return

    hf = load_hf_split("train", data_cfg["train_dataset"], data_cfg.get("cache_dir"))
    labels = hf["label"]
    val = stratified_val_indices(labels, data_cfg["val_fraction"], data_cfg["split_seed"])
    save_split(
        path, val, len(labels),
        dataset=data_cfg["train_dataset"],
        val_fraction=data_cfg["val_fraction"],
        seed=data_cfg["split_seed"],
    )
    per_class = np.bincount([labels[i] for i in val], minlength=data_cfg["num_classes"])
    print(f"Wrote {path}: {len(val)} val / {len(labels) - len(val)} train "
          f"(per class: min {per_class.min()}, max {per_class.max()})")


if __name__ == "__main__":
    main()
