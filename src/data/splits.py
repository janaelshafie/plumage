"""Train / validation split.

Hugging Face only gives us train and test, and we must never tune on test.
So we hold out 10% of each class of the train set as validation.

The chosen indices are saved to configs/splits/val_indices.json and committed,
so everyone on the team validates on exactly the same 600 images.
"""

import json
import os

import numpy as np


def make_val_split(labels, val_fraction, seed):
    """Pick `val_fraction` of the images of every class (at least 1 per class).

    We split per class because each class has only ~30 images: a plain random
    10% could leave some classes with no validation images at all.
    """
    rng = np.random.default_rng(seed)
    val_indices = []
    for c in sorted(set(labels)):
        class_indices = [i for i, label in enumerate(labels) if label == c]
        n_val = max(1, round(val_fraction * len(class_indices)))
        chosen = rng.choice(class_indices, size=n_val, replace=False)
        val_indices.extend(int(i) for i in chosen)
    return sorted(val_indices)


def get_train_val_indices(cfg, labels):
    """Return (train_indices, val_indices) into the Hugging Face train split.

    Loads the saved split from val_indices.json. If the file doesn't exist yet
    it is created first.
    """
    data_cfg = cfg["data"]
    path = data_cfg["val_indices_path"]

    if not os.path.exists(path):
        val_indices = make_val_split(labels, data_cfg["val_fraction"], data_cfg["split_seed"])
        info = {
            "dataset": data_cfg["train_dataset"],
            "val_fraction": data_cfg["val_fraction"],
            "seed": data_cfg["split_seed"],
            "n_train_total": len(labels),
            "n_val": len(val_indices),
            "val_indices": val_indices,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)

    with open(path, encoding="utf-8") as f:
        val_indices = json.load(f)["val_indices"]

    val_set = set(val_indices)
    train_indices = [i for i in range(len(labels)) if i not in val_set]
    return train_indices, val_indices
