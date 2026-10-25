"""CUB-200-2011 over the Hugging Face ``Multimodal-Fatima/CUB_train`` / ``CUB_test`` splits.

Each HF row has ``image`` (PIL), ``description`` (10 captions joined by ``\\n``),
``label`` (0-199), ``file_name`` and ``id``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

HF_REPOS = {
    "train": "Multimodal-Fatima/CUB_train",
    "test": "Multimodal-Fatima/CUB_test",
}


def load_hf_split(split: str, repo: str | None = None, cache_dir: str | None = None) -> HFDataset:
    """Download (or reuse the cache of) one CUB split as a Hugging Face dataset."""
    if split not in HF_REPOS:
        raise ValueError(f"split must be one of {list(HF_REPOS)}, got {split!r}")
    return load_dataset(repo or HF_REPOS[split], split=split, cache_dir=cache_dir)


class CUBDataset(Dataset):
    """Returns ``(image_tensor, label)``.

    Args:
        split: ``"train"`` or ``"test"``. The validation set is a subset of
            ``"train"`` chosen with ``indices`` (see ``src/data/splits.py``).
        transform: callable applied to the RGB PIL image. It should return a
            tensor; if ``None`` the image is only converted with ``to_tensor``.
        indices: optional row indices into the HF split to restrict to.
        hf_dataset: an already-loaded HF dataset, so train and val can share
            one download instead of loading it twice.
        repo, cache_dir: passed to ``load_dataset`` when ``hf_dataset`` is None.
    """

    def __init__(
        self,
        split: str = "train",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        indices: Sequence[int] | None = None,
        hf_dataset: HFDataset | None = None,
        repo: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.split = split
        self.transform = transform
        self.hf = hf_dataset if hf_dataset is not None else load_hf_split(split, repo, cache_dir)
        self.indices = list(indices) if indices is not None else list(range(len(self.hf)))
        self.classes: list[str] = list(self.hf.features["label"].names)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.hf[self.indices[idx]]
        image = row["image"]
        # CUB has a few grayscale / CMYK images; models expect 3 channels.
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image) if self.transform is not None else TF.to_tensor(image)
        return image, int(row["label"])

    @property
    def labels(self) -> list[int]:
        """Labels for the rows in this dataset, read without decoding any image."""
        all_labels = self.hf["label"]
        return [int(all_labels[i]) for i in self.indices]

    def get_description(self, idx: int) -> str:
        """Raw caption block (10 captions joined by newlines) for item ``idx``."""
        return self.hf[self.indices[idx]]["description"]

    def get_file_name(self, idx: int) -> str:
        return self.hf[self.indices[idx]]["file_name"]
