"""The CUB-200-2011 dataset, downloaded from Hugging Face.

Each row has: image (PIL image), description (10 captions separated by
newlines), label (0-199) and file_name.
"""

from datasets import load_dataset
from torch.utils.data import Dataset


def load_cub(split, cfg):
    """Load the "train" or "test" split. The first call downloads it (~1.2 GB)."""
    if split == "train":
        name = cfg["data"]["train_dataset"]
    else:
        name = cfg["data"]["test_dataset"]
    return load_dataset(name, split=split, cache_dir=cfg["data"]["cache_dir"])


class CUBDataset(Dataset):
    """Gives (image_tensor, label) pairs for PyTorch.

    hf_data:   a split returned by load_cub()
    indices:   which rows of hf_data to use (this is how train and val are separated)
    transform: turns the PIL image into a tensor (see transforms.py)
    """

    def __init__(self, hf_data, indices, transform):
        self.hf_data = hf_data
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.hf_data[self.indices[i]]
        # 8 CUB images are grayscale, but the model needs 3 color channels.
        image = row["image"].convert("RGB")
        return self.transform(image), row["label"]
