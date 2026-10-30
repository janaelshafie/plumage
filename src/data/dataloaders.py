"""Builds the train, val and test DataLoaders."""

from torch.utils.data import DataLoader

from src.data.cub_dataset import CUBDataset, load_cub
from src.data.splits import get_train_val_indices
from src.data.transforms import get_eval_transforms, get_train_transforms


def build_dataloaders(cfg):
    """Return (train_loader, val_loader, test_loader).

    train: the HF train split, minus the val images and the duplicates found in the EDA
    val:   10% of each class of the HF train split (fixed, see splits.py)
    test:  the whole HF test split, used only once for the final evaluation
    """
    data_cfg = cfg["data"]
    hf_train = load_cub("train", cfg)
    hf_test = load_cub("test", cfg)

    train_indices, val_indices = get_train_val_indices(cfg, list(hf_train["label"]))

    # Remove the duplicate images from training (val and test are not changed).
    file_names = hf_train["file_name"]
    duplicates = set(data_cfg["exclude_train_files"])
    train_indices = [i for i in train_indices if file_names[i] not in duplicates]

    train_set = CUBDataset(hf_train, train_indices, get_train_transforms(cfg))
    val_set = CUBDataset(hf_train, val_indices, get_eval_transforms(cfg))
    test_set = CUBDataset(hf_test, list(range(len(hf_test))), get_eval_transforms(cfg))

    batch_size = data_cfg["batch_size"]
    workers = data_cfg["num_workers"]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader, test_loader
