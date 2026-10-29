"""Image transforms (preprocessing and augmentation). Settings are under `data.augment` in the config.

Training: random crop, horizontal flip, small rotation and small color changes.
Val/test: resize to 256 and take the center 224x224 crop (no randomness).

We don't flip vertically (birds are never upside down in CUB), and we keep the
hue change very small because plumage color is what tells many species apart.
"""

import torch
from torchvision import transforms as T


def get_mean_std(cfg):
    """ImageNet mean/std for pretrained models, CUB mean/std for models trained from scratch."""
    data = cfg["data"]
    if data["normalization"] == "cub":
        return data["cub_mean"], data["cub_std"]
    return data["imagenet_mean"], data["imagenet_std"]


def get_train_transforms(cfg):
    size = cfg["data"]["image_size"]
    aug = cfg["data"]["augment"]
    mean, std = get_mean_std(cfg)
    return T.Compose([
        T.RandomResizedCrop(size, scale=aug["crop_scale"]),
        T.RandomHorizontalFlip(aug["hflip_p"]),
        T.RandomRotation(aug["rotation_deg"]),
        T.ColorJitter(**aug["color_jitter"]),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def get_eval_transforms(cfg):
    size = cfg["data"]["image_size"]
    mean, std = get_mean_std(cfg)
    return T.Compose([
        T.Resize(cfg["data"]["augment"]["eval_resize"]),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def denormalize(image, mean, std):
    """Undo Normalize on a [3, H, W] tensor so it can be shown with matplotlib."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)
