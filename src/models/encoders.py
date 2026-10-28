"""Image encoders, registered by name so the trainer never imports a concrete class.

Step 4 (Mohamed) extends ``ENCODERS`` with pretrained backbones (ResNet-50, EfficientNet-B0)
behind the same interface: ``forward_features(x)`` -> spatial map ``[B, C, H, W]``, ``out_dim``,
``gradcam_layer`` (the last conv layer, for Milestone 1 Step 5 Grad-CAM). Milestone 2
colorization reuses ``ScratchCNN.forward_features``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRATCH_CHANNELS = (32, 64, 128, 256, 512)


class ConvBlock(nn.Module):
    """``Conv-BN-ReLU x2`` followed by a ``2x2`` max pool."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.pool(x)


class ScratchCNN(nn.Module):
    """Five ``ConvBlock``s, channels 32 -> 512, then global average pooling.

    224x224 input is downsampled by 2**5 = 32 to a 7x7 feature map before pooling.
    """

    def __init__(self, in_channels: int = 3, channels: tuple[int, ...] = SCRATCH_CHANNELS) -> None:
        super().__init__()
        widths = [in_channels, *channels]
        self.blocks = nn.ModuleList([ConvBlock(widths[i], widths[i + 1]) for i in range(len(channels))])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = channels[-1]

    @property
    def gradcam_layer(self) -> nn.Module:
        """Last conv layer, for Grad-CAM hooks (Step 5)."""
        return self.blocks[-1].conv2

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial feature map ``[B, out_dim, H', W']`` before pooling."""
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled feature vector ``[B, out_dim]``."""
        features = self.forward_features(x)
        return torch.flatten(self.pool(features), 1)


ENCODERS: dict[str, type[nn.Module]] = {
    "scratch_cnn": ScratchCNN,
}


def build_encoder(name: str, **kwargs) -> nn.Module:
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; available: {sorted(ENCODERS)}")
    return ENCODERS[name](**kwargs)
