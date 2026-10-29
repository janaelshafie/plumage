"""Image encoders. An encoder turns an image into a feature vector.

Every encoder has the same methods, so the classifier, the training loop and
Grad-CAM work with any of them:
    forward_features(x)  feature map before pooling, shape [B, C, H, W]
                         (Milestone 2 colorization also uses this)
    forward(x)           pooled feature vector, shape [B, out_dim]
    out_dim              size of that vector
    get_gradcam_layer()  the last conv layer, for Grad-CAM

To add an encoder (e.g. pretrained ResNet-50 in Step 4), write a class like
ScratchCNN and add its name to build_encoder() at the bottom.
"""

import torch.nn as nn


def conv_block(in_channels, out_channels):
    """Two 3x3 convolutions (each followed by BatchNorm and ReLU), then a 2x2 max pool
    that halves the height and width."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class ScratchCNN(nn.Module):
    """Our CNN trained from scratch: 5 conv blocks (32 -> 512 channels), then global average pooling.

    A 224x224 image is halved 5 times: 224 -> 112 -> 56 -> 28 -> 14 -> 7,
    so the feature map is 512 x 7 x 7, and pooling turns it into 512 numbers.
    """

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # average each 7x7 map into one number
        self.out_dim = 512

    def forward_features(self, x):
        return self.blocks(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pool(x)
        return x.flatten(1)  # [B, 512, 1, 1] -> [B, 512]

    def get_gradcam_layer(self):
        # The second conv (index 3) of the last block.
        return self.blocks[-1][3]


def build_encoder(name):
    """Create an encoder from its name in the config (`model.encoder`)."""
    if name == "scratch_cnn":
        return ScratchCNN()
    raise ValueError(f"Unknown encoder: {name}")
