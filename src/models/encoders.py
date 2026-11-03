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
from torchvision import models


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


class ResNet50Encoder(nn.Module):
    """ResNet-50 pretrained on ImageNet (1.2M photos, 1000 classes).

    We keep every layer up to layer4 and drop the last two: avgpool and fc
    (the 1000-class ImageNet layer). Our Classifier adds its own 200-species layer.

    A 224x224 image gives a 2048 x 7 x 7 feature map, and pooling turns it into 2048 numbers.

    pretrained=False skips the ImageNet download. We use it when loading our own
    checkpoint, because its weights replace the ImageNet ones anyway.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # Downloads ~100 MB the first time when pretrained.
        resnet = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        # children() = conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 2048

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pool(x)
        return x.flatten(1)  # [B, 2048, 1, 1] -> [B, 2048]

    def get_gradcam_layer(self):
        # layer4, the last block of ResNet-50.
        return self.backbone[-1]


def build_encoder(name, pretrained=True):
    """Create an encoder from its name in the config (`model.encoder`).

    pretrained only matters for pretrained encoders (ResNet-50); ScratchCNN always starts random.
    """
    if name == "scratch_cnn":
        return ScratchCNN()
    if name == "resnet50":
        return ResNet50Encoder(pretrained)
    raise ValueError(f"Unknown encoder: {name}")
