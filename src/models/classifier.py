"""Species classifier: an encoder followed by one linear layer."""

import torch
import torch.nn as nn

from src.models.encoders import build_encoder


class Classifier(nn.Module):
    """encoder -> dropout -> linear layer with one score per species.

    freeze_encoder=True locks the encoder, so only the last layer (fc) learns.
    We use this for a pretrained encoder, to keep what it learned on ImageNet.
    """

    def __init__(self, encoder, num_classes, dropout, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(encoder.out_dim, num_classes)
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            # No gradients for the encoder, so the optimizer never changes its weights.
            for p in self.encoder.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            # BatchNorm layers update their running mean/std in train mode even without
            # gradients. Keeping the encoder in eval mode keeps it really frozen.
            self.encoder.eval()
        return self

    def forward(self, x):
        features = self.encoder(x)
        return self.fc(self.dropout(features))

    def get_gradcam_layer(self):
        return self.encoder.get_gradcam_layer()


def build_model(cfg, pretrained=True):
    """Build the classifier from the `model:` section of the config.

    To use another encoder, only `model.encoder` in the config changes.
    `model.freeze_backbone: true` trains only the last layer.
    """
    model_cfg = cfg["model"]
    encoder = build_encoder(model_cfg["encoder"], pretrained)
    return Classifier(encoder, cfg["data"]["num_classes"], model_cfg["dropout"],
                      freeze_encoder=model_cfg.get("freeze_backbone", False))


def load_checkpoint(path, device):
    """Rebuild a trained model from a checkpoint saved by train_model().

    The checkpoint holds the config it was trained with, so we know which encoder
    to build and which normalization its images need. Returns (model, cfg).
    """
    # weights_only=False because the file also holds the config dict, not only tensors.
    # Only load checkpoints you trust (here: our own).
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = build_model(cfg, pretrained=False)  # our trained weights replace everything below
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, cfg
