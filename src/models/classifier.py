"""Species classifier: an encoder followed by one linear layer."""

import torch.nn as nn

from src.models.encoders import build_encoder


class Classifier(nn.Module):
    """encoder -> dropout -> linear layer with one score per species."""

    def __init__(self, encoder, num_classes, dropout):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(encoder.out_dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return self.fc(self.dropout(features))

    def get_gradcam_layer(self):
        return self.encoder.get_gradcam_layer()


def build_model(cfg):
    """Build the classifier from the `model:` section of the config.

    To use another encoder, only `model.encoder` in the config changes.
    """
    encoder = build_encoder(cfg["model"]["encoder"])
    return Classifier(encoder, cfg["data"]["num_classes"], cfg["model"]["dropout"])
