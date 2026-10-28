"""Classifier head over an encoder, and the model registry the CLI builds from config."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders import build_encoder


class Classifier(nn.Module):
    """``encoder -> dropout -> linear`` over ``num_classes``."""

    def __init__(self, encoder: nn.Module, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(encoder.out_dim, num_classes)

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.encoder.gradcam_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.fc(self.dropout(features))


def build_model(cfg: dict) -> nn.Module:
    """Build the model named in ``cfg["model"]``.

    Swapping encoders (e.g. to Step 4's ResNet-50) means changing ``model.encoder``
    in the config; the trainer and CLI are unchanged.
    """
    model_cfg = cfg["model"]
    encoder = build_encoder(model_cfg["encoder"], **model_cfg.get("encoder_kwargs", {}))
    return Classifier(encoder, cfg["data"]["num_classes"], dropout=model_cfg.get("dropout", 0.3))
