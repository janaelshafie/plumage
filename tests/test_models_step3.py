"""Scratch encoder and classifier registry; no dataset download."""

import pytest
import torch

from src.models.classifier import Classifier, build_model
from src.models.encoders import ScratchCNN, build_encoder


def test_scratch_cnn_shapes():
    model = ScratchCNN()
    x = torch.randn(2, 3, 64, 64)
    features = model.forward_features(x)
    assert features.shape == (2, 512, 2, 2)
    pooled = model(x)
    assert pooled.shape == (2, model.out_dim)
    assert model.out_dim == 512


def test_scratch_cnn_gradcam_layer_is_last_conv():
    model = ScratchCNN()
    assert model.gradcam_layer is model.blocks[-1].conv2


def test_build_encoder_unknown_name_raises():
    with pytest.raises(ValueError):
        build_encoder("not_a_real_encoder")


def test_classifier_forward_and_gradcam_layer():
    encoder = ScratchCNN()
    model = Classifier(encoder, num_classes=200, dropout=0.3)
    logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 200)
    assert model.gradcam_layer is encoder.gradcam_layer


def test_build_model_from_config():
    cfg = {"model": {"encoder": "scratch_cnn", "dropout": 0.5}, "data": {"num_classes": 200}}
    model = build_model(cfg)
    assert isinstance(model, Classifier)
    assert model.fc.out_features == 200
