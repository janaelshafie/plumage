"""Trainer loop on tiny synthetic data; no dataset download, no GPU."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.classifier import build_model
from src.training.trainer import Trainer, load_checkpoint


def _tiny_loaders(num_classes=4, n=8, image_size=64, batch_size=4):
    images = torch.randn(n, 3, image_size, image_size)
    labels = torch.randint(0, num_classes, (n,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size)
    return loader, loader  # reuse as train/val; only wiring is under test


def test_trainer_runs_and_improves_loss(tmp_path):
    cfg = {
        "model": {"encoder": "scratch_cnn", "dropout": 0.0},
        "data": {"num_classes": 4},
        "train": {"epochs": 3, "lr": 1e-2, "early_stopping_patience": 10, "amp": False, "checkpoint_dir": str(tmp_path)},
        "output_dir": str(tmp_path),
    }
    model = build_model(cfg)
    train_loader, val_loader = _tiny_loaders(num_classes=cfg["data"]["num_classes"])

    trainer = Trainer(model, cfg)
    result = trainer.fit(train_loader, val_loader, run_name="unit-test")

    assert len(result.history["train_loss"]) == 3
    assert result.best_epoch >= 1
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()

    checkpoint = load_checkpoint(result.checkpoint_path)
    assert checkpoint["epoch"] == result.best_epoch
    assert "model_state" in checkpoint


def test_trainer_early_stops(tmp_path):
    cfg = {
        "model": {"encoder": "scratch_cnn", "dropout": 0.0},
        "data": {"num_classes": 4},
        "train": {
            "epochs": 50,
            "lr": 0.0,
            "early_stopping_patience": 2,
            "amp": False,
            "checkpoint_dir": str(tmp_path),
        },
    }
    model = build_model(cfg)
    # Freeze BatchNorm running stats so val accuracy is exactly reproducible
    # across epochs when lr=0 (otherwise BN stats still drift during the
    # training forward pass and could nudge a prediction's argmax).
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.momentum = 0.0
    train_loader, val_loader = _tiny_loaders(num_classes=cfg["data"]["num_classes"])

    trainer = Trainer(model, cfg)
    result = trainer.fit(train_loader, val_loader, run_name="unit-test-early-stop")

    # lr=0.0 means val accuracy never improves after epoch 1, so patience triggers.
    assert result.stopped_early
    assert len(result.history["train_loss"]) == 1 + cfg["train"]["early_stopping_patience"]
