"""Generic train/validate loop. Every model (scratch, transfer, tuning trials)
goes through this one ``Trainer`` so a new encoder or search method never
touches the training code, only the config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import get_device
from src.utils.tracking import Tracker


@dataclass
class TrainResult:
    history: dict[str, list[float]] = field(default_factory=dict)
    best_val_acc: float = -1.0
    best_epoch: int = -1
    checkpoint_path: Path | None = None
    stopped_early: bool = False


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint saved by ``Trainer.fit`` (``model_state``, ``epoch``, ``val_acc``, ``config``)."""
    return torch.load(path, map_location=map_location, weights_only=False)


class Trainer:
    """Reads its settings from ``cfg["train"]``:

    ``epochs``, ``lr``, ``weight_decay``, ``early_stopping_patience``, ``amp``
    (mixed precision; only takes effect on CUDA), ``checkpoint_dir``.
    """

    def __init__(self, model: nn.Module, cfg: dict, tracker: Tracker | None = None) -> None:
        self.model = model
        self.cfg = cfg
        train_cfg = cfg.get("train", {})
        self.epochs: int = train_cfg.get("epochs", 40)
        self.patience: int = train_cfg.get("early_stopping_patience", 10)
        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", Path(cfg.get("output_dir", "outputs")) / "checkpoints"))

        self.device = get_device(cfg.get("device"))
        self.model.to(self.device)
        self.amp = bool(train_cfg.get("amp", True)) and self.device.type == "cuda"
        self.tracker = tracker if tracker is not None else Tracker()

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_cfg.get("lr", 3e-4), weight_decay=train_cfg.get("weight_decay", 1e-4)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.criterion = nn.CrossEntropyLoss()

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss, correct, n = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=self.device.type, enabled=self.amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                correct += (logits.argmax(dim=1) == labels).sum().item()
                n += batch_size
        return total_loss / n, correct / n

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, run_name: str = "run") -> TrainResult:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"{run_name}_best.pt"

        result = TrainResult(history={"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []})
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)
            self.scheduler.step()

            result.history["train_loss"].append(train_loss)
            result.history["train_acc"].append(train_acc)
            result.history["val_loss"].append(val_loss)
            result.history["val_acc"].append(val_acc)
            self.tracker.log_metrics(
                {
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "lr": self.optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

            improved = val_acc > result.best_val_acc
            if improved:
                result.best_val_acc, result.best_epoch = val_acc, epoch
                epochs_without_improvement = 0
                torch.save(
                    {"model_state": self.model.state_dict(), "epoch": epoch, "val_acc": val_acc, "config": self.cfg},
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            marker = "  * new best" if improved else ""
            print(
                f"epoch {epoch}/{self.epochs}  train loss {train_loss:.3f} acc {train_acc:.3f}"
                f"  val loss {val_loss:.3f} acc {val_acc:.3f}{marker}"
            )

            if epochs_without_improvement >= self.patience:
                print(f"early stopping: no val improvement for {self.patience} epochs")
                result.stopped_early = True
                break

        result.checkpoint_path = checkpoint_path if result.best_epoch > 0 else None
        return result
