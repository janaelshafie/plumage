"""Training loop. Every classification model goes through train_model(),
so a new encoder only needs a new config, not new training code.

Settings come from the `train:` section of the config:
epochs, lr, weight_decay, early_stopping_patience, amp, checkpoint_dir.
"""

import os

import torch
import torch.nn as nn

from src.utils.config import get_device
from src.utils.tracking import log_metrics


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, use_amp):
    """Train on every batch once. Returns (average loss, accuracy)."""
    model.train()
    total_loss, correct, count = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Mixed precision: on the GPU some math runs in float16, which is faster.
        with torch.autocast(device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # The scaler scales the loss up so small float16 gradients don't become 0.
        # When AMP is off it does nothing and this is a normal backward + step.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        count += len(labels)

    return total_loss / count, correct / count


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Compute loss and accuracy without training. Returns (average loss, accuracy)."""
    model.eval()
    total_loss, correct, count = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        count += len(labels)

    return total_loss / count, correct / count


def train_model(model, train_loader, val_loader, cfg, run_name):
    """Train with AdamW and a cosine learning-rate schedule.

    After every epoch we check the validation accuracy:
      - if it is the best so far, the model is saved as a checkpoint
      - if it hasn't improved for `early_stopping_patience` epochs, we stop

    Returns (history, checkpoint_path). history has one value per epoch for
    train_loss, train_acc, val_loss and val_acc.
    """
    train_cfg = cfg["train"]
    epochs = train_cfg["epochs"]
    patience = train_cfg["early_stopping_patience"]

    device = get_device()
    model.to(device)
    use_amp = train_cfg["amp"] and device.type == "cuda"

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    # Lowers the learning rate smoothly from lr to ~0 over all epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(train_cfg["checkpoint_dir"], run_name + "_best.pt")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, use_amp)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        log_metrics({"train/loss": train_loss, "train/acc": train_acc,
                     "val/loss": val_loss, "val/acc": val_acc, "lr": lr}, step=epoch)

        print(f"epoch {epoch}/{epochs}  train loss {train_loss:.3f} acc {train_acc:.3f}"
              f"  val loss {val_loss:.3f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_acc": val_acc, "config": cfg},
                       checkpoint_path)
            print("  new best model saved")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping: val accuracy did not improve for {patience} epochs")
                break

    print(f"Best val accuracy: {best_val_acc:.3f}")
    return history, checkpoint_path
