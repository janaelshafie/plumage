"""Classification metrics shared by every model (scratch, transfer, tuning)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Fraction of rows where the true label is among the top ``k`` logits."""
    k = min(k, logits.shape[1])
    top_k = logits.topk(k, dim=1).indices
    return (top_k == labels.unsqueeze(1)).any(dim=1).float().mean().item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run ``model`` over ``loader`` and return accuracy, top-k accuracy, macro P/R/F1
    and the confusion matrix. Macro metrics matter here because CUB's test split is
    imbalanced (11-30 images per class) while accuracy alone hides the rare classes.
    """
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        all_logits.append(model(images).cpu())
        all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    predictions = logits.argmax(dim=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.numpy(), predictions.numpy(), average="macro", zero_division=0
    )
    return {
        "accuracy": (predictions == labels).float().mean().item(),
        f"top{top_k}_accuracy": topk_accuracy(logits, labels, top_k),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(labels.numpy(), predictions.numpy(), labels=list(range(num_classes))),
        "labels": labels.numpy(),
        "predictions": predictions.numpy(),
    }
