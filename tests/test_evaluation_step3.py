"""Classification metrics on synthetic logits; no dataset download."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import evaluate, topk_accuracy


def test_topk_accuracy():
    # Only row 0's label is the top-1 logit; all three are within the top 2.
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.2, 0.7, 0.1], [0.1, 0.6, 0.3]])
    labels = torch.tensor([1, 0, 2])
    assert topk_accuracy(logits, labels, k=1) == pytest.approx(1 / 3)
    assert topk_accuracy(logits, labels, k=2) == 1.0


def test_topk_accuracy_caps_k_at_num_classes():
    logits = torch.randn(4, 3)
    labels = torch.randint(0, 3, (4,))
    assert topk_accuracy(logits, labels, k=10) == 1.0


class _OneHotLookup(nn.Module):
    """Returns the one-hot row of ``table`` for each integer input; a stand-in model
    so ``evaluate`` can be tested without any real encoder."""

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.table = table

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.table[x.long().view(-1)]


def test_evaluate_perfect_predictions():
    num_classes = 4
    model = _OneHotLookup(torch.eye(num_classes))
    inputs = torch.arange(num_classes)
    labels = torch.arange(num_classes)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)

    metrics = evaluate(model, loader, torch.device("cpu"), num_classes)
    assert metrics["accuracy"] == 1.0
    assert metrics["top5_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["confusion_matrix"].shape == (num_classes, num_classes)
    assert metrics["confusion_matrix"].trace() == num_classes


def test_evaluate_reports_macro_f1_below_one_on_mistakes():
    num_classes = 3
    table = torch.eye(num_classes)
    table[1] = table[0]  # class 1 is always predicted as class 0
    model = _OneHotLookup(table)
    inputs = torch.arange(num_classes)
    labels = torch.arange(num_classes)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=3)

    metrics = evaluate(model, loader, torch.device("cpu"), num_classes)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] < 1.0
