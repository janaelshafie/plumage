"""Classification metrics, used for every model."""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Run the model on a whole loader and compute the metrics.

    accuracy         share of images whose top prediction is the right species
    top5_accuracy    share of images where the right species is in the 5 highest scores
    macro_precision, macro_recall, macro_f1
                     computed for each species and then averaged, so every species
                     counts the same. This matters because the test set has
                     11 to 30 images per species, and accuracy alone hides the small ones.
    confusion_matrix rows = true species, columns = predicted species
    predictions, labels
                     one entry per image, in loader order (used to pick examples for Grad-CAM)
    """
    model.eval()
    all_outputs, all_labels = [], []
    for images, labels in loader:
        outputs = model(images.to(device))
        all_outputs.append(outputs.cpu())
        all_labels.append(labels)

    outputs = torch.cat(all_outputs)
    labels = torch.cat(all_labels)
    predictions = outputs.argmax(dim=1)

    top5 = outputs.topk(5, dim=1).indices                       # [N, 5]
    top5_correct = (top5 == labels.unsqueeze(1)).any(dim=1)     # is the label one of the 5?

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0)

    return {
        "accuracy": (predictions == labels).float().mean().item(),
        "top5_accuracy": top5_correct.float().mean().item(),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(num_classes))),
        "predictions": predictions.numpy(),
        "labels": labels.numpy(),
    }


def most_confused_pairs(cm, class_names, k=5):
    """The k species pairs the model mixes up most often.

    A pair counts both directions: A predicted as B plus B predicted as A,
    because "Least Auklet vs Crested Auklet" is one problem, not two.
    Returns a list of dicts, most confused first.
    """
    cm = np.asarray(cm)
    mistakes = cm + cm.T            # [a, b] = mix-ups between a and b, both directions
    np.fill_diagonal(mistakes, 0)   # the diagonal holds correct predictions, not mistakes

    pairs = []
    for a in range(len(class_names)):
        for b in range(a + 1, len(class_names)):  # b > a, so each pair is counted once
            if mistakes[a, b] > 0:
                pairs.append({"species_a": class_names[a], "species_b": class_names[b],
                              "a_predicted_as_b": int(cm[a, b]), "b_predicted_as_a": int(cm[b, a]),
                              "total": int(mistakes[a, b])})
    pairs.sort(key=lambda p: p["total"], reverse=True)
    return pairs[:k]
