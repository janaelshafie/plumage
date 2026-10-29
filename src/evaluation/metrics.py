"""Classification metrics, used for every model."""

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
    }
