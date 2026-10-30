"""Training curves: train vs validation loss and accuracy."""

import os

import matplotlib.pyplot as plt


def plot_training_curves(history, path):
    """Save a figure with two plots (loss and accuracy) per epoch to `path`."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(10, 4))

    loss_ax.plot(epochs, history["train_loss"], label="train")
    loss_ax.plot(epochs, history["val_loss"], label="val")
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("epoch")
    loss_ax.legend()

    acc_ax.plot(epochs, history["train_acc"], label="train")
    acc_ax.plot(epochs, history["val_acc"], label="val")
    acc_ax.set_title("Accuracy")
    acc_ax.set_xlabel("epoch")
    acc_ax.legend()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
