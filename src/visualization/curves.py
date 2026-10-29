"""Train vs validation loss/accuracy curves from a ``Trainer`` history."""

from __future__ import annotations

from pathlib import Path


def plot_training_curves(history: dict[str, list[float]], out_path: str | Path) -> Path:
    """Save a two-panel (loss, accuracy) figure of ``history`` to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in [(loss_ax, "loss", "Loss"), (acc_ax, "acc", "Accuracy")]:
        ax.plot(epochs, history[f"train_{key}"], label="train")
        ax.plot(epochs, history[f"val_{key}"], label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
