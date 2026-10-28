"""Training curves figure; no dataset download."""

from src.visualization.curves import plot_training_curves


def test_plot_training_curves_writes_file(tmp_path):
    history = {
        "train_loss": [2.0, 1.5, 1.1],
        "val_loss": [2.1, 1.8, 1.6],
        "train_acc": [0.1, 0.3, 0.5],
        "val_acc": [0.1, 0.2, 0.3],
    }
    out_path = plot_training_curves(history, tmp_path / "nested" / "curves.png")
    assert out_path.exists()
    assert out_path.stat().st_size > 0
