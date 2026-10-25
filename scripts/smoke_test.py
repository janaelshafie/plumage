"""End-to-end check of the Step 1 pipeline.

Downloads CUB (first run only), builds the three loaders, pulls one batch from
each, and logs a run to the experiment tracker.

    python scripts/smoke_test.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataloaders import build_dataloaders  # noqa: E402
from src.utils.config import load_config, set_seed  # noqa: E402
from src.utils.tracking import get_tracker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--no-track", action="store_true", help="skip logging to the tracker")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["num_workers"] = 0
    set_seed(cfg["seed"])
    num_classes = cfg["data"]["num_classes"]
    size = cfg["data"]["image_size"]

    t0 = time.time()
    loaders = dict(zip(["train", "val", "test"], build_dataloaders(cfg)))
    print(f"Loaders ready in {time.time() - t0:.1f}s")

    metrics = {}
    for name, loader in loaders.items():
        images, labels = next(iter(loader))
        assert images.ndim == 4 and images.shape[1:] == (3, size, size), images.shape
        assert labels.min() >= 0 and labels.max() < num_classes, (labels.min(), labels.max())
        n = len(loader.dataset)
        print(f"{name:>5}: {n:5d} images | batch {tuple(images.shape)} | labels {labels.min().item()}-{labels.max().item()}")
        metrics[f"{name}/n_images"] = n
        metrics[f"{name}/batch_pixel_mean"] = images.mean().item()

    train_ds, val_ds = loaders["train"].dataset, loaders["val"].dataset
    assert not set(train_ds.indices) & set(val_ds.indices), "train and val overlap"
    assert len(set(val_ds.labels)) == num_classes, "some class is missing from val"
    print("Train/val are disjoint and every class appears in val.")

    if not args.no_track:
        tracker = get_tracker(cfg)
        with tracker.run("smoke-test", tags={"step": "1", "owner": "jana", "kind": "smoke"}, config=cfg):
            tracker.log_metrics(metrics, step=0)
            print(f"Logged tracker run: {tracker.run_url}")

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
