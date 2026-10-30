"""Command line interface. Every part of the project is run from here.

    python -m src.cli eda     --config configs/base.yaml
    python -m src.cli train   --task classification --config configs/cls_scratch.yaml
    python -m src.cli tune    --task classification --method grid --config configs/tune_resnet50.yaml
    python -m src.cli predict --task classification --image bird.jpg --checkpoint best.pt --gradcam

`tune` (Step 4) and `predict` (Step 5) are not written yet.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # save figures to files without opening windows

import torch

from src.data.dataloaders import build_dataloaders
from src.evaluation.metrics import evaluate
from src.models.classifier import build_model
from src.training.trainer import train_model
from src.utils.config import get_device, load_config, set_seed
from src.utils.tracking import log_file, log_folder, log_metrics, start_run
from src.visualization.curves import plot_training_curves
from src.visualization.eda_plots import run_eda


def eda_command(args, cfg):
    summary, out_dir = run_eda(cfg)
    print(f"Saved figures and summary.json to {out_dir}")
    print(f"CUB train mean/std: {summary['channel_stats_train']}")

    with start_run(cfg, "eda"):
        log_metrics({
            "train/imbalance_ratio": summary["class_balance"]["train"]["imbalance_ratio"],
            "test/imbalance_ratio": summary["class_balance"]["test"]["imbalance_ratio"],
            "duplicates/train_train": summary["duplicates"]["train_train_pairs"],
            "duplicates/train_test": summary["duplicates"]["train_test_pairs"],
            "captions/vocab_clean": summary["captions"]["vocab_clean"],
        })
        log_folder(out_dir)


def train_command(args, cfg):
    if args.task != "classification":
        print(f"Training for '{args.task}' is part of Milestone 2.")
        return

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = build_model(cfg)
    run_name = cfg["model"]["encoder"] + "-baseline"

    with start_run(cfg, run_name):
        history, checkpoint_path = train_model(model, train_loader, val_loader, cfg, run_name)

        curves_path = os.path.join(cfg["output_dir"], "figures", "curves", run_name + ".png")
        plot_training_curves(history, curves_path)
        log_file(curves_path)
        log_file(checkpoint_path)

        # Load the best epoch's weights and evaluate once on the test set.
        device = get_device()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        results = evaluate(model, test_loader, device, cfg["data"]["num_classes"])

        log_metrics({
            "test/accuracy": results["accuracy"],
            "test/top5_accuracy": results["top5_accuracy"],
            "test/macro_precision": results["macro_precision"],
            "test/macro_recall": results["macro_recall"],
            "test/macro_f1": results["macro_f1"],
        })
        print(f"Test accuracy {results['accuracy']:.3f}, macro F1 {results['macro_f1']:.3f}")


def tune_command(args, cfg):
    print("`tune` is not implemented yet (Step 4).")


def predict_command(args, cfg):
    print("`predict` is not implemented yet (Step 5).")


def main():
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="Plumage: CUB-200 bird project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eda = subparsers.add_parser("eda", help="data analysis, saves figures to outputs/figures/eda/")
    train = subparsers.add_parser("train", help="train a model")
    tune = subparsers.add_parser("tune", help="hyperparameter search")
    predict = subparsers.add_parser("predict", help="predict the species of one image")

    # Arguments every command has
    for sub in [eda, train, tune, predict]:
        sub.add_argument("--task", default="classification",
                         choices=["classification", "colorization", "captioning"])
        sub.add_argument("--config", default="configs/base.yaml", help="path to a YAML config")

    tune.add_argument("--method", choices=["grid", "random"], default="grid")
    predict.add_argument("--image", required=True, help="path to an image")
    predict.add_argument("--checkpoint", required=True, help="path to a saved model (.pt)")
    predict.add_argument("--gradcam", action="store_true", help="also save a Grad-CAM heatmap")

    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    commands = {"eda": eda_command, "train": train_command, "tune": tune_command, "predict": predict_command}
    commands[args.command](args, cfg)


if __name__ == "__main__":
    main()
