"""Command line interface. Every part of the project is run from here.

    python -m src.cli eda     --config configs/base.yaml
    python -m src.cli train   --task classification --config configs/cls_scratch.yaml
    python -m src.cli tune    --task classification --method grid --config configs/cls_resnet50_finetune.yaml
    python -m src.cli predict --task classification --image bird.jpg --checkpoint best.pt --gradcam
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # save figures to files without opening windows

import torch
from PIL import Image

from src.data.dataloaders import build_dataloaders
from src.data.transforms import denormalize, get_eval_transforms, get_mean_std
from src.evaluation.metrics import evaluate
from src.models.classifier import build_model, load_checkpoint
from src.training.trainer import train_model
from src.training.tuning import grid_search, random_search
from src.utils.config import get_device, load_config, set_seed
from src.utils.tracking import log_file, log_folder, log_metrics, start_run
from src.visualization.curves import plot_training_curves
from src.visualization.eda_plots import run_eda
from src.visualization.gradcam import GradCAM, save_gradcam_figure


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
    # Name the run after the config file (e.g. "cls_resnet50_frozen"), so two configs
    # with the same encoder don't overwrite each other's checkpoint and curves.
    run_name = os.path.splitext(os.path.basename(args.config))[0]

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
    if args.task != "classification":
        print("Tuning is only required for classification (Milestone 1).")
        return

    search = grid_search if args.method == "grid" else random_search
    trials = search(cfg)

    best = max(trials, key=lambda t: t["best_val_acc"])
    total_time = sum(t["time_sec"] for t in trials)
    print(f"\n{args.method} search results:")
    for t in trials:
        print(f"  lr {t['lr']:.2e}  weight decay {t['weight_decay']:.2e}  "
              f"val acc {t['best_val_acc']:.3f}  ({t['time_sec'] / 60:.1f} min)")
    print(f"Best: lr {best['lr']:.2e}, weight decay {best['weight_decay']:.2e}, val acc {best['best_val_acc']:.3f}")
    print(f"{len(trials)} trials in {total_time / 60:.1f} min")

    # Saved so the grid vs random table can be made from it.
    path = os.path.join(cfg["output_dir"], "tuning", args.method + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"method": args.method, "trials": trials, "best": best, "total_time_sec": total_time}, f, indent=1)
    print(f"Saved to {path}")


def predict_command(args, cfg):
    if args.task != "classification":
        print(f"Prediction for '{args.task}' is part of Milestone 2.")
        return

    device = get_device()
    # The model and its own config (encoder, normalization) come from the checkpoint.
    model, model_cfg = load_checkpoint(args.checkpoint, device)
    with open(cfg["data"]["class_names_path"], encoding="utf-8") as f:
        class_names = json.load(f)

    # Same preprocessing as val/test: resize 256, center crop 224, normalize.
    image = get_eval_transforms(model_cfg)(Image.open(args.image).convert("RGB")).to(device)

    if args.gradcam:
        gradcam = GradCAM(model)
        cam, _, probs = gradcam(image)  # Grad-CAM also gives the probabilities
        gradcam.remove()
    else:
        with torch.no_grad():
            probs = model(image.unsqueeze(0)).softmax(dim=1)[0].cpu()

    top_probs, top_classes = probs.topk(5)
    print(f"Top 5 for {args.image}:")
    for rank, (p, c) in enumerate(zip(top_probs.tolist(), top_classes.tolist()), start=1):
        print(f"  {rank}. {class_names[c]:<35} {p:6.1%}")

    if args.gradcam:
        # Named after the image and the checkpoint, so two models don't overwrite each other.
        image_name = os.path.splitext(os.path.basename(args.image))[0]
        model_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        path = os.path.join(cfg["output_dir"], "predictions", f"{image_name}_{model_name}.png")
        shown = denormalize(image.cpu(), *get_mean_std(model_cfg))
        title = f"{model_name}: {class_names[top_classes[0]]} ({top_probs[0]:.0%})"
        save_gradcam_figure(shown, cam, title, path)
        print(f"Saved Grad-CAM to {path}")


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
