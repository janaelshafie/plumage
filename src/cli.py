"""Command line interface. Every part of the project is run from here.

    python -m src.cli eda     --config configs/base.yaml
    python -m src.cli train   --task classification --config configs/cls_scratch.yaml
    python -m src.cli tune    --task classification --method grid --config configs/cls_resnet50_finetune.yaml
    python -m src.cli predict --task classification --image bird.jpg --checkpoint best.pt --gradcam
    python -m src.cli analyze --task classification --checkpoint scratch.pt --compare resnet50.pt
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # save figures to files without opening windows

import numpy as np
import torch
from PIL import Image

from src.data.cub_dataset import CUBDataset
from src.data.dataloaders import build_dataloaders
from src.data.transforms import denormalize, get_eval_transforms, get_mean_std
from src.evaluation.metrics import evaluate, most_confused_pairs
from src.models.classifier import build_model, load_checkpoint
from src.training.trainer import train_model
from src.training.tuning import grid_search, random_search
from src.utils.config import get_device, load_config, set_seed
from src.utils.tracking import log_file, log_folder, log_metrics, start_run
from src.visualization.curves import plot_training_curves
from src.visualization.eda_plots import run_eda
from src.visualization.gradcam import GradCAM, overlay, save_gradcam_figure, save_image_grid


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
    class_names = load_class_names(cfg)

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
        image_name = file_stem(args.image)
        model_name = file_stem(args.checkpoint)
        path = os.path.join(cfg["output_dir"], "predictions", f"{image_name}_{model_name}.png")
        shown = denormalize(image.cpu(), *get_mean_std(model_cfg))
        title = f"{model_name}: {class_names[top_classes[0]]} ({top_probs[0]:.0%})"
        save_gradcam_figure(shown, cam, title, path)
        print(f"Saved Grad-CAM to {path}")


def analyze_command(args, cfg):
    """Error analysis of one checkpoint on the test set:
    the most confused species pairs, and Grad-CAM on correct and wrong predictions.
    With --compare, the same images are also explained by a second model."""
    if args.task != "classification":
        print(f"Analysis for '{args.task}' is part of Milestone 2.")
        return

    device = get_device()
    model, model_cfg = load_checkpoint(args.checkpoint, device)
    class_names = load_class_names(cfg)
    model_name = file_stem(args.checkpoint)
    out_dir = os.path.join(cfg["output_dir"], "analysis", model_name)

    # The checkpoint's own config, so the test images get the normalization this model was trained with.
    _, _, test_loader = build_dataloaders(model_cfg)
    test_set = test_loader.dataset
    print(f"Evaluating {model_name} on {len(test_set)} test images...")
    results = evaluate(model, test_loader, device, len(class_names))
    pairs = most_confused_pairs(results["confusion_matrix"], class_names, k=5)

    print(f"Test accuracy {results['accuracy']:.3f}, macro F1 {results['macro_f1']:.3f}")
    print("Most confused species pairs (mistakes in both directions):")
    for p in pairs:
        print(f"  {p['species_a']} <-> {p['species_b']}: {p['total']} "
              f"({p['a_predicted_as_b']} + {p['b_predicted_as_a']})")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "confused_pairs.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": results["accuracy"], "macro_f1": results["macro_f1"], "pairs": pairs}, f, indent=1)

    # Pick random correct and wrong test images. The fixed seed gives the same images every run.
    rng = np.random.default_rng(cfg["seed"])
    is_correct = results["predictions"] == results["labels"]
    examples = {
        "correct": rng.choice(np.flatnonzero(is_correct), args.n_examples, replace=False),
        "wrong": rng.choice(np.flatnonzero(~is_correct), args.n_examples, replace=False),
    }

    mean_std = get_mean_std(model_cfg)
    gradcam = GradCAM(model)
    for kind, indices in examples.items():
        panels = []
        for i in indices:
            image, label = test_set[i]
            cam, pred, probs = gradcam(image.to(device))
            if kind == "correct":
                title = f"{class_names[label]} ({probs[pred]:.0%})"
            else:
                title = f"true: {class_names[label]}\npred: {class_names[pred]} ({probs[pred]:.0%})"
            panels.append((overlay(denormalize(image, *mean_std), cam), title))
        save_image_grid(panels, os.path.join(out_dir, f"gradcam_{kind}.png"))
    gradcam.remove()

    if args.compare:
        # Half correct and half wrong examples (for the first model). Each row:
        # input | first model's Grad-CAM | second model's Grad-CAM.
        compare_model(model, model_cfg, model_name, args.compare, test_set, class_names,
                      list(examples["correct"][:3]) + list(examples["wrong"][:3]), out_dir, device)

    with start_run(model_cfg, "analyze_" + model_name):
        log_metrics({"test/accuracy": results["accuracy"], "test/macro_f1": results["macro_f1"]})
        log_folder(out_dir)
    print(f"Saved confused_pairs.json and Grad-CAM figures to {out_dir}")


def compare_model(model, model_cfg, model_name, other_path, test_set, class_names, indices, out_dir, device):
    """Grad-CAM of two models on the same test images, side by side."""
    other, other_cfg = load_checkpoint(other_path, device)
    other_name = file_stem(other_path)
    # The two models may normalize differently (CUB vs ImageNet), so each gets its own transforms.
    other_set = CUBDataset(test_set.hf_data, test_set.indices, get_eval_transforms(other_cfg))

    gradcam, other_gradcam = GradCAM(model), GradCAM(other)
    panels = []
    for i in indices:
        image, label = test_set[i]
        other_image, _ = other_set[i]
        cam, pred, probs = gradcam(image.to(device))
        other_cam, other_pred, other_probs = other_gradcam(other_image.to(device))

        shown = denormalize(image, *get_mean_std(model_cfg))
        other_shown = denormalize(other_image, *get_mean_std(other_cfg))
        panels += [
            (shown.permute(1, 2, 0).numpy(), f"true: {class_names[label]}"),
            (overlay(shown, cam), f"{model_name}:\n{class_names[pred]} ({probs[pred]:.0%})"),
            (overlay(other_shown, other_cam), f"{other_name}:\n{class_names[other_pred]} ({other_probs[other_pred]:.0%})"),
        ]
    gradcam.remove()
    other_gradcam.remove()
    save_image_grid(panels, os.path.join(out_dir, f"compare_{other_name}.png"), columns=3)


def load_class_names(cfg):
    """The 200 species names; position i is the name of label i."""
    with open(cfg["data"]["class_names_path"], encoding="utf-8") as f:
        return json.load(f)


def file_stem(path):
    """'outputs/checkpoints/best.pt' -> 'best'"""
    return os.path.splitext(os.path.basename(path))[0]


def main():
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="Plumage: CUB-200 bird project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eda = subparsers.add_parser("eda", help="data analysis, saves figures to outputs/figures/eda/")
    train = subparsers.add_parser("train", help="train a model")
    tune = subparsers.add_parser("tune", help="hyperparameter search")
    predict = subparsers.add_parser("predict", help="predict the species of one image")
    analyze = subparsers.add_parser("analyze", help="test-set error analysis with Grad-CAM figures")

    # Arguments every command has
    for sub in [eda, train, tune, predict, analyze]:
        sub.add_argument("--task", default="classification",
                         choices=["classification", "colorization", "captioning"])
        sub.add_argument("--config", default="configs/base.yaml", help="path to a YAML config")

    tune.add_argument("--method", choices=["grid", "random"], default="grid")
    predict.add_argument("--image", required=True, help="path to an image")
    predict.add_argument("--checkpoint", required=True, help="path to a saved model (.pt)")
    predict.add_argument("--gradcam", action="store_true", help="also save a Grad-CAM heatmap")
    analyze.add_argument("--checkpoint", required=True, help="path to a saved model (.pt)")
    analyze.add_argument("--compare", help="a second checkpoint to compare Grad-CAM with, on the same images")
    analyze.add_argument("--n-examples", type=int, default=6, help="how many correct and how many wrong images to show")

    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    commands = {"eda": eda_command, "train": train_command, "tune": tune_command,
                "predict": predict_command, "analyze": analyze_command}
    commands[args.command](args, cfg)


if __name__ == "__main__":
    main()
