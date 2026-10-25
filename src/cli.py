"""Single entry point: ``python -m src.cli <command> --task <task> --config <yaml>``.

Each command is a stub until its step lands:
    eda      -> Step 2 (Habiba)
    train    -> Step 3 (Nouran)
    tune     -> Step 4 (Mohamed)
    predict  -> Step 5 (Abdallah)
"""

from __future__ import annotations

import argparse
import sys

from src.utils.config import load_config, set_seed

TASKS = ["classification", "colorization", "captioning"]


def _not_implemented(name: str, args: argparse.Namespace) -> int:
    print(f"`{name}` for task '{args.task}' is not implemented yet.")
    return 1


def cmd_eda(args: argparse.Namespace, cfg: dict) -> int:
    return _not_implemented("eda", args)


def cmd_train(args: argparse.Namespace, cfg: dict) -> int:
    return _not_implemented("train", args)


def cmd_tune(args: argparse.Namespace, cfg: dict) -> int:
    return _not_implemented("tune", args)


def cmd_predict(args: argparse.Namespace, cfg: dict) -> int:
    return _not_implemented("predict", args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="Plumage: CUB-200 pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_: str, fn, default_task: str = "classification") -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--task", choices=TASKS, default=default_task)
        p.add_argument("--config", default="configs/base.yaml", help="path to a YAML config")
        p.add_argument("--seed", type=int, default=None, help="override the config seed")
        p.set_defaults(func=fn)
        return p

    add("eda", "exploratory data analysis; writes figures to outputs/figures/eda/", cmd_eda)
    add("train", "train a model", cmd_train)

    tune = add("tune", "hyperparameter search", cmd_tune)
    tune.add_argument("--method", choices=["grid", "random"], default="grid")
    tune.add_argument("--n-trials", type=int, default=None, help="random search trials")

    predict = add("predict", "run inference on one image", cmd_predict)
    predict.add_argument("--image", required=True, help="path to an image file")
    predict.add_argument("--checkpoint", required=True, help="path to a model checkpoint")
    predict.add_argument("--gradcam", action="store_true", help="also save a Grad-CAM heatmap")
    predict.add_argument("--top-k", type=int, default=5)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg.get("seed", 42))
    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
