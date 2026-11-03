"""Hyperparameter search over the learning rate and weight decay.

Settings come from the `tune:` section of the config: epochs, n_trials, lr, weight_decay.

grid_search:   tries every combination of the listed values.
random_search: picks random values between the smallest and largest listed value.

Each trial is a short train_model() run and its own MLflow run. Both functions
return a list with one dict per trial: lr, weight_decay, best_val_acc, time_sec.
"""

import copy
import itertools
import math
import random
import time

from src.data.dataloaders import build_dataloaders
from src.models.classifier import build_model
from src.training.trainer import train_model
from src.utils.config import set_seed
from src.utils.tracking import log_metrics, start_run


def run_trial(cfg, lr, weight_decay, train_loader, val_loader, run_name):
    """Train for `tune.epochs` epochs with one (lr, weight_decay) pair."""
    trial_cfg = copy.deepcopy(cfg)
    trial_cfg["train"]["lr"] = lr
    trial_cfg["train"]["weight_decay"] = weight_decay
    trial_cfg["train"]["epochs"] = cfg["tune"]["epochs"]

    # Same seed for every trial, so they start from the same weights and only lr and weight decay differ.
    set_seed(cfg["seed"])
    model = build_model(trial_cfg)

    print(f"\n{run_name}: lr {lr:.2e}, weight decay {weight_decay:.2e}")
    start = time.time()
    with start_run(trial_cfg, run_name):
        history, _ = train_model(model, train_loader, val_loader, trial_cfg, run_name)
        best_val_acc = max(history["val_acc"])
        log_metrics({"tune/best_val_acc": best_val_acc})

    return {"lr": lr, "weight_decay": weight_decay, "best_val_acc": best_val_acc,
            "time_sec": time.time() - start}


def grid_search(cfg):
    """Try every (lr, weight_decay) combination from the config."""
    train_loader, val_loader, _ = build_dataloaders(cfg)  # test is never used for tuning
    pairs = list(itertools.product(cfg["tune"]["lr"], cfg["tune"]["weight_decay"]))
    return [run_trial(cfg, lr, wd, train_loader, val_loader, f"grid-{i + 1}")
            for i, (lr, wd) in enumerate(pairs)]


def random_search(cfg):
    """Try `tune.n_trials` random (lr, weight_decay) pairs from the config ranges."""
    train_loader, val_loader, _ = build_dataloaders(cfg)
    # Its own random generator, so set_seed() inside each trial doesn't repeat the same picks.
    rng = random.Random(cfg["seed"])
    trials = []
    for i in range(cfg["tune"]["n_trials"]):
        lr = log_uniform(rng, cfg["tune"]["lr"])
        wd = log_uniform(rng, cfg["tune"]["weight_decay"])
        trials.append(run_trial(cfg, lr, wd, train_loader, val_loader, f"random-{i + 1}"))
    return trials


def log_uniform(rng, values):
    """A random number between min(values) and max(values) on a log scale.

    On a log scale 1e-4 -> 3e-4 is as likely as 3e-4 -> 1e-3, which suits values
    like the learning rate that matter by their order of magnitude.
    """
    low, high = math.log10(min(values)), math.log10(max(values))
    return 10 ** rng.uniform(low, high)
