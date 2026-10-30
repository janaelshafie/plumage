"""Experiment tracking with MLflow.

Every run is saved to a local file, mlflow.db. To look at the runs:
    mlflow ui --backend-store-uri sqlite:///mlflow.db

All MLflow calls live in this file, so changing to another tracker
(e.g. Weights & Biases) only means changing these functions.
"""

import mlflow


def start_run(cfg, run_name):
    """Start a new run and save every config value as a hyperparameter.

    Use it with `with`, so the run is closed at the end:
        with start_run(cfg, "my-run"):
            ...
    """
    # We give the database path ourselves. MLflow's default path breaks on
    # Windows when the user folder has a space in its name.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(cfg["tracking"]["experiment"])
    run = mlflow.start_run(run_name=run_name)
    mlflow.log_params(flatten(cfg))
    return run


def flatten(cfg, prefix=""):
    """Turn {"data": {"batch_size": 32}} into {"data.batch_size": 32}."""
    flat = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            flat.update(flatten(value, prefix + key + "."))
        else:
            flat[prefix + key] = value
    return flat


def log_metrics(metrics, step=None):
    """Log a dict of numbers, e.g. {"val/acc": 0.4}. `step` is the epoch number."""
    mlflow.log_metrics(metrics, step=step)


def log_file(path):
    """Attach a file (figure, checkpoint, ...) to the current run."""
    mlflow.log_artifact(path)


def log_folder(path):
    """Attach every file in a folder to the current run."""
    mlflow.log_artifacts(path)
