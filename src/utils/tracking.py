"""Thin experiment-tracker wrapper. Swap the backend here, not in the trainer.

Usage:
    tracker = get_tracker(cfg)
    with tracker.run("scratch-baseline", tags={"task": "classification"}, config=cfg):
        tracker.log_metrics({"train/loss": 1.2, "val/acc": 0.3}, step=epoch)
        tracker.log_artifact("outputs/figures/curves.png")

Shared server (DagsHub-hosted MLflow): set these environment variables before running
(never commit them):
    MLFLOW_TRACKING_URI=https://dagshub.com/<owner>/plumage.mlflow
    MLFLOW_TRACKING_USERNAME=<your dagshub username>
    MLFLOW_TRACKING_PASSWORD=<your dagshub token>
Without them, runs go to MLflow's local default (``mlflow.db`` in the working directory).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Mapping


def flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """``{"data": {"batch_size": 32}}`` -> ``{"data.batch_size": 32}``."""
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


class Tracker:
    """No-op base tracker; also used when ``tracking.backend: none``."""

    @contextmanager
    def run(self, name: str, tags: dict | None = None, config: dict | None = None) -> Iterator["Tracker"]:
        self.start(name, tags, config)
        try:
            yield self
        except BaseException:
            self.end(status="FAILED")
            raise
        self.end()

    def start(self, name: str, tags: dict | None = None, config: dict | None = None) -> None: ...
    def log_params(self, params: Mapping[str, Any]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None: ...
    def log_artifact(self, path: str) -> None: ...
    def set_tags(self, tags: Mapping[str, Any]) -> None: ...
    def end(self, status: str = "FINISHED") -> None: ...

    @property
    def run_url(self) -> str | None:
        return None


class MLflowTracker(Tracker):
    def __init__(self, experiment: str, tracking_uri: str | None = None) -> None:
        import mlflow

        self.mlflow = mlflow
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        self._run = None

    def start(self, name, tags=None, config=None):
        self._run = self.mlflow.start_run(run_name=name, tags={k: str(v) for k, v in (tags or {}).items()})
        if config:
            self.log_params(flatten(config))

    def log_params(self, params):
        # MLflow caps param values at 500 chars.
        self.mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    def log_metrics(self, metrics, step=None):
        self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path):
        self.mlflow.log_artifact(path)

    def set_tags(self, tags):
        self.mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def end(self, status="FINISHED"):
        if self._run is not None:
            self.mlflow.end_run(status=status)
            self._run = None

    @property
    def run_url(self):
        if self._run is None:
            return None
        uri = self.mlflow.get_tracking_uri()
        info = self._run.info
        if uri.startswith("http"):
            return f"{uri.rstrip('/')}/#/experiments/{info.experiment_id}/runs/{info.run_id}"
        return f"{uri} (run {info.run_id})"


class WandbTracker(Tracker):
    def __init__(self, project: str, entity: str | None = None) -> None:
        import wandb

        self.wandb = wandb
        self.project, self.entity = project, entity
        self._run = None

    def start(self, name, tags=None, config=None):
        tags = tags or {}
        self._run = self.wandb.init(
            project=self.project, entity=self.entity, name=name, config=config,
            tags=[f"{k}:{v}" for k, v in tags.items()], reinit=True,
        )

    def log_params(self, params):
        self._run.config.update(dict(params), allow_val_change=True)

    def log_metrics(self, metrics, step=None):
        self._run.log(dict(metrics), step=step)

    def log_artifact(self, path):
        self._run.save(path, policy="now")

    def set_tags(self, tags):
        self._run.tags = tuple(self._run.tags) + tuple(f"{k}:{v}" for k, v in tags.items())

    def end(self, status="FINISHED"):
        if self._run is not None:
            self._run.finish(exit_code=0 if status == "FINISHED" else 1)
            self._run = None

    @property
    def run_url(self):
        return self._run.url if self._run is not None else None


def get_tracker(cfg: dict) -> Tracker:
    t = cfg.get("tracking", {})
    backend = t.get("backend", "mlflow")
    if backend == "mlflow":
        return MLflowTracker(t.get("experiment", "plumage"), t.get("tracking_uri"))
    if backend == "wandb":
        return WandbTracker(t.get("experiment", "plumage"), t.get("wandb_entity"))
    if backend == "none":
        return Tracker()
    raise ValueError(f"unknown tracking backend {backend!r}")
