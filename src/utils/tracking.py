"""Thin experiment-tracker wrapper. Swap the backend here, not in the trainer.

Usage:
    tracker = get_tracker(cfg)
    with tracker.run("scratch-baseline", tags={"task": "classification"}, config=cfg):
        tracker.log_metrics({"train/loss": 1.2, "val/acc": 0.3}, step=epoch)
        tracker.log_artifact("outputs/figures/curves.png")

Runs are logged locally with MLflow (``mlflow.db`` in the working directory).
Browse them with ``mlflow ui --backend-store-uri sqlite:///mlflow.db``.
"""

from __future__ import annotations

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
    def __init__(self, experiment: str) -> None:
        import mlflow

        self.mlflow = mlflow
        # Set an explicit relative URI: mlflow's own default resolves "mlflow.db" to an
        # absolute, URL-quoted path, which breaks sqlite directory creation on Windows
        # when the path contains a space (e.g. "C:/Users/Nouran Mokhtar/...").
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
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
        return f"{self.mlflow.get_tracking_uri()} (run {self._run.info.run_id})"


def get_tracker(cfg: dict) -> Tracker:
    t = cfg.get("tracking", {})
    backend = t.get("backend", "mlflow")
    if backend == "mlflow":
        return MLflowTracker(t.get("experiment", "plumage"))
    if backend == "none":
        return Tracker()
    raise ValueError(f"unknown tracking backend {backend!r}")
