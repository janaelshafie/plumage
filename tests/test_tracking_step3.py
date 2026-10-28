"""Regression test for the Windows sqlite-path bug: mlflow's own default tracking
URI resolves to an absolute, URL-quoted path and fails to create `mlflow.db`'s
parent directory when that path contains a space. No dataset download.
"""

from src.utils.tracking import MLflowTracker


def test_mlflow_tracker_uses_relative_sqlite_uri(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = MLflowTracker("test-experiment")
    assert tracker.mlflow.get_tracking_uri() == "sqlite:///mlflow.db"
    assert (tmp_path / "mlflow.db").exists()
