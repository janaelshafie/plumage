"""Fast tests that need no dataset download."""

import pytest

from src.cli import build_parser
from src.data.splits import load_split, save_split, stratified_val_indices
from src.utils.config import load_config
from src.utils.tracking import flatten


def test_stratified_split_is_per_class_and_deterministic():
    labels = [c for c in range(5) for _ in range(30)]  # 5 classes x 30 images
    val = stratified_val_indices(labels, 0.1, seed=0)
    assert val == stratified_val_indices(labels, 0.1, seed=0)
    assert val != stratified_val_indices(labels, 0.1, seed=1)
    per_class = [sum(1 for i in val if labels[i] == c) for c in range(5)]
    assert per_class == [3] * 5
    assert len(set(val)) == len(val)


def test_split_holds_out_at_least_one_per_class():
    labels = [0, 0, 0, 1, 1, 1]
    val = stratified_val_indices(labels, 0.1, seed=0)
    assert sorted(labels[i] for i in val) == [0, 1]


def test_split_roundtrip(tmp_path):
    path = tmp_path / "val.json"
    save_split(path, [1, 4], n_train_total=6, seed=0)
    train, val = load_split(path, 6)
    assert val == [1, 4]
    assert train == [0, 2, 3, 5]
    with pytest.raises(ValueError):
        load_split(path, 7)


def test_config_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 1\ndata:\n  batch_size: 32\n  image_size: 224\n")
    (tmp_path / "child.yaml").write_text("base: base.yaml\ndata:\n  batch_size: 64\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg == {"seed": 1, "data": {"batch_size": 64, "image_size": 224}}


def test_base_config_loads():
    cfg = load_config("configs/base.yaml")
    assert cfg["data"]["num_classes"] == 200


def test_flatten():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


@pytest.mark.parametrize("cmd", ["eda", "train", "tune"])
def test_cli_parses(cmd):
    args = build_parser().parse_args([cmd, "--task", "classification", "--config", "configs/base.yaml"])
    assert args.command == cmd
