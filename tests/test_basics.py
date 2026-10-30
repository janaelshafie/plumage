"""Quick checks that don't need the dataset. Run with:  python -m pytest"""

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.data.splits import make_val_split
from src.data.text import clean_caption, clean_captions
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.evaluation.metrics import evaluate
from src.models.classifier import build_model
from src.utils.config import load_config


def test_config_inherits_from_base():
    cfg = load_config("configs/cls_scratch.yaml")
    assert cfg["data"]["normalization"] == "cub"   # changed in cls_scratch.yaml
    assert cfg["data"]["num_classes"] == 200       # comes from base.yaml


def test_val_split_takes_10_percent_of_each_class():
    labels = [c for c in range(5) for _ in range(30)]  # 5 classes, 30 images each
    val = make_val_split(labels, 0.1, seed=0)
    for c in range(5):
        assert sum(1 for i in val if labels[i] == c) == 3
    assert val == make_val_split(labels, 0.1, seed=0)  # same seed -> same split


def test_clean_caption():
    assert clean_caption("This bird is RED .") == "this bird is red"
    assert clean_caption("small bird,with a yellow belly..") == "small bird, with a yellow belly"
    assert clean_caption("this��bird��has black wings") == "this bird has black wings"
    assert clean_caption("black & white, red/orange beak") == "black and white, red orange beak"
    assert clean_captions("A Bird.\n!!!\n\nAnother bird\n") == ["a bird", "another bird"]


def test_transforms_give_224_images():
    cfg = load_config("configs/base.yaml")
    image = Image.new("RGB", (500, 333))
    assert get_train_transforms(cfg)(image).shape == (3, 224, 224)
    assert get_eval_transforms(cfg)(image).shape == (3, 224, 224)


def test_model_output_shape():
    cfg = load_config("configs/cls_scratch.yaml")
    model = build_model(cfg)
    outputs = model(torch.randn(2, 3, 64, 64))
    assert outputs.shape == (2, 200)


def test_evaluate_on_perfect_model():
    # A fake "model" that already outputs one-hot scores, so every prediction is right.
    scores = torch.eye(6)
    labels = torch.arange(6)
    loader = DataLoader(TensorDataset(scores, labels), batch_size=2)
    results = evaluate(torch.nn.Identity(), loader, torch.device("cpu"), num_classes=6)
    assert results["accuracy"] == 1.0
    assert results["macro_f1"] == 1.0
