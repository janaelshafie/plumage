"""Quick checks that don't need the dataset. Run with:  python -m pytest"""

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.data.splits import make_val_split
from src.data.text import clean_caption, clean_captions
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.evaluation.metrics import evaluate
from src.models.classifier import build_model, load_checkpoint
from src.utils.config import load_config
from src.visualization.gradcam import GradCAM


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


def test_freeze_backbone_trains_only_last_layer():
    cfg = load_config("configs/cls_scratch.yaml")
    cfg["model"]["freeze_backbone"] = True
    model = build_model(cfg)
    model.train()
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.fc.parameters())
    assert not model.encoder.training  # BatchNorm stays frozen too


def test_load_checkpoint_rebuilds_the_same_model(tmp_path):
    cfg = load_config("configs/cls_scratch.yaml")
    model = build_model(cfg).eval()
    path = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg}, path)  # same format as train_model()

    loaded, loaded_cfg = load_checkpoint(path, torch.device("cpu"))
    x = torch.randn(1, 3, 64, 64)
    assert torch.allclose(loaded(x), model(x))     # same weights -> same outputs
    assert loaded_cfg["data"]["normalization"] == "cub"


def test_gradcam_heatmap_shape_and_range():
    cfg = load_config("configs/cls_scratch.yaml")
    # Also with a frozen encoder, where no gradient would reach the layer without requires_grad on the input.
    for freeze in [False, True]:
        cfg["model"]["freeze_backbone"] = freeze
        gradcam = GradCAM(build_model(cfg))
        cam, pred, probs = gradcam(torch.randn(3, 64, 64))
        gradcam.remove()
        assert cam.shape == (64, 64)               # same size as the image
        assert cam.min() >= 0 and cam.max() <= 1
        assert 0 <= pred < 200 and probs.shape == (200,)


def test_evaluate_on_perfect_model():
    # A fake "model" that already outputs one-hot scores, so every prediction is right.
    scores = torch.eye(6)
    labels = torch.arange(6)
    loader = DataLoader(TensorDataset(scores, labels), batch_size=2)
    results = evaluate(torch.nn.Identity(), loader, torch.device("cpu"), num_classes=6)
    assert results["accuracy"] == 1.0
    assert results["macro_f1"] == 1.0
