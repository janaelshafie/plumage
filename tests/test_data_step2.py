"""Caption cleaning and transforms; no dataset download."""

import pytest
import torch
from PIL import Image

from src.data.text import clean_caption, clean_captions, split_description, tokenize
from src.data.transforms import denormalize, get_eval_transforms, get_norm_stats, get_train_transforms
from src.utils.config import load_config


def test_split_description_drops_blank_lines():
    assert split_description("a red bird\n\n  a blue bird \n") == ["a red bird", "a blue bird"]


@pytest.mark.parametrize("raw, clean", [
    ("This bird is RED .", "this bird is red"),
    ("It has a short,  pointy beak!", "it has a short, pointy beak"),
    ("small bird,with a “yellow” belly..", "small bird, with a yellow belly"),
    ("the bird’s wings are grey", "the bird's wings are grey"),
    ("  ...  ", ""),
])
def test_clean_caption(raw, clean):
    assert clean_caption(raw) == clean


def test_clean_captions_drops_empty():
    assert clean_captions("A Bird.\n!!!\nAnother bird\n") == ["a bird", "another bird"]


def test_tokenize():
    assert tokenize("the bird's wings, grey and 2 tone") == ["the", "bird's", "wings", "grey", "and", "2", "tone"]


@pytest.fixture
def cfg():
    return load_config("configs/base.yaml")


def test_norm_stats_follow_config(cfg):
    assert get_norm_stats(cfg) == (cfg["data"]["imagenet_mean"], cfg["data"]["imagenet_std"])
    cfg["data"]["normalization"] = "cub"
    assert get_norm_stats(cfg) == (cfg["data"]["cub_mean"], cfg["data"]["cub_std"])
    cfg["data"]["normalization"] = "other"
    with pytest.raises(ValueError):
        get_norm_stats(cfg)


def test_transforms_output_shape(cfg):
    size = cfg["data"]["image_size"]
    image = Image.new("RGB", (500, 333), (120, 80, 40))
    for tf in (get_train_transforms(cfg), get_eval_transforms(cfg)):
        assert tf(image).shape == (3, size, size)


def test_eval_transform_is_deterministic(cfg):
    image = Image.effect_noise((300, 200), 50).convert("RGB")
    tf = get_eval_transforms(cfg)
    assert torch.equal(tf(image), tf(image))


def test_denormalize_inverts_normalize(cfg):
    mean, std = get_norm_stats(cfg)
    x = torch.rand(2, 3, 8, 8)
    normed = (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
    assert torch.allclose(denormalize(normed, mean, std), x, atol=1e-6)
