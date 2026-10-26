# Plumage — From Pixels to Language

CSE 485 / 488 course project on **CUB-200-2011** (200 bird species, ~11.8k images, 10 captions per image).
Milestone 1 covers EDA, species classification and encoder training. See [MILESTONE1_PLAN.md](MILESTONE1_PLAN.md) for who does what.

## Environment setup

Python 3.10+ is required. On Kaggle/Colab only steps 1 and 3 are needed; they already have CUDA PyTorch.

```bash
# 1. Clone
git clone <repo-url> plumage
cd plumage

# 2. Virtual environment (local machines only)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt
# Local CPU-only machine? Install the smaller CPU wheels first:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Check the setup

```bash
python -m src.cli --help                           # CLI is wired up
python -m pytest                                   # fast unit tests, no download
python scripts/smoke_test.py --config configs/base.yaml
```

The smoke test downloads the dataset on its first run (~1.2 GB into `data/hf_cache/`), pulls one batch from each
of train/val/test, checks shapes `[B, 3, 224, 224]` and labels in 0–199, and logs a run to the local MLflow tracker.

### Experiment tracker

Every run is logged locally with MLflow through [src/utils/tracking.py](src/utils/tracking.py) to a `mlflow.db` file
(gitignored). Browse runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db`. Set `tracking.backend: none` in a
config to turn logging off.

## Data

| Split | Source | Images | Use |
|-------|--------|--------|-----|
| train | `Multimodal-Fatima/CUB_train` minus val and 5 duplicates | 5,389 | training |
| val   | stratified 10% per class of `CUB_train` | 600 | model selection and tuning |
| test  | `Multimodal-Fatima/CUB_test` | 5,794 | final evaluation only, never tuning |

The validation indices are fixed in [configs/splits/val_indices.json](configs/splits/val_indices.json) (seed 42) so everyone
tunes on the same images. Don't regenerate it without telling the team.

```python
from src.utils.config import load_config
from src.data.dataloaders import build_dataloaders

cfg = load_config("configs/base.yaml")
train_loader, val_loader, test_loader = build_dataloaders(cfg, train_tf, eval_tf)
```

`train_tf` and `eval_tf` are any callables mapping an RGB PIL image to a tensor. If omitted, `get_train_transforms(cfg)` and
`get_eval_transforms(cfg)` from [src/data/transforms.py](src/data/transforms.py) are used (augmentation settings under `data.augment`).

- **Normalization:** `data.normalization: imagenet` (default, for pretrained backbones) or `cub` (CUB train stats, for from-scratch models).
- **Duplicates:** `data.exclude_train_files` lists 5 train images that duplicate a test image or another train image. They are dropped from
  training only; val and test are unchanged.
- **Captions:** `clean_captions(description)` in [src/data/text.py](src/data/text.py) splits and cleans the 10 captions of an image.

`python -m src.cli eda --config configs/base.yaml` regenerates every EDA figure, `summary.json` and CSV in `outputs/figures/eda/` (about 3 minutes).

## Project layout

```
configs/            YAML configs (task configs inherit base.yaml via `base: base.yaml`)
configs/splits/     committed validation split
docs/               report notes and benchmark notes
notebooks/          exploration only; anything kept moves into src/
scripts/            one-off checks (smoke test)
src/cli.py          entry point: eda | train | tune | predict
src/data/           dataset, splits, dataloaders, transforms, caption cleaning
src/models/         encoders and classifier registry
src/training/       trainer and hyperparameter search
src/evaluation/     metrics and error analysis
src/visualization/  EDA plots, curves, Grad-CAM
src/utils/          config, seeding, device, tracker
tests/              unit tests
outputs/            figures and checkpoints (gitignored; checkpoints go to the team Drive)
```

## Running

Commands are filled in as each step lands:

```bash
python -m src.cli eda     --config configs/base.yaml
python -m src.cli train   --task classification --config configs/cls_scratch.yaml
python -m src.cli tune    --task classification --method grid --config configs/tune_resnet50.yaml
python -m src.cli predict --task classification --image path/to/bird.jpg --checkpoint path/to/best.pt --gradcam
```
