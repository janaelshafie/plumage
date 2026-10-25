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
of train/val/test, checks shapes `[B, 3, 224, 224]` and labels in 0–199, and logs a run to the tracker.

### Experiment tracker

Every run is logged through [src/utils/tracking.py](src/utils/tracking.py), so the backend can be swapped in that one file
(`tracking.backend` in the config: `mlflow`, `wandb` or `none`).

The shared tracker is MLflow hosted on DagsHub. Set your credentials as environment variables before running and **never commit them**:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<owner>/plumage.mlflow
export MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
export MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

On Colab/Kaggle, store them as notebook secrets and set `os.environ[...]` in the first cell.
Without these variables, runs go to a local `mlflow.db` file (browse with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).

## Data

| Split | Source | Images | Use |
|-------|--------|--------|-----|
| train | `Multimodal-Fatima/CUB_train` minus val | 5,394 | training |
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

`train_tf` and `eval_tf` are any callables mapping an RGB PIL image to a tensor. If omitted, a resize + center-crop placeholder is used.

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
