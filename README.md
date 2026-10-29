# Plumage

CSE 485 / 488 project on the **CUB-200-2011** bird dataset (200 species, ~11.8k images, 10 captions per image).
Milestone 1: data analysis (EDA) and bird species classification. The team plan is in [MILESTONE1_PLAN.md](MILESTONE1_PLAN.md).

## Setup

```bash
git clone <repo-url> plumage
cd plumage
pip install -r requirements.txt
```

On your own machine you can make a virtual environment first (`python -m venv .venv`, then activate it).
On Colab/Kaggle just run the `pip install`.

The dataset (~1.2 GB) downloads automatically into `data/hf_cache/` the first time you run a command.

## How to run

```bash
# Data analysis: figures go to outputs/figures/eda/
python -m src.cli eda --config configs/base.yaml

# Train the CNN from scratch: checkpoint goes to outputs/checkpoints/, curves to outputs/figures/curves/
python -m src.cli train --task classification --config configs/cls_scratch.yaml

# Quick tests
python -m pytest
```

All runs are saved with MLflow. To see them:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Data split

| Split | Images | Used for |
|-------|--------|----------|
| train | 5,389 | training |
| val   | 600 (3 per class, taken from the train set) | choosing models and hyperparameters |
| test  | 5,794 | final evaluation only |

The val images are saved in `configs/splits/val_indices.json` so everyone uses the same ones.

## Folders

```
configs/      settings for each run (YAML)
notebooks/    experiments only
docs/         report notes
src/
  cli.py          all commands start here
  data/           dataset, split, transforms, caption cleaning
  models/         CNN encoder and classifier
  training/       training loop
  evaluation/     metrics
  visualization/  EDA plots, training curves
  utils/          config, seed, MLflow
tests/        quick tests
outputs/      figures and checkpoints (not on GitHub)
```
