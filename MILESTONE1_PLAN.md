# Milestone 1 Implementation Plan

**CSE 485 / 488 – From Pixels to Language**
**Scope:** EDA, classification and encoder training
**Due:** Sunday 1 November 2026 (LMS)

Five people, five steps, run in order. Each person finishes their step, commits it from their own GitHub account, updates the **Progress tracker** below in their last commit, and hands off to the next person.

**Order:** Jana → Habiba → Nouran → Mohamed → Abdallah

---

## Progress tracker

> Update this table in your **final commit** of your step. Change your status to `✅ Done`, fill in the date, and set the next person to `🔵 In progress`.

| # | Owner | Step | Planned | Status | Finished on |
|---|-------|------|---------|--------|-------------|
| 1 | Jana | Foundation and data pipeline | Mon 26 Oct | 🔵 In progress | |
| 2 | Habiba | EDA, cleaning, preprocessing, augmentation | Tue 27 Oct | ⚪ Waiting | |
| 3 | Nouran | From-scratch CNN, trainer, metrics | Wed 28 Oct | ⚪ Waiting | |
| 4 | Mohamed | Transfer learning and hyperparameter tuning | Thu 29 – Fri 30 Oct (2 pm) | ⚪ Waiting | |
| 5 | Abdallah | Grad-CAM, inference, benchmarking, submission | Fri 30 – Sat 31 Oct | ⚪ Waiting | |
| – | Everyone | Final review and LMS submission | Sun 1 Nov | ⚪ Waiting | |

**Baton is currently with:** Jana

---

## Rules for every step

- **Pull first.** `git pull` on `main` before you start. Work on a branch named `stepN-yourname` (e.g. `step2-habiba`).
- **Commit from your own account,** small and often. Several meaningful commits per person, never one big upload at the end.
- **Open a PR; the previous person reviews it.** Reviewing is how everyone learns the whole project for the individual discussion (35% of the grade).
- **No final logic in notebooks.** Explore in `notebooks/`, but anything kept moves into `src/` and runs through the CLI.
- **Log every run** to the shared experiment tracker. Never commit checkpoints or data; checkpoints go to the team Google Drive folder.
- **Write report notes** in `docs/report_notes.md` under your step: what you did, why, and what failed.
- **Use the "Before your turn" time.** Don't commit early, but prepare so your step goes fast.

---

## Step 1 — Jana: Foundation and data pipeline

**When:** Mon 26 Oct
**Goal:** Build the skeleton everyone else builds on: repo, project layout, configs, dataset over the Hugging Face splits, a fixed validation split, the shared tracker, and the CLI entry point.

### Checklist
- [ ] Create the GitHub repo and add Habiba, Nouran, Mohamed and Abdallah as collaborators. Add a `.gitignore` for `data/`, `outputs/`, `mlruns/`, `*.pt`, `*.pth`, `.ipynb_checkpoints`.
- [ ] Create the folder layout from section 5.2 of the brief: `configs/`, `notebooks/`, `docs/`, `src/{data,models,training,evaluation,visualization,utils}/` with `__init__.py`, `outputs/`.
- [ ] Write `requirements.txt`: torch, torchvision, datasets, mlflow (or wandb), matplotlib, seaborn, scikit-learn, pandas, pyyaml, tqdm, pillow, imagehash.
- [ ] Config and reproducibility helpers in `src/utils/`: `load_config(path)`, `set_seed(seed)` (python/numpy/torch), `get_device()`.
- [ ] `CUBDataset` in `src/data/cub_dataset.py`: loads `Multimodal-Fatima/CUB_train` and `CUB_test`, converts every image to RGB, returns `(image_tensor, label)`, takes a `transform` argument so Habiba can plug in augmentation.
- [ ] Stratified validation split: hold out ~10% of the train set per class with a fixed seed. Save indices to `configs/splits/val_indices.json` and commit it. **The test set is never used for tuning.**
- [ ] `build_dataloaders(cfg, train_tf, eval_tf)` returning train / val / test loaders.
- [ ] Shared experiment tracker everyone logs to (W&B team project or DagsHub-hosted MLflow), behind a small wrapper in `src/utils/tracking.py` so the backend can be swapped in one file.
- [ ] CLI skeleton in `src/cli.py` with subcommands `train`, `predict`, `eda`, `tune`, each taking `--task` and `--config` (can print "not implemented yet" for now).
- [ ] README: environment setup section.
- [ ] Update the progress tracker and hand off to Habiba.

### Files
`src/utils/config.py`, `src/utils/tracking.py`, `src/data/cub_dataset.py`, `src/data/dataloaders.py`, `configs/base.yaml`, `configs/splits/val_indices.json`, `src/cli.py`, `requirements.txt`, `README.md`

### Example commits
```
chore: project skeleton and gitignore
feat(utils): yaml config loader and set_seed
feat(data): CUB dataset wrapper over HF splits
feat(data): stratified validation split
feat(utils): experiment tracker wrapper
feat(cli): train/predict/eda/tune skeleton
```

### Ready to hand off when
- `python -m src.cli --help` runs
- A smoke test pulls one batch with shape `[B, 3, H, W]` and labels in 0–199
- A test run appears in the shared tracker

---

## Step 2 — Habiba: EDA, cleaning, preprocessing and augmentation

**When:** Tue 27 Oct
**Before your turn:** Load the dataset in a notebook and browse it; read section 6.1.1 of the brief.
**Goal:** Understand the data and turn findings into code: image and caption plots, caption cleaning, normalization statistics, and a justified augmentation strategy.

### Checklist
- [ ] Class distribution plot for train / val / test, plus imbalance ratio (max/min).
- [ ] Image statistics: width, height, aspect ratio, image modes (look for grayscale or RGBA images), file sizes.
- [ ] Duplicate and corruption check: perceptual hashes (`imagehash`) within train and across train/test; list anything that fails to decode.
- [ ] Channel mean and std on the train split; compare with ImageNet stats, write both into `configs/base.yaml`, note which is used for which model.
- [ ] Sample grid of images with species names.
- [ ] Caption analysis: split `description` on `\n`, check each image has 10 captions, plot caption length in words, vocabulary size, top words, rare words (freq < 5) as typo candidates, odd characters.
- [ ] Caption cleaning in `src/data/text.py`: `split_description()` and `clean_caption()` (strip, lowercase, normalize punctuation/whitespace, drop empty lines). Milestone 2 captioning reuses this.
- [ ] Augmentation in `src/data/transforms.py`:
  - Train: `RandomResizedCrop(224)`, horizontal flip, small rotation, mild `ColorJitter`
  - Eval: `Resize(256)` + `CenterCrop(224)`
  - No vertical flips; keep hue jitter small because plumage color identifies species.
- [ ] Augmentation preview figure (one image, 8 random augmentations).
- [ ] `eda` CLI command: `python -m src.cli eda --config configs/base.yaml` writes all figures to `outputs/figures/eda/`.
- [ ] Report notes, Data section: each finding, the decision it led to, and why.
- [ ] Update the progress tracker and hand off to Nouran.

### Files
`notebooks/01_eda.ipynb`, `src/visualization/eda_plots.py`, `src/data/text.py`, `src/data/transforms.py`, `docs/report_notes.md`

### Example commits
```
feat(eda): class distribution and image statistics
feat(eda): duplicate and corruption check
feat(data): caption split and cleaning
feat(data): train/eval transforms
feat(cli): eda command
docs: EDA findings and justifications
```

### Ready to hand off when
- `python -m src.cli eda` regenerates all figures
- `get_train_transforms()` and `get_eval_transforms()` import cleanly
- Mean/std are in the config

---

## Step 3 — Nouran: From-scratch CNN, training engine and metrics

**When:** Wed 28 Oct
**Before your turn:** Sketch the CNN architecture and set up a Kaggle/Colab notebook that clones the repo and installs requirements.
**Goal:** Build the from-scratch encoder, the reusable trainer every later model goes through, and the evaluation module. Run and log the baseline.

### Checklist
- [ ] Scratch encoder in `src/models/encoders.py`: ~5 blocks of `[Conv-BN-ReLU ×2, MaxPool]`, channels 32→512, global average pooling. Expose `forward_features()` (spatial map), `out_dim`, and `gradcam_layer` (last conv). Colorization in Milestone 2 reuses this encoder.
- [ ] Classifier head and model registry in `src/models/classifier.py`: `Classifier(encoder, num_classes, dropout)` and `build_model(cfg)` that picks the encoder by name, so new encoders don't touch the trainer.
- [ ] Generic `Trainer` in `src/training/trainer.py`: train/validate loops, AdamW, cosine schedule, mixed precision for the T4, early stopping, best-checkpoint saving, per-epoch logging through the tracker wrapper.
- [ ] Metrics in `src/evaluation/metrics.py`: accuracy, top-5 accuracy, macro precision/recall/F1, confusion matrix; `evaluate(model, loader)`.
- [ ] Training curves in `src/visualization/curves.py` (train vs val loss and accuracy).
- [ ] Wire up: `python -m src.cli train --task classification --config configs/cls_scratch.yaml`
- [ ] Run the baseline on a T4 for 30–50 epochs. Log it, evaluate on test once, upload best checkpoint to Drive.
- [ ] Report notes: architecture table/diagram, settings, curves, results, and why accuracy is modest (6k images, 200 fine-grained classes, no pretraining).
- [ ] Update the progress tracker and hand off to Mohamed (include the checkpoint link and run link in the PR).

### Files
`src/models/encoders.py`, `src/models/classifier.py`, `src/training/trainer.py`, `src/evaluation/metrics.py`, `src/visualization/curves.py`, `configs/cls_scratch.yaml`

### Example commits
```
feat(models): scratch CNN encoder
feat(models): classifier head and registry
feat(training): generic trainer with AMP and early stopping
feat(eval): classification metrics
feat(viz): training curves
exp: scratch baseline config
```

### Ready to hand off when
- Baseline run is in the tracker with val and test metrics
- Checkpoint link is shared
- Swapping the encoder means changing one line of config

---

## Step 4 — Mohamed: Transfer learning and hyperparameter tuning

**When:** Thu 29 – Fri 30 Oct (2 pm)
**Before your turn:** Read the trainer interface in Nouran's PR; pre-download ResNet-50 and EfficientNet-B0 weights in your Kaggle notebook.
**Goal:** Add pretrained backbones through the same registry, compare frozen vs fine-tuned, run grid and random search, and train the final selected model.

### Checklist
- [ ] Pretrained encoders in the registry: ResNet-50 (`IMAGENET1K_V2`) and a lighter one (e.g. EfficientNet-B0), with the same `forward_features` / `out_dim` / `gradcam_layer` interface.
- [ ] Two transfer strategies: frozen backbone + trained head, and full fine-tuning with a lower LR on the backbone than on the head.
- [ ] Tuning module `src/training/tuning.py`: `grid_search(space, n_epochs)` and `random_search(space, n_trials, n_epochs)`. Every trial is a tracked run tagged with its method.
- [ ] Grid search: e.g. learning rate × weight decay (3×3 = 9 trials) on fine-tuned ResNet-50, short runs (~8 epochs) to fit the T4 budget.
- [ ] Random search with the same budget: 9 trials from the same ranges (log-uniform LR).
- [ ] Compare the two methods: table + plot of best val accuracy, number of trials, total time. (Optuna optional.)
- [ ] `tune` CLI: `python -m src.cli tune --task classification --method grid --config configs/tune_resnet50.yaml`
- [ ] Retrain the best configuration at full length, evaluate on test, upload checkpoint to Drive.
- [ ] Scratch vs transfer comparison table: accuracy, macro F1, parameters, training time for every model.
- [ ] Report notes: search spaces, why those hyperparameters, results, final model choice.
- [ ] Update the progress tracker and hand off to Abdallah.

### Files
`src/models/encoders.py` (extend), `src/training/tuning.py`, `configs/cls_resnet50.yaml`, `configs/cls_effnet_b0.yaml`, `configs/tune_resnet50.yaml`

### Example commits
```
feat(models): pretrained ResNet-50 and EfficientNet encoders
feat(training): frozen vs fine-tune strategies
feat(tuning): grid and random search
feat(cli): tune command
exp: final tuned ResNet-50 config
```

### Ready to hand off when
- Best model checkpoint is on Drive with its config committed
- Grid vs random comparison table exists
- Scratch vs transfer table exists

---

## Step 5 — Abdallah: Grad-CAM, inference, benchmarking and submission

**When:** Fri 30 – Sat 31 Oct
**Before your turn:** Read the Grad-CAM paper (Selvaraju et al., 2017) and try hooks on Nouran's scratch checkpoint.
**Goal:** Explain the models with Grad-CAM, make inference work from the CLI, collect published results, and package Milestone 1 so it reproduces from a fresh clone.

### Checklist
- [ ] Grad-CAM in `src/visualization/gradcam.py`: own implementation with forward and backward hooks on `encoder.gradcam_layer`, so it works for every registered encoder.
- [ ] Grad-CAM figures: correct high-confidence predictions, wrong predictions (true vs predicted), and the same images under scratch vs transfer model side by side.
- [ ] Error analysis: most-confused species pairs from the confusion matrix, with Grad-CAM on a few.
- [ ] `predict` CLI: `python -m src.cli predict --task classification --image path/to/bird.jpg --checkpoint ... --gradcam` prints top-5 species and saves the heatmap.
- [ ] Benchmark notes (`docs/benchmark.md`): published CUB-200-2011 accuracies for the same backbones, with sources, split and metric, and why numbers differ (input size, epochs, ImageNet overlap with CUB).
- [ ] README: exact, copy-pasteable training and inference commands, and where checkpoints live.
- [ ] Fresh-clone reproducibility check on a new Colab: install, run `eda` and one short `train`. Fix whatever breaks.
- [ ] Compile everyone's report notes and tracker screenshots into the Milestone 1 write-up.
- [ ] Prepare the LMS submission: repo link, Drive link with checkpoints, write-up.
- [ ] Update the progress tracker.

### Files
`src/visualization/gradcam.py`, `src/evaluation/error_analysis.py`, `src/inference/predict.py`, `docs/benchmark.md`, `README.md` (final)

### Example commits
```
feat(viz): Grad-CAM with hooks
feat(eval): confusion pair analysis
feat(cli): classification predict with Grad-CAM
docs: benchmark comparison
docs: README run commands
```

### Ready to hand off when
- `predict` works on any image with any checkpoint
- Grad-CAM figures for correct and wrong cases exist
- A fresh clone reproduces the EDA and a short training run

---

## Sunday 1 November — everyone

- [ ] Each person runs one README command on a fresh Colab clone.
- [ ] Each person explains another person's step out loud for ~5 minutes (the discussion is individual and covers the whole project).
- [ ] Abdallah uploads the Milestone 1 submission to the LMS before the deadline.

---

## Handoff log

> Add one entry when you finish. Keep it short: what's done, where things are, what the next person must know.

### Jana → Habiba
- Date:
- Notes:

### Habiba → Nouran
- Date:
- Notes:

### Nouran → Mohamed
- Date:
- Notes:

### Mohamed → Abdallah
- Date:
- Notes:

### Abdallah → Team (submission)
- Date:
- Notes:
