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
| 1 | Jana | Foundation and data pipeline | Mon 26 Oct | ✅ Done | Mon 26 Oct |
| 2 | Habiba | EDA, cleaning, preprocessing, augmentation | Tue 27 Oct | ✅ Done | Mon 26 Oct |
| 3 | Nouran | From-scratch CNN, trainer, metrics | Wed 28 Oct | 🟡 Code done, baseline run pending | |
| 4 | Mohamed | Transfer learning and hyperparameter tuning | Thu 29 – Fri 30 Oct (2 pm) | ⚪ Waiting | |
| 5 | Abdallah | Grad-CAM, inference, benchmarking, submission | Fri 30 – Sat 31 Oct | ⚪ Waiting | |
| – | Everyone | Final review and LMS submission | Sun 1 Nov | ⚪ Waiting | |

**Baton is currently with:** Nouran (full baseline still needs to run on Kaggle/Colab T4 before handoff to Mohamed)

---

## Rules for every step

- **Pull first.** `git pull` on `main` before you start. Work on a branch named `stepN-yourname` (e.g. `step2-habiba`).
- **Commit from your own account,** small and often. Several meaningful commits per person, never one big upload at the end.
- **Open a PR; the previous person reviews it.** Reviewing is how everyone learns the whole project for the individual discussion (35% of the grade).
- **No final logic in notebooks.** Explore in `notebooks/`, but anything kept moves into `src/` and runs through the CLI.
- **Log every run** in MLflow (`src/utils/tracking.py`). Never commit checkpoints or data; checkpoints go to the team Google Drive folder.
- **Write report notes** in `docs/report_notes.md` under your step: what you did, why, and what failed.
- **Use the "Before your turn" time.** Don't commit early, but prepare so your step goes fast.

---

## Step 1 — Jana: Foundation and data pipeline

**When:** Mon 26 Oct
**Goal:** Build the skeleton everyone else builds on: repo, project layout, configs, dataset over the Hugging Face splits, a fixed validation split, the tracker wrapper, and the CLI entry point.

### Checklist
- [x] Create the GitHub repo and add Habiba, Nouran, Mohamed and Abdallah as collaborators. Add a `.gitignore` for `data/`, `outputs/`, `mlruns/`, `*.pt`, `*.pth`, `.ipynb_checkpoints`.
- [x] Create the folder layout from section 5.2 of the brief: `configs/`, `notebooks/`, `docs/`, `src/{data,models,training,evaluation,visualization,utils}/` with `__init__.py`, `outputs/`.
- [x] Write `requirements.txt`: torch, torchvision, datasets, mlflow, matplotlib, seaborn, scikit-learn, pandas, pyyaml, tqdm, pillow, imagehash.
- [x] Config and reproducibility helpers in `src/utils/`: `load_config(path)`, `set_seed(seed)` (python/numpy/torch), `get_device()`.
- [x] `CUBDataset` in `src/data/cub_dataset.py`: loads `Multimodal-Fatima/CUB_train` and `CUB_test`, converts every image to RGB, returns `(image_tensor, label)`, takes a `transform` argument so Habiba can plug in augmentation.
- [x] Stratified validation split: hold out ~10% of the train set per class with a fixed seed. Save indices to `configs/splits/val_indices.json` and commit it. **The test set is never used for tuning.**
- [x] `build_dataloaders(cfg, train_tf, eval_tf)` returning train / val / test loaders.
- [x] Experiment tracker wrapper in `src/utils/tracking.py` (local MLflow) so the backend can be swapped in one file.
- [x] CLI skeleton in `src/cli.py` with subcommands `train`, `predict`, `eda`, `tune`, each taking `--task` and `--config` (can print "not implemented yet" for now).
- [x] README: environment setup section.
- [x] Update the progress tracker and hand off to Habiba.

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
- One batch from the dataloaders has shape `[B, 3, H, W]` and labels in 0–199
- A test run appears in the local MLflow tracker

---

## Step 2 — Habiba: EDA, cleaning, preprocessing and augmentation

**When:** Tue 27 Oct
**Before your turn:** Load the dataset in a notebook and browse it; read section 6.1.1 of the brief.
**Goal:** Understand the data and turn findings into code: image and caption plots, caption cleaning, normalization statistics, and a justified augmentation strategy.

### Checklist
- [x] Class distribution plot for train / val / test, plus imbalance ratio (max/min).
- [x] Image statistics: width, height, aspect ratio, image modes (look for grayscale or RGBA images), file sizes.
- [x] Duplicate and corruption check: perceptual hashes (`imagehash`) within train and across train/test; list anything that fails to decode.
- [x] Channel mean and std on the train split; compare with ImageNet stats, write both into `configs/base.yaml`, note which is used for which model.
- [x] Sample grid of images with species names.
- [x] Caption analysis: split `description` on `\n`, check each image has 10 captions, plot caption length in words, vocabulary size, top words, rare words (freq < 5) as typo candidates, odd characters.
- [x] Caption cleaning in `src/data/text.py`: `split_description()` and `clean_caption()` (strip, lowercase, normalize punctuation/whitespace, drop empty lines). Milestone 2 captioning reuses this.
- [x] Augmentation in `src/data/transforms.py`:
  - Train: `RandomResizedCrop(224)`, horizontal flip, small rotation, mild `ColorJitter`
  - Eval: `Resize(256)` + `CenterCrop(224)`
  - No vertical flips; keep hue jitter small because plumage color identifies species.
- [x] Augmentation preview figure (one image, 8 random augmentations).
- [x] `eda` CLI command: `python -m src.cli eda --config configs/base.yaml` writes all figures to `outputs/figures/eda/`.
- [x] Report notes, Data section: each finding, the decision it led to, and why.
- [x] Update the progress tracker and hand off to Nouran.

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
- [x] Scratch encoder in `src/models/encoders.py`: ~5 blocks of `[Conv-BN-ReLU ×2, MaxPool]`, channels 32→512, global average pooling. Expose `forward_features()` (spatial map), `out_dim`, and `get_gradcam_layer()` (last conv). Colorization in Milestone 2 reuses this encoder.
- [x] Classifier head in `src/models/classifier.py`: `Classifier(encoder, num_classes, dropout)` and `build_model(cfg)` that picks the encoder by name, so new encoders don't touch the trainer.
- [x] `train_model()` in `src/training/trainer.py`: train/validate loops, AdamW, cosine schedule, mixed precision for the T4, early stopping, best-checkpoint saving, per-epoch logging to MLflow.
- [x] Metrics in `src/evaluation/metrics.py`: accuracy, top-5 accuracy, macro precision/recall/F1, confusion matrix; `evaluate(model, loader)`.
- [x] Training curves in `src/visualization/curves.py` (train vs val loss and accuracy).
- [x] Wire up: `python -m src.cli train --task classification --config configs/cls_scratch.yaml`
- [ ] **Run the baseline on a T4 for 30–50 epochs. Log it, evaluate on test once, upload best checkpoint to Drive.** Not done on this CPU-only dev machine (~16 min/epoch, 8–13+ hours for 30–50 epochs) — needs Kaggle/Colab. A reduced-scale local run (96px, 3 epochs) verified the pipeline works end to end; see report notes.
- [x] Report notes: architecture table/diagram, settings, curves, results, and why accuracy is modest (6k images, 200 fine-grained classes, no pretraining).
- [x] Update the progress tracker and hand off to Mohamed (include the checkpoint link and run link in the PR). — tracker updated; checkpoint/run links still to come once the real baseline is run.

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
**Before your turn:** Look at `ScratchCNN` and `build_encoder()` in `src/models/encoders.py`, and at `train_model()` in `src/training/trainer.py`.
**Goal:** Add one pretrained model, try it frozen and fine-tuned, tune it with grid and random search, and train the final model.

### Checklist
- [ ] Add a `ResNet50Encoder` class in `src/models/encoders.py`: load `torchvision.models.resnet50(weights="IMAGENET1K_V2")`, remove its last layer, and give it the same methods as `ScratchCNN` (`forward_features`, `forward`, `out_dim`, `get_gradcam_layer`). Add `"resnet50"` to `build_encoder()`.
- [ ] Add `freeze_backbone: true/false` under `model:` in the config. If true, set `requires_grad = False` on the encoder's parameters so only the last layer trains.
- [ ] Two configs: `configs/cls_resnet50_frozen.yaml` and `configs/cls_resnet50_finetune.yaml` (keep `normalization: imagenet`). Train both with the normal `train` command.
- [ ] `src/training/tuning.py` with two simple functions:
  - `grid_search(cfg)`: tries every combination of learning rate `[1e-4, 3e-4]` × weight decay `[1e-4, 1e-3]` = 4 trials
  - `random_search(cfg, n_trials=4)`: picks 4 random combinations from the same ranges
  - Each trial is a short `train_model()` run (~5 epochs) and its own MLflow run. Print the best one at the end.
- [ ] Add `tune` to the CLI: `python -m src.cli tune --task classification --method grid --config configs/cls_resnet50_finetune.yaml`
- [ ] Small table: grid vs random (best val accuracy, number of trials, total time).
- [ ] Train the best setting for the full number of epochs, evaluate on test, and upload the checkpoint to Drive.
- [ ] Small table: scratch vs frozen vs fine-tuned (test accuracy, macro F1, number of parameters, training time).
- [ ] Report notes, then update the progress tracker and hand off to Abdallah.

### Files
`src/models/encoders.py`, `src/training/tuning.py`, `src/cli.py`, `configs/cls_resnet50_frozen.yaml`, `configs/cls_resnet50_finetune.yaml`

### Example commits
```
add pretrained ResNet-50 encoder
add option to freeze the backbone
add grid and random search
add tune command to the CLI
config for the final tuned model
```

### Ready to hand off when
- The best checkpoint is on Drive and its config is committed
- Both tables are in the report notes

---

## Step 5 — Abdallah: Grad-CAM, inference, benchmarking and submission

**When:** Fri 30 – Sat 31 Oct
**Before your turn:** Read how Grad-CAM works (Selvaraju et al., 2017) and look at `get_gradcam_layer()` in `src/models/encoders.py`.
**Goal:** Show what the models look at with Grad-CAM, add the `predict` command, compare with published results, and submit.

### Checklist
- [ ] `src/visualization/gradcam.py`: our own Grad-CAM.
  - Use a forward hook and a backward hook on `model.get_gradcam_layer()` to save the feature maps and their gradients.
  - Average the gradients to get one weight per channel, take the weighted sum of the feature maps, apply ReLU, resize to the image size, and draw it over the image.
- [ ] Grad-CAM figures: a few correct predictions, a few wrong ones (show true vs predicted), and the same images with the scratch model vs the ResNet-50 model.
- [ ] Error analysis: use the confusion matrix from `evaluate()` to list the 5 most confused species pairs.
- [ ] Add `predict` to the CLI: `python -m src.cli predict --task classification --image bird.jpg --checkpoint best.pt --gradcam`. It prints the top-5 species and saves the heatmap. The checkpoint already contains the config, so the model can be rebuilt with `build_model(checkpoint["config"])`.
- [ ] `docs/benchmark.md`: a small table with 2–3 published CUB-200 accuracies for ResNet-50 (with links), next to ours, and a few sentences on why they differ (image size, epochs, tricks).
- [ ] README: add the final `train`, `tune` and `predict` commands and the Drive link for checkpoints.
- [ ] Clone the repo on a fresh Colab, install, and run `eda` and a short `train`. Fix anything that breaks.
- [ ] Put everyone's report notes and MLflow screenshots together into the Milestone 1 write-up.
- [ ] Update the progress tracker.

### Files
`src/visualization/gradcam.py`, `src/cli.py`, `docs/benchmark.md`, `README.md`

### Example commits
```
add Grad-CAM
add Grad-CAM figures for correct and wrong predictions
add predict command to the CLI
add benchmark comparison
update README with final commands
```

### Ready to hand off when
- `predict` works on an image with both checkpoints
- Grad-CAM figures for correct and wrong cases exist
- A fresh Colab clone runs `eda` and a short `train`

---

## Sunday 1 November — everyone

- [ ] Each person runs one README command on a fresh Colab clone.
- [ ] Each person explains another person's step out loud for ~5 minutes (the discussion is individual and covers the whole project).
- [ ] Abdallah uploads the Milestone 1 submission to the LMS before the deadline.

---

## Handoff log

> Add one entry when you finish. Keep it short: what's done, where things are, what the next person must know.

### Jana → Habiba
- Date: Mon 26 Oct 2026
- Notes:
  - Setup and checks are in the README: `pip install -r requirements.txt`, then `python -m pytest`. The first run downloads ~1.2 GB into `data/hf_cache/`.
  - Splits: train 5,394 / val 600 (3 per class, seed 42, `configs/splits/val_indices.json`, don't regenerate) / test 5,794.
  - `CUBDataset(hf_data, indices, transform)` returns `(tensor, label)`. For EDA load the raw split with `load_cub("train", cfg)` and read columns directly (`hf["label"]`, `hf["description"]`).
  - `build_dataloaders(cfg)` builds the transforms from `src/data/transforms.py`.
  - Put the CUB mean/std next to the ImageNet ones under `data:` in `configs/base.yaml`.
  - The `eda` CLI command is `eda_command` in `src/cli.py`.
  - Tracking is local MLflow only (`mlflow.db`, gitignored); there is no shared tracker.

### Habiba → Nouran
- Date: Mon 26 Oct 2026
- Notes:
  - Transforms: `from src.data.transforms import get_train_transforms, get_eval_transforms`. Both take `cfg`. `build_dataloaders(cfg)` already uses them.
  - Normalization: set `data.normalization: cub` in `configs/cls_scratch.yaml`; the CUB train mean/std are in `base.yaml`. Pretrained models (Step 4) keep the default `imagenet`.
  - Train is now 5,389 images: `data.exclude_train_files` drops 5 duplicates (2 of them were also in test). Val (600) and test (5,794) are unchanged.
  - Classes are balanced in train (26–27 each) but not in test (11–30), so report macro F1 next to accuracy.
  - `python -m src.cli eda` takes about 3 minutes; figures and `summary.json` are in `outputs/figures/eda/`. Findings and decisions are in `docs/report_notes.md`.
  - Captions for Milestone 2: `clean_captions(description)` in `src/data/text.py`.

### Nouran → Mohamed
- Date: Wed 28 Oct 2026
- Notes:
  - **Code is done and merged-ready; the real 30–50 epoch baseline still needs to run on Kaggle/Colab T4** before this step is truly finished — this CPU dev machine measured ~16 min/epoch at 224px, so the full run isn't practical here. Don't start Step 4 training off an unrun baseline; ping me once it's run or run it yourself off `configs/cls_scratch.yaml` (same command below) if you need to start sooner.
  - `python -m src.cli train --task classification --config configs/cls_scratch.yaml` trains, early-stops on val accuracy, saves the best checkpoint to `outputs/checkpoints/<run_name>_best.pt`, plots curves to `outputs/figures/curves/<run_name>.png`, evaluates once on test, and logs everything to MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).
  - Encoders: `build_model(cfg)` in `src/models/classifier.py` calls `build_encoder(cfg["model"]["encoder"])` in `src/models/encoders.py`. Add your pretrained backbones as new classes there with the same `forward_features()` / `forward()` / `out_dim` / `get_gradcam_layer()`, and add their names to `build_encoder()`. `train_model()`, `evaluate()` and the CLI don't change.
  - `train_model(model, train_loader, val_loader, cfg, run_name)` (`src/training/trainer.py`) reads everything from `cfg["train"]`: `epochs`, `lr`, `weight_decay`, `early_stopping_patience`, `amp`, `checkpoint_dir`. It returns `(history, checkpoint_path)`. For frozen-backbone training, set the backbone's params to `requires_grad_(False)` before calling it.
  - Fixed a real bug while testing: MLflow 3.x's default tracking URI breaks on Windows when the home folder has a space (e.g. this machine's `Nouran Mokhtar`). `start_run()` in `src/utils/tracking.py` sets `sqlite:///mlflow.db` itself, so you shouldn't need to touch it.
  - `data.normalization: cub` is set in `configs/cls_scratch.yaml` for the from-scratch model; keep `imagenet` (the `base.yaml` default) for your pretrained backbones since they expect ImageNet stats.

### Mohamed → Abdallah
- Date:
- Notes:

### Abdallah → Team (submission)
- Date:
- Notes:
