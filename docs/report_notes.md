# Milestone 1 — Report notes

Each person writes under their step: what was done, why, and what failed.

## Step 1 — Foundation and data pipeline (Jana)

**What**
- Project layout from section 5.2 of the brief, a YAML config system with inheritance (`base: base.yaml`) and seeding/device helpers.
- `CUBDataset` over the Hugging Face splits `Multimodal-Fatima/CUB_train` (5,994 images) and `CUB_test` (5,794 images), 200 classes.
  Every image is converted to RGB because CUB contains a few non-RGB images. Transforms are injected, so augmentation (Step 2) plugs in without touching the dataset.
- A fixed stratified validation split: 10% of each class of the train set (seed 42), saved to `configs/splits/val_indices.json`.
- `build_dataloaders(cfg, train_tf, eval_tf)` → train / val / test loaders with seeded shuffling and worker seeding.
- Tracker wrapper (`src/utils/tracking.py`) logging runs locally with MLflow.
- CLI skeleton `python -m src.cli {eda,train,tune,predict} --task --config`.

**Why**
- *Validation from train, not test:* the HF dataset ships only train and test. Tuning on test would leak and inflate reported accuracy, so test is used once per final model.
- *Stratified per class:* ~30 images per class means a random 10% split could leave a class with 0–1 val images. Per-class sampling guarantees ~3 per class, so val accuracy covers all 200 species.
- *Committed indices:* every teammate and every tuning trial compares on exactly the same 600 images.
- *Tracker behind a wrapper:* the trainer only calls `log_metrics`/`log_artifact`, so the logging backend can change without touching training code.

**What failed / notes**
- `data/` in `.gitignore` initially also matched `src/data/`; anchored to `/data/`.
