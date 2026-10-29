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

## Step 2 — EDA, cleaning, preprocessing and augmentation (Habiba)

All numbers come from `python -m src.cli eda --config configs/base.yaml`, which writes the figures, `summary.json` and CSVs to `outputs/figures/eda/` and logs them to MLflow.

| Finding | Decision | Why |
|---------|----------|-----|
| **Class balance.** Train 26–27 images per class (imbalance 1.04), val exactly 3 per class, test 11–30 per class (imbalance 2.73; smallest: Least Auklet, Crested Auklet, Spotted Catbird). | No class weighting or resampling. Report macro F1 next to accuracy. | Training is balanced, so a weighted loss changes nothing. Test is the uneven split, and macro F1 keeps rare test classes from being hidden by accuracy. |
| **Image sizes.** Max side is 500 px for almost all images (median 500×375). Aspect ratio 0.38–3.6, median 1.33. 148 images have a side under 224 px. | Train: `RandomResizedCrop(224, scale=(0.5, 1.0))`. Eval: `Resize(256)` + `CenterCrop(224)`. | Every image is larger than or close to 224 px, so 224 loses little detail. The crop range covers the varied aspect ratios without squashing birds. The lower bound 0.5 keeps most of the bird in frame because birds are often small in the photo. |
| **Image modes.** 11,780 RGB and 8 grayscale (`L`): 4 train, 4 test (gulls, cormorant, nutcracker, raven, mallard, blackbird). No file fails to decode. | Keep them; `CUBDataset` already converts to RGB. | Eight images are too few to matter, and a grayscale photo is still a valid example of its species. |
| **Duplicates.** pHash (≤ 4 bits) gave 8 candidate pairs. Three were different species against plain sky or snow. A second hash (dHash ≤ 8) separates them cleanly: true duplicates differ by 0–1 bits, false matches by 15–17. Confirmed: 3 train–train pairs and 2 train–test pairs (Bronzed Cowbird, Prothonotary Warbler). None involve val. | Drop 5 train images (`data.exclude_train_files`): the 2 that are also in test, and one from each train–train pair. Train goes from 5,394 to 5,389. | The train–test pairs leak test images into training and would inflate test accuracy. Train–train duplicates only double-count a photo. Val and test are untouched, so `val_indices.json` stays valid for everyone. |
| **Channel stats** (train portion, val excluded, full resolution): mean (0.486, 0.501, 0.433), std (0.232, 0.228, 0.267). ImageNet: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225). | `data.normalization: imagenet` by default (pretrained backbones, Step 4). From-scratch models (Step 3) set `normalization: cub`. | Pretrained weights expect the statistics they were trained with. A scratch model has no such constraint, so it uses the true data statistics. CUB is greener and bluer than ImageNet (foliage, sky, water). |
| **Captions.** Every image in train and test has exactly 10 captions. Captions are 15.3 words on average (median 14, 95th percentile 24, max 66). | Milestone 2 can cap caption length at about 25–30 tokens. | Covers 95%+ of captions without padding to the 66-word outlier. |
| **Caption noise.** About 100 train captions store every space as `U+FFFD U+FFFD` (`this��bird��has…`). Other non-alphanumeric characters: `/` 266, `;` 247, `&` 196, quotes, brackets. | `clean_caption` maps `U+FFFD` to a space, `&` to "and", `/` to a space, lowercases, removes other symbols and normalizes spacing and punctuation. | Without the fix, those captions become single giant "words" and pollute the vocabulary. |
| **Vocabulary.** 6,690 distinct whitespace tokens raw, 4,076 after cleaning. Only 1,243 words appear ≥ 5 times; 2,833 are rare, mostly typos (`abdomin`, `abdomon`, `aare`, `ablack`). | Rare words are listed in `rare_words.csv`. Milestone 2 should use a min-frequency cutoff (5) with `<unk>`, not spell-correction. | A frequency cutoff removes almost all typos automatically and keeps the vocabulary small. Hand-correcting 2,800 words isn't worth it. |

**Augmentation** (`src/data/transforms.py`, settings under `data.augment`): random resized crop, horizontal flip (p = 0.5), rotation ±10°, color jitter (brightness, contrast and saturation 0.2; hue 0.02). There are no vertical flips because birds are never upside down in CUB. Hue jitter stays small because plumage color is what separates many species (for example Summer vs Scarlet Tanager); a large hue shift would turn one species into another. `augmentation_preview.png` shows 8 samples of one image.

**What failed / notes**
- pHash alone at distance ≤ 4 flagged different species photographed against uniform backgrounds. Adding dHash as a second check fixed it without lowering the threshold, which would have missed the Laysan Albatross duplicate (pHash 4, dHash 0).
- The full EDA takes about 3 minutes because it decodes and hashes all 11,788 images.

## Step 3 — From-scratch CNN, trainer and metrics (Nouran)

**What**
- `ScratchCNN` (`src/models/encoders.py`): 5 `ConvBlock`s (`Conv-BN-ReLU x2` then `MaxPool(2)`), channels 32→64→128→256→512, global average pooling. `forward_features()` returns the pre-pool spatial map (`7x7x512` at 224px input) and `gradcam_layer` exposes the last conv layer for Step 5's hooks. Registered under `ENCODERS["scratch_cnn"]` so Step 4 only has to add entries, not touch callers.
- `Classifier` (`src/models/classifier.py`): dropout + linear head over any registered encoder's `out_dim`. `build_model(cfg)` reads `model.encoder`/`model.dropout`, so swapping to a pretrained backbone later is a one-line config change.
- `Trainer` (`src/training/trainer.py`): AdamW, cosine LR schedule over `train.epochs`, autocast mixed precision (enabled only on CUDA — a no-op on this CPU dev machine), early stopping on val accuracy (`train.early_stopping_patience`), best-checkpoint saving (`model_state`, `epoch`, `val_acc`, full `config`), per-epoch metrics through the tracker wrapper. One `Trainer` serves every later model and every tuning trial.
- `evaluate()` (`src/evaluation/metrics.py`): accuracy, top-5 accuracy, macro precision/recall/F1 and the confusion matrix in one pass. Macro metrics are reported because test is imbalanced (11–30 images/class, Step 2's finding).
- `plot_training_curves()` (`src/visualization/curves.py`): train vs val loss and accuracy, two panels, one PNG.
- Wired into the CLI: `python -m src.cli train --task classification --config configs/cls_scratch.yaml` trains, saves the best checkpoint, plots curves, evaluates once on test, and logs everything (params, per-epoch metrics, curves figure, checkpoint, test metrics) to the local MLflow tracker.
- Architecture: 4.82M parameters total (4.72M encoder, 103k classifier head).

| Block | In → Out channels | Output size @224px input |
|---|---|---|
| 1 | 3 → 32 | 112×112 |
| 2 | 32 → 64 | 56×56 |
| 3 | 64 → 128 | 28×28 |
| 4 | 128 → 256 | 14×14 |
| 5 | 256 → 512 | 7×7 |
| GAP + FC | 512 → 200 | 1×1 |

**Settings** (`configs/cls_scratch.yaml`): `data.normalization: cub` (Step 2's CUB train stats, not ImageNet's, since nothing here is pretrained), dropout 0.3, AdamW lr 3e-4, weight decay 1e-4, cosine schedule, up to 40 epochs with early-stopping patience 10, mixed precision on.

**Why**
- *Global average pooling, not flatten:* keeps the classifier head small (`512 -> 200` instead of `512*7*7 -> 200`) and makes the encoder input-size agnostic, which Milestone 2's colorization head also needs.
- *`forward_features` / `gradcam_layer` on the encoder, not the classifier:* Step 5's Grad-CAM and Milestone 2's colorization decoder both need the pre-pool spatial map; exposing it on every registered encoder means neither has to special-case which model it's given.
- *AdamW + cosine schedule over flat SGD:* adaptive per-parameter learning rates and decoupled weight decay converge faster with no LR warmup/decay tuning needed for a short-epoch budget.
- *Early stopping on val accuracy, not val loss:* accuracy is the metric that is reported, and with 200 fine-grained classes and only ~27 train images/class, val loss can keep falling slightly while val accuracy plateaus or overfits.
- *AMP as a no-op off CUDA:* `torch.autocast`/`GradScaler` are gated on `device.type == "cuda"`, so the same trainer code runs correctly (if slowly) on this CPU dev machine and at full speed on the T4.

**Baseline run**
- One real epoch over the full 5,389-image train set at 224px takes about 16 minutes of pure forward+backward on this CPU-only dev machine (measured with `time` around 5 raw training batches, batch size 32) — 30–50 epochs would be 8–13+ hours, far past what's practical here. This matches the plan's own assumption (Step 3's "before your turn" note: set up a Kaggle/Colab notebook) that the actual baseline run happens on a T4, not a laptop CPU.
- To verify the full pipeline end to end on this machine, I ran a reduced-scale sanity check instead of the real baseline: same code, same `configs/cls_scratch.yaml`, but `data.image_size: 96` and `train.epochs: 3` (not committed — a throwaway config). Result: train loss 5.24→4.54, train acc 1.1%→4.2%, val acc 2.2%→4.5% over the 3 epochs, each epoch saved as the new best checkpoint, curves plotted, and test evaluation logged accuracy 4.9% / macro F1 2.9% next to each other as intended. Low numbers are expected (3 epochs, 96px, 200 fine-grained classes) — this run exists only to prove the wiring, not to report a result.
- **The actual 40-epoch, 224px baseline still needs to run on Kaggle/Colab T4** before this step is fully done; see the handoff note below for what Mohamed (or whoever runs it) needs.

**What failed / notes**
- The first `train` run (and, it turns out, Step 1's `smoke_test.py` run too) crashed inside `get_tracker`: MLflow 3.x's default tracking URI resolves `mlflow.db` to an absolute, URL-quoted path (`sqlite:///C:/Users/Nouran%20Mokhtar/...`) and then fails to create that path's parent directory on Windows because the `%20` isn't decoded back to a space before the filesystem call. It only shows up on a machine whose user folder has a space in the name. Fixed by having `MLflowTracker.__init__` call `mlflow.set_tracking_uri("sqlite:///mlflow.db")` explicitly before `set_experiment`, which keeps mlflow from re-resolving the path itself (`src/utils/tracking.py`, regression test in `tests/test_tracking_step3.py`).
- Local verification uses `image_size: 96` (not committed) purely because CPU time scales with image area; the committed `configs/cls_scratch.yaml` keeps `image_size: 224` inherited from `base.yaml` untouched for the real run.
