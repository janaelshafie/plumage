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
