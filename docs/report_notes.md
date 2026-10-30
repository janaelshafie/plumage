# Milestone 1 — Report notes

Each of us writes under our own step: what we did, why, and what went wrong.

## Step 1 — Project setup and data pipeline (Jana)

**What I did**
- Made the folder structure from section 5.2 of the brief, `requirements.txt` and the README.
- Config files in YAML. `configs/base.yaml` has the shared settings, and other configs (like `cls_scratch.yaml`) start from it and only change what they need.
- `CUBDataset` loads the Hugging Face splits `Multimodal-Fatima/CUB_train` (5,994 images) and `CUB_test` (5,794 images). Every image is converted to RGB because a few CUB images are grayscale.
- A validation split: 10% of each class from the train set (seed 42), saved in `configs/splits/val_indices.json`.
- `build_dataloaders(cfg)` gives the train, val and test loaders.
- MLflow tracking in `src/utils/tracking.py`, and the CLI (`python -m src.cli eda/train/tune/predict`).

**Why**
- The dataset only has train and test. If we tuned on test, our test accuracy would look better than it really is, so we take val from train and use test only once at the end.
- Each class has only ~30 images. A normal random split could give some classes 0 or 1 val images, so we split per class to get exactly 3 per class.
- We saved the val indices in a file so everyone tunes on the same 600 images.
- All the MLflow code is in one file, so the training code doesn't depend on MLflow directly.

**Problems**
- `data/` in `.gitignore` also ignored `src/data/` by mistake. Changed it to `/data/`.

## Step 2 — EDA, cleaning, preprocessing and augmentation (Habiba)

All numbers come from `python -m src.cli eda --config configs/base.yaml`. It saves the figures, CSVs and `summary.json` to `outputs/figures/eda/`.

| What we found | What we did | Why |
|---------------|-------------|-----|
| **Classes:** train has 26–27 images per class (imbalance 1.04), val has exactly 3, test has 11–30 (imbalance 2.73; smallest: Least Auklet, Crested Auklet, Spotted Catbird). | No class weights. We report macro F1 next to accuracy. | Train is balanced, so class weights wouldn't change anything. Test is not balanced, and macro F1 counts every species equally. |
| **Image sizes:** almost all images have a longest side of 500 px (median 500×375). Aspect ratio goes from 0.38 to 3.6 (median 1.33). 148 images have a side smaller than 224 px. | Train: `RandomResizedCrop(224, scale=(0.5, 1.0))`. Val/test: `Resize(256)` + `CenterCrop(224)`. | Images are bigger than 224, so we don't lose much detail. We kept at least 50% of the image in the crop because the bird is often small. |
| **Color modes:** 11,780 RGB and 8 grayscale images (4 train, 4 test). No file is broken. | Keep them and convert to RGB. | Only 8 images, and they still show the right bird. |
| **Duplicates:** pHash (≤ 4 bits different) found 8 candidate pairs, but 3 were just different birds on a plain sky or snow background. We checked again with dHash (≤ 8): real duplicates differ by 0–1 bits, wrong matches by 15–17. In the end: 3 train–train pairs and 2 train–test pairs (Bronzed Cowbird, Prothonotary Warbler). | Removed 5 train images (`data.exclude_train_files`), so train went from 5,394 to 5,389. | The train–test duplicates mean the model would see test images while training, which makes test accuracy too high. Val and test are not changed. |
| **Color mean/std** of our train images: mean (0.486, 0.501, 0.433), std (0.232, 0.228, 0.267). ImageNet: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225). | Pretrained models use ImageNet values, and the scratch model uses `normalization: cub`. | Pretrained models expect the values they were trained with. Our images are greener and bluer than ImageNet (trees, sky, water). |
| **Captions:** every image has exactly 10 captions. Captions are 15.3 words on average (median 14, 95% under 24, longest 66). | For Milestone 2, captions can be cut at about 25–30 words. | That covers more than 95% of the captions. |
| **Caption noise:** about 100 captions have a broken character (`�`) instead of every space (`this��bird��has…`). Other odd characters: `/` 266 times, `;` 247, `&` 196, plus quotes and brackets. | `clean_caption` replaces `�` with a space, `&` with "and" and `/` with a space, lowercases everything, removes other symbols and fixes spaces around punctuation. | Without this, those captions would become one giant "word" each. |
| **Vocabulary:** 6,690 different words before cleaning and 4,076 after. Only 1,243 words appear 5 times or more; the other 2,833 are rare and are mostly typos (`abdomin`, `abdomon`, `aare`, `ablack`). | Rare words are saved in `rare_words.csv`. For Milestone 2 we'll keep words that appear at least 5 times and replace the rest with `<unk>`. | This removes almost all typos without fixing 2,800 words by hand. |

**Augmentation** (`src/data/transforms.py`): random crop, horizontal flip, rotation up to 10°, and small changes in brightness, contrast and saturation (0.2) and hue (0.02). No vertical flips, because birds are never upside down. We kept the hue change very small because color is what separates many species (for example Summer Tanager vs Scarlet Tanager). `augmentation_preview.png` shows 8 examples.

**Problems**
- pHash alone matched different birds on plain backgrounds. Lowering the threshold would have missed the Laysan Albatross duplicate (pHash distance 4, dHash 0), so we added dHash as a second check.
- The EDA takes about 3 minutes because it opens all 11,788 images.

## Step 3 — CNN from scratch, training and metrics (Nouran)

**What I did**
- `ScratchCNN` in `src/models/encoders.py`: 5 blocks of (Conv-BN-ReLU ×2, then MaxPool), channels 32 → 64 → 128 → 256 → 512, then global average pooling. `forward_features()` gives the 7×7×512 feature map, and `get_gradcam_layer()` gives the last conv layer for Grad-CAM in Step 5.
- `Classifier` in `src/models/classifier.py`: dropout + one linear layer. `build_model(cfg)` picks the encoder from the config, so changing to ResNet-50 later is a one-line config change.
- `train_model()` in `src/training/trainer.py`: AdamW, cosine learning-rate schedule, mixed precision (only on GPU), early stopping on val accuracy, saves the best model, and logs every epoch to MLflow. The same function will train the pretrained models and the tuning trials.
- `evaluate()` in `src/evaluation/metrics.py`: accuracy, top-5 accuracy, macro precision, recall and F1, and the confusion matrix.
- `plot_training_curves()` in `src/visualization/curves.py`: train vs val loss and accuracy.
- `python -m src.cli train --task classification --config configs/cls_scratch.yaml` runs everything: training, curves, test evaluation and MLflow logging.
- Total: 4.82M parameters (4.72M in the encoder, 103k in the classifier).

| Block | Channels | Output size (224px input) |
|---|---|---|
| 1 | 3 → 32 | 112×112 |
| 2 | 32 → 64 | 56×56 |
| 3 | 64 → 128 | 28×28 |
| 4 | 128 → 256 | 14×14 |
| 5 | 256 → 512 | 7×7 |
| Pooling + linear | 512 → 200 | 1×1 |

**Settings** (`configs/cls_scratch.yaml`): CUB normalization, dropout 0.3, AdamW with lr 3e-4 and weight decay 1e-4, cosine schedule, up to 40 epochs, early stopping after 10 epochs without improvement, mixed precision on.

**Why**
- *Global average pooling instead of flatten:* the last layer is 512 → 200 instead of 25,088 → 200, so there are far fewer parameters and less overfitting.
- *`forward_features` and `get_gradcam_layer` on the encoder:* Grad-CAM (Step 5) and colorization (Milestone 2) both need the feature map before pooling.
- *AdamW + cosine schedule:* it trains well without much learning-rate tuning.
- *Early stopping on val accuracy, not val loss:* accuracy is what we report, and with only ~27 images per class the val loss can keep going down while accuracy stops improving.
- *Mixed precision only on GPU:* the same code runs on a laptop CPU and on the T4.

**Baseline run**
- One epoch at 224px takes about 16 minutes on my CPU, so 30–50 epochs would take 8–13 hours. The full run has to be done on a Kaggle/Colab T4.
- To check that everything works, I did a short run with 96px images and 3 epochs (temporary config, not committed): train loss went from 5.24 to 4.54, val accuracy from 2.2% to 4.5%, and test accuracy was 4.9% (macro F1 2.9%). The numbers are low because it was only 3 epochs on small images. This run was only to test the code.
- **The real 40-epoch, 224px baseline still needs to run on a GPU.**

**Problems**
- The first `train` run crashed when starting MLflow. On Windows, MLflow's default database path breaks when the user folder name has a space (`C:/Users/Nouran Mokhtar/...`). We fixed it by setting the path ourselves with `mlflow.set_tracking_uri("sqlite:///mlflow.db")` in `start_run()`.
