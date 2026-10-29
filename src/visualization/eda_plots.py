"""Exploratory data analysis (EDA) of CUB.

Run it with:  python -m src.cli eda --config configs/base.yaml

Everything is saved in outputs/figures/eda/:
    class_distribution.png    images per class in train / val / test
    image_stats.png           image width, height, aspect ratio, file size, color mode
    channel_stats.png         mean and std of each color channel, CUB vs ImageNet
    duplicates.png            near-duplicate image pairs
    caption_stats.png         captions per image and words per caption
    top_words.png             most common caption words
    sample_grid.png           one image of 24 random species
    augmentation_preview.png  one image with 8 random augmentations
    summary.json              all the numbers used in the report
    image_stats.csv, duplicates.csv, rare_words.csv
"""

import io
import json
import os
import random
import re
from collections import Counter

import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Image as HFImage
from PIL import Image
from tqdm import tqdm

from src.data.cub_dataset import load_cub
from src.data.splits import get_train_val_indices
from src.data.text import clean_captions, split_description, tokenize
from src.data.transforms import denormalize, get_mean_std, get_train_transforms

# Two images are duplicate candidates if their pHashes differ in at most 4 of 64 bits.
# A candidate is confirmed if the dHashes also differ in at most 8 bits.
PHASH_MAX_DISTANCE = 4
DHASH_MAX_DISTANCE = 8
RARE_WORD_COUNT = 5  # words seen fewer than 5 times are probably typos
STOPWORDS = {
    "a", "an", "the", "this", "that", "is", "are", "has", "have", "with", "and", "of", "on", "in", "its",
    "it", "to", "bird", "very", "small", "as", "at", "for", "be", "there", "which", "from", "some",
}


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- class balance

def class_balance(split_labels, class_names, out_dir):
    """Count images per class in each split and plot it."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    summary = {}

    for ax, (split, labels) in zip(axes, split_labels.items()):
        counts = np.bincount(labels, minlength=len(class_names))
        imbalance = counts.max() / counts.min()
        smallest = np.argsort(counts, kind="stable")[:3]
        summary[split] = {
            "images": len(labels),
            "min_per_class": int(counts.min()),
            "max_per_class": int(counts.max()),
            "imbalance_ratio": round(float(imbalance), 3),
            "smallest_classes": [class_names[c] for c in smallest],
        }

        ax.bar(range(len(counts)), counts, width=1.0)
        ax.axhline(counts.mean(), color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"{split}: {len(labels)} images, {counts.min()}-{counts.max()} per class, "
                     f"imbalance {imbalance:.2f}")
        ax.set_ylabel("images")
    axes[-1].set_xlabel("class id")

    save_figure(fig, os.path.join(out_dir, "class_distribution.png"))
    return summary


# ---------------------------------------------------------------- images

def scan_images(hf_data, split, stats_indices):
    """Open every image once and record its size, color mode, file size and hashes.

    Also computes the mean and std of each color channel, using only the images
    in `stats_indices` (the training images; val and test are left out).
    Returns (table with one row per image, channel stats).
    """
    # decode=False gives us the raw file bytes, so we can get the file size
    # and catch files that fail to open.
    raw = hf_data.select_columns(["image", "label", "file_name"])
    raw = raw.cast_column("image", HFImage(decode=False))

    # For the channel stats: sum of pixel values, sum of squared values, pixel count.
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    n_pixels = 0

    rows = []
    for i, item in enumerate(tqdm(raw, desc=f"scanning {split}")):
        data = item["image"]["bytes"]
        row = {"split": split, "index": i, "label": item["label"], "file_name": item["file_name"],
               "file_kb": len(data) / 1024, "ok": True, "error": ""}
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            row["width"], row["height"], row["mode"] = img.width, img.height, img.mode

            rgb = img.convert("RGB")
            row["phash"] = str(imagehash.phash(rgb))
            row["dhash"] = str(imagehash.dhash(rgb))

            if stats_indices is not None and i in stats_indices:
                pixels = np.asarray(rgb) / 255.0              # [H, W, 3] in 0..1
                pixel_sum += pixels.sum(axis=(0, 1))
                pixel_sq_sum += (pixels ** 2).sum(axis=(0, 1))
                n_pixels += pixels.shape[0] * pixels.shape[1]
        except Exception as e:  # a file that can't be opened is a finding, not a crash
            row["ok"] = False
            row["error"] = repr(e)
        rows.append(row)

    channel_stats = None
    if n_pixels > 0:
        mean = pixel_sum / n_pixels
        std = np.sqrt(pixel_sq_sum / n_pixels - mean ** 2)   # std = sqrt(E[x^2] - E[x]^2)
        channel_stats = {"mean": mean.round(4).tolist(), "std": std.round(4).tolist()}
    return pd.DataFrame(rows), channel_stats


def image_statistics(images, image_size, out_dir):
    """Plot and summarize image sizes, aspect ratios, file sizes and color modes."""
    images.to_csv(os.path.join(out_dir, "image_stats.csv"), index=False)
    ok = images[images.ok]
    aspect = ok.width / ok.height

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, column, title in [(axes[0, 0], "width", "width (px)"),
                              (axes[0, 1], "height", "height (px)"),
                              (axes[1, 0], "file_kb", "file size (KB)")]:
        for split in ["train", "test"]:
            ax.hist(ok[ok.split == split][column], bins=50, alpha=0.6, label=split)
        ax.set_xlabel(title)
        ax.legend()
    axes[0, 2].hist(aspect, bins=60)
    axes[0, 2].set_xlabel("aspect ratio (width / height)")
    axes[1, 1].scatter(ok.width, ok.height, s=3, alpha=0.3)
    axes[1, 1].set_xlabel("width")
    axes[1, 1].set_ylabel("height")
    modes = ok.groupby(["mode", "split"]).size().unstack(fill_value=0)
    modes.plot.bar(ax=axes[1, 2], logy=True, rot=0)
    axes[1, 2].set_title("color modes (log scale)")
    save_figure(fig, os.path.join(out_dir, "image_stats.png"))

    too_small = (ok.width < image_size) | (ok.height < image_size)
    return {
        "failed_to_open": images[~images.ok][["split", "file_name", "error"]].to_dict("records"),
        "modes": {f"{split}/{mode}": int(n) for (split, mode), n in ok.groupby(["split", "mode"]).size().items()},
        "non_rgb_files": ok[ok["mode"] != "RGB"][["split", "file_name", "mode"]].to_dict("records"),
        "width": ok.width.describe().round(1).to_dict(),
        "height": ok.height.describe().round(1).to_dict(),
        "aspect_ratio": {"min": round(aspect.min(), 3), "median": round(aspect.median(), 3),
                         "max": round(aspect.max(), 3)},
        "file_kb": ok.file_kb.describe().round(1).to_dict(),
        "smaller_than_224": int(too_small.sum()),
    }


def plot_channel_stats(cub_stats, cfg, out_dir):
    """Bar chart of the CUB channel mean/std next to the ImageNet values."""
    imagenet = {"mean": cfg["data"]["imagenet_mean"], "std": cfg["data"]["imagenet_std"]}
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, key in zip(axes, ["mean", "std"]):
        ax.bar(x - 0.2, cub_stats[key], 0.4, label="CUB train")
        ax.bar(x + 0.2, imagenet[key], 0.4, label="ImageNet")
        ax.set_xticks(x, ["R", "G", "B"])
        ax.set_title(f"channel {key}")
        ax.legend()
    save_figure(fig, os.path.join(out_dir, "channel_stats.png"))


# ---------------------------------------------------------------- duplicates

def hash_to_bits(hex_hashes):
    """Turn hash strings into an array of shape [n, 64] with one 0/1 value per bit."""
    return np.array([imagehash.hex_to_hash(h).hash.flatten() for h in hex_hashes])


def find_duplicates(train_info, test_info, hf_splits, class_names, out_dir):
    """Find near-duplicate images within train and between train and test.

    A perceptual hash (pHash) is a 64-bit fingerprint of an image: similar
    images get similar hashes. Two images are duplicate candidates if their
    pHashes differ in at most 4 bits. pHash alone also matched different
    birds on a plain sky or snow background, so each candidate is checked
    again with a second hash (dHash).
    """
    train = train_info[train_info.ok].reset_index(drop=True)
    test = test_info[test_info.ok].reset_index(drop=True)
    train_bits = hash_to_bits(train.phash)
    test_bits = hash_to_bits(test.phash)

    pairs = []
    for other_split, other, other_bits in [("train", train, train_bits), ("test", test, test_bits)]:
        for i in range(len(train)):
            # number of different bits between train image i and every other image
            distances = (train_bits[i] != other_bits).sum(axis=1)
            for j in np.nonzero(distances <= PHASH_MAX_DISTANCE)[0]:
                if other_split == "train" and j <= i:
                    continue  # skip the image itself, and pairs we already have
                a, b = train.iloc[i], other.iloc[j]
                dhash_distance = imagehash.hex_to_hash(a.dhash) - imagehash.hex_to_hash(b.dhash)
                pairs.append({
                    "split_a": "train", "index_a": a["index"], "file_a": a.file_name,
                    "label_a": class_names[a.label],
                    "split_b": other_split, "index_b": b["index"], "file_b": b.file_name,
                    "label_b": class_names[b.label],
                    "distance": int(distances[j]), "dhash_distance": dhash_distance,
                    "confirmed": dhash_distance <= DHASH_MAX_DISTANCE,
                })

    pairs = pd.DataFrame(pairs).sort_values(["confirmed", "distance"], ascending=[False, True])
    pairs.to_csv(os.path.join(out_dir, "duplicates.csv"), index=False)
    plot_duplicates(pairs.head(8), hf_splits, os.path.join(out_dir, "duplicates.png"))

    confirmed = pairs[pairs.confirmed]
    return {
        "candidate_pairs": len(pairs),
        "train_train_pairs": int((confirmed.split_b == "train").sum()),
        "train_test_pairs": int((confirmed.split_b == "test").sum()),
        "confirmed": confirmed[["file_a", "split_b", "file_b"]].to_dict("records"),
    }


def plot_duplicates(pairs, hf_splits, path):
    """Show each pair side by side with both hash distances."""
    fig, axes = plt.subplots(len(pairs), 2, figsize=(7, 3.2 * len(pairs)), squeeze=False)
    for row_axes, (_, pair) in zip(axes, pairs.iterrows()):
        verdict = "duplicate" if pair.confirmed else "false match"
        for ax, side in zip(row_axes, ["a", "b"]):
            split = pair[f"split_{side}"]
            ax.imshow(hf_splits[split][int(pair[f"index_{side}"])]["image"].convert("RGB"))
            ax.set_title(f"{split}: {pair[f'file_{side}']}\npHash {pair.distance}, "
                         f"dHash {pair.dhash_distance}: {verdict}", fontsize=8)
            ax.axis("off")
    save_figure(fig, path)


# ---------------------------------------------------------------- captions

def caption_statistics(train_descriptions, test_descriptions, out_dir):
    """Captions per image, caption length, vocabulary, rare words (typos) and odd characters."""
    captions_per_image = [len(split_description(d)) for d in train_descriptions]
    cleaned = [clean_captions(d) for d in train_descriptions]
    clean_per_image = [len(caps) for caps in cleaned]
    lengths = [len(tokenize(c)) for caps in cleaned for c in caps]

    raw_vocab = {word for d in train_descriptions for word in d.lower().split()}
    word_counts = Counter(word for caps in cleaned for c in caps for word in tokenize(c))
    # every character that is not a letter, digit, space or normal punctuation
    odd_chars = Counter(re.sub(r"[A-Za-z0-9 ,.'\-\n]", "", "".join(train_descriptions)))

    rare = sorted((n, w) for w, n in word_counts.items() if n < RARE_WORD_COUNT)
    rare = pd.DataFrame([(w, n) for n, w in rare], columns=["word", "count"])
    rare.to_csv(os.path.join(out_dir, "rare_words.csv"), index=False)

    # Figure 1: captions per image and words per caption
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = Counter(clean_per_image)
    axes[0].bar([str(k) for k in sorted(counts)], [counts[k] for k in sorted(counts)])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("captions per image (after cleaning)")
    axes[0].set_ylabel("images (log scale)")
    axes[1].hist(lengths, bins=range(0, max(lengths) + 2))
    axes[1].axvline(np.percentile(lengths, 95), color="black", linestyle="--", label="95th percentile")
    axes[1].set_xlabel("words per caption")
    axes[1].legend()
    save_figure(fig, os.path.join(out_dir, "caption_stats.png"))

    # Figure 2: most common words, without stopwords
    top = [(w, n) for w, n in word_counts.most_common() if w not in STOPWORDS][:30]
    top.reverse()  # largest at the top of the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh([w for w, _ in top], [n for _, n in top])
    ax.set_title("top 30 words (stopwords removed)")
    ax.set_xlabel("count")
    save_figure(fig, os.path.join(out_dir, "top_words.png"))

    return {
        "captions_per_image_train": dict(sorted(Counter(captions_per_image).items())),
        "captions_per_image_test": dict(sorted(Counter(len(split_description(d)) for d in test_descriptions).items())),
        "n_captions": len(lengths),
        "caption_words_mean": round(float(np.mean(lengths)), 2),
        "caption_words_median": float(np.median(lengths)),
        "caption_words_p95": float(np.percentile(lengths, 95)),
        "caption_words_max": max(lengths),
        "vocab_raw": len(raw_vocab),
        "vocab_clean": len(word_counts),
        "words_seen_5_times_or_more": sum(n >= RARE_WORD_COUNT for n in word_counts.values()),
        "rare_words": len(rare),
        "rare_word_examples": rare.word.head(25).tolist(),
        "odd_chars": {repr(ch): n for ch, n in odd_chars.most_common(15)},
    }


# ---------------------------------------------------------------- sample images

def plot_sample_grid(hf_train, labels, class_names, seed, out_dir):
    """Show the first image of 24 random species."""
    first_image_of_class = {}
    for i, label in enumerate(labels):
        if label not in first_image_of_class:
            first_image_of_class[label] = i
    classes = sorted(random.Random(seed).sample(sorted(first_image_of_class), 24))

    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    for ax, c in zip(axes.flat, classes):
        ax.imshow(hf_train[first_image_of_class[c]]["image"].convert("RGB"))
        ax.set_title(class_names[c], fontsize=9)
        ax.axis("off")
    save_figure(fig, os.path.join(out_dir, "sample_grid.png"))


def plot_augmentation_preview(image, cfg, out_dir):
    """Show one image next to 8 random training augmentations of it."""
    transform = get_train_transforms(cfg)
    mean, std = get_mean_std(cfg)

    fig, axes = plt.subplots(1, 9, figsize=(20, 2.6))
    axes[0].imshow(image)
    axes[0].set_title("original")
    for ax in axes[1:]:
        augmented = denormalize(transform(image), mean, std)
        ax.imshow(augmented.permute(1, 2, 0))  # [3, H, W] -> [H, W, 3] for matplotlib
    for ax in axes:
        ax.axis("off")
    save_figure(fig, os.path.join(out_dir, "augmentation_preview.png"))


# ---------------------------------------------------------------- main

def run_eda(cfg):
    """Run every analysis, save all figures and summary.json. Returns (summary, out_dir)."""
    out_dir = os.path.join(cfg["output_dir"], "figures", "eda")
    os.makedirs(out_dir, exist_ok=True)
    seed = cfg["seed"]

    hf_train = load_cub("train", cfg)
    hf_test = load_cub("test", cfg)
    class_names = hf_train.features["label"].names
    train_labels = list(hf_train["label"])
    test_labels = list(hf_test["label"])
    train_idx, val_idx = get_train_val_indices(cfg, train_labels)

    summary = {}

    print("Class balance...")
    split_labels = {
        "train": [train_labels[i] for i in train_idx],
        "val": [train_labels[i] for i in val_idx],
        "test": test_labels,
    }
    summary["class_balance"] = class_balance(split_labels, class_names, out_dir)

    print("Image statistics (this opens every image, ~3 minutes)...")
    train_info, channel_stats = scan_images(hf_train, "train", set(train_idx))
    test_info, _ = scan_images(hf_test, "test", None)
    all_images = pd.concat([train_info, test_info], ignore_index=True)
    summary["images"] = image_statistics(all_images, cfg["data"]["image_size"], out_dir)
    summary["channel_stats_train"] = channel_stats
    plot_channel_stats(channel_stats, cfg, out_dir)

    print("Duplicates...")
    hf_splits = {"train": hf_train, "test": hf_test}
    summary["duplicates"] = find_duplicates(train_info, test_info, hf_splits, class_names, out_dir)

    print("Captions...")
    summary["captions"] = caption_statistics(list(hf_train["description"]), list(hf_test["description"]), out_dir)

    print("Sample images and augmentation preview...")
    plot_sample_grid(hf_train, train_labels, class_names, seed, out_dir)
    preview_index = random.Random(seed).choice(train_idx)
    plot_augmentation_preview(hf_train[preview_index]["image"].convert("RGB"), cfg, out_dir)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary, out_dir
