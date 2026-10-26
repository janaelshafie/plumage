"""EDA for CUB: statistics, figures and a ``summary.json``.

Entry point is ``run_eda(cfg)``, called by ``python -m src.cli eda``. Everything
goes to ``outputs/figures/eda/``:

    class_distribution.png   images per class for train / val / test
    image_stats.png          width, height, aspect ratio, file size, modes
    channel_stats.png        CUB train mean/std next to ImageNet
    sample_grid.png          one image for each of 24 random species
    duplicates.png           near-duplicate pairs found with perceptual hashes
    caption_stats.png        captions per image and caption length
    top_words.png            most frequent content words
    augmentation_preview.png one image under 8 random train augmentations
    summary.json             every number quoted in the report notes
    image_stats.csv, duplicates.csv, rare_words.csv
"""

from __future__ import annotations

import io
import json
import random
import re
from collections import Counter
from pathlib import Path

import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset as HFDataset
from datasets import Image as HFImage
from PIL import Image
from tqdm import tqdm

from src.data.cub_dataset import load_hf_split
from src.data.splits import get_train_val_indices
from src.data.text import clean_captions, split_description, tokenize
from src.data.transforms import denormalize, get_norm_stats, get_train_transforms

DUP_MAX_HAMMING = 4   # pHash bit distance (of 64) that makes a pair a duplicate candidate
DUP_CONFIRM_DHASH = 8  # candidates also within this dHash distance count as confirmed duplicates
RARE_WORD_FREQ = 5    # words seen fewer times are typo candidates
EXPECTED_CAPTIONS = 10
STOPWORDS = {
    "a", "an", "the", "this", "that", "is", "are", "has", "have", "with", "and", "of", "on", "in", "its",
    "it", "to", "bird", "very", "small", "as", "at", "for", "be", "there", "which", "from", "some",
}


# ---------------------------------------------------------------- image scan

def scan_images(hf: HFDataset, split: str, stats_indices: set[int] | None = None) -> tuple[pd.DataFrame, dict]:
    """Decode every image once from its raw bytes.

    Returns one row per image (size, mode, file size, pHash, dHash, decode error) and
    per-channel pixel mean/std over the rows in ``stats_indices`` (full
    resolution, RGB in [0, 1]).
    """
    raw = hf.select_columns(["image", "label", "file_name"]).cast_column("image", HFImage(decode=False))
    sums, sq_sums, n_pixels = np.zeros(3), np.zeros(3), 0
    rows = []
    for i, rec in enumerate(tqdm(raw, desc=f"scan {split}", total=len(raw))):
        data = rec["image"]["bytes"]
        if data is None:
            data = Path(rec["image"]["path"]).read_bytes()
        row = {"split": split, "index": i, "label": rec["label"], "file_name": rec["file_name"],
               "file_kb": len(data) / 1024, "ok": True, "error": ""}
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            row.update(width=img.width, height=img.height, mode=img.mode, format=img.format)
            rgb = img.convert("RGB")
            row["phash"] = str(imagehash.phash(rgb))
            row["dhash"] = str(imagehash.dhash(rgb))
            if stats_indices is not None and i in stats_indices:
                arr = np.asarray(rgb, dtype=np.float64) / 255.0
                sums += arr.sum(axis=(0, 1))
                sq_sums += (arr ** 2).sum(axis=(0, 1))
                n_pixels += arr.shape[0] * arr.shape[1]
        except Exception as e:  # noqa: BLE001 - any decode failure is a finding
            row.update(ok=False, error=repr(e))
        rows.append(row)

    channel = {}
    if n_pixels:
        mean = sums / n_pixels
        channel = {"mean": mean.round(4).tolist(),
                   "std": np.sqrt(sq_sums / n_pixels - mean ** 2).round(4).tolist(),
                   "n_pixels": int(n_pixels)}
    return pd.DataFrame(rows), channel


def find_near_duplicates(hashes_a: list[str], hashes_b: list[str] | None = None,
                         max_dist: int = DUP_MAX_HAMMING) -> list[tuple[int, int, int]]:
    """``(i, j, distance)`` for hash pairs within ``max_dist`` bits.

    With ``hashes_b=None`` it searches within ``hashes_a`` (each pair once, i < j).
    """
    a = np.array([int(h, 16) for h in hashes_a], dtype=np.uint64)
    b = a if hashes_b is None else np.array([int(h, 16) for h in hashes_b], dtype=np.uint64)
    pairs = []
    for start in range(0, len(a), 512):
        dist = np.bitwise_count(a[start:start + 512, None] ^ b[None, :])
        for di, j in zip(*np.nonzero(dist <= max_dist)):
            i = start + int(di)
            if hashes_b is None and j <= i:
                continue
            pairs.append((i, int(j), int(dist[di, j])))
    return pairs


# ----------------------------------------------------------------- captions

def caption_stats(descriptions: list[str]) -> tuple[dict, Counter, pd.DataFrame]:
    """Caption counts, lengths, vocabulary and odd characters over raw description blocks."""
    raw_counts = [len(split_description(d)) for d in descriptions]
    cleaned = [clean_captions(d) for d in descriptions]
    clean_counts = [len(c) for c in cleaned]
    lengths = [len(tokenize(c)) for caps in cleaned for c in caps]

    raw_vocab = {w for d in descriptions for w in d.lower().split()}
    words = Counter(w for caps in cleaned for c in caps for w in tokenize(c))
    odd_chars = Counter(ch for d in descriptions for ch in re.sub(r"[A-Za-z0-9 ,.'\-\n]", "", d))

    stats = {
        "n_images": len(descriptions),
        "captions_per_image": dict(sorted(Counter(raw_counts).items())),
        "images_not_10_captions_raw": sum(c != EXPECTED_CAPTIONS for c in raw_counts),
        "images_not_10_captions_clean": sum(c != EXPECTED_CAPTIONS for c in clean_counts),
        "n_captions": len(lengths),
        "caption_words_mean": round(float(np.mean(lengths)), 2),
        "caption_words_median": float(np.median(lengths)),
        "caption_words_p95": float(np.percentile(lengths, 95)),
        "caption_words_max": int(max(lengths)),
        "vocab_raw_whitespace": len(raw_vocab),
        "vocab_clean": len(words),
        "vocab_freq_ge_5": sum(n >= RARE_WORD_FREQ for n in words.values()),
        "rare_words": sum(n < RARE_WORD_FREQ for n in words.values()),
        "odd_chars": {repr(ch): n for ch, n in odd_chars.most_common(15)},
    }
    rare = pd.DataFrame(
        sorted(((w, n) for w, n in words.items() if n < RARE_WORD_FREQ), key=lambda x: (x[1], x[0])),
        columns=["word", "count"],
    )
    stats["_lengths"] = lengths
    stats["_clean_counts"] = clean_counts
    return stats, words, rare


# ------------------------------------------------------------------ figures

def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_class_distribution(label_lists: dict[str, list[int]], path: Path) -> Path:
    fig, axes = plt.subplots(len(label_lists), 1, figsize=(14, 3 * len(label_lists)), sharex=True)
    for ax, (name, labels) in zip(axes, label_lists.items()):
        counts = np.bincount(labels, minlength=200)
        ax.bar(range(200), counts, width=1.0, color="#4c72b0")
        ax.axhline(counts.mean(), color="k", lw=0.8, ls="--")
        ax.set_title(f"{name}: {len(labels)} images, {counts.min()}-{counts.max()} per class, "
                     f"imbalance {counts.max() / counts.min():.2f}")
        ax.set_ylabel("images")
    axes[-1].set_xlabel("class id")
    return _save(fig, path)


def plot_image_stats(df: pd.DataFrame, path: Path) -> Path:
    ok = df[df.ok]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col, label in [(axes[0, 0], "width", "width (px)"), (axes[0, 1], "height", "height (px)"),
                           (axes[1, 0], "file_kb", "file size (KB)")]:
        for split, part in ok.groupby("split"):
            ax.hist(part[col], bins=50, alpha=0.6, label=split)
        ax.set_xlabel(label)
        ax.legend()
    aspect = ok.width / ok.height
    axes[0, 2].hist(aspect, bins=60, color="#55a868")
    axes[0, 2].set_xlabel("aspect ratio (w / h)")
    axes[1, 1].scatter(ok.width, ok.height, s=3, alpha=0.3)
    axes[1, 1].set_xlabel("width")
    axes[1, 1].set_ylabel("height")
    modes = ok.groupby(["mode", "split"]).size().unstack(fill_value=0)
    modes.plot.bar(ax=axes[1, 2], logy=True, rot=0)
    axes[1, 2].set_title("image modes (log scale)")
    return _save(fig, path)


def plot_channel_stats(cub: dict, imagenet_mean: list[float], imagenet_std: list[float], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    x = np.arange(3)
    for ax, key, ref in [(axes[0], "mean", imagenet_mean), (axes[1], "std", imagenet_std)]:
        ax.bar(x - 0.2, cub[key], 0.4, label="CUB train")
        ax.bar(x + 0.2, ref, 0.4, label="ImageNet")
        ax.set_xticks(x, ["R", "G", "B"])
        ax.set_title(f"channel {key}")
        ax.legend()
    return _save(fig, path)


def _decode(hf: HFDataset, i: int) -> Image.Image:
    return hf[i]["image"].convert("RGB")


def plot_sample_grid(hf: HFDataset, labels: list[int], class_names: list[str], path: Path,
                     seed: int, n: int = 24) -> Path:
    rng = random.Random(seed)
    first = {}
    for i, y in enumerate(labels):
        first.setdefault(y, i)
    classes = sorted(rng.sample(sorted(first), n))
    fig, axes = plt.subplots(4, n // 4, figsize=(18, 12))
    for ax, y in zip(axes.flat, classes):
        ax.imshow(_decode(hf, first[y]))
        ax.set_title(class_names[y], fontsize=9)
        ax.axis("off")
    return _save(fig, path)


def plot_duplicates(pairs: list[dict], images: dict[str, HFDataset], path: Path, max_pairs: int = 8) -> Path:
    shown = pairs[:max_pairs]
    fig, axes = plt.subplots(max(len(shown), 1), 2, figsize=(7, 3.2 * max(len(shown), 1)), squeeze=False)
    if not shown:
        for ax in axes.flat:
            ax.axis("off")
        axes[0, 0].text(0.5, 0.5, "no near-duplicates found", ha="center")
    for row, p in zip(axes, shown):
        for ax, side in zip(row, ("a", "b")):
            ax.imshow(_decode(images[p[f"split_{side}"]], p[f"index_{side}"]))
            verdict = "duplicate" if p["confirmed"] else "false match"
            ax.set_title(f"{p[f'split_{side}']}: {p[f'file_{side}']}\n"
                         f"pHash {p['distance']}, dHash {p['dhash_distance']}: {verdict}", fontsize=8)
            ax.axis("off")
    return _save(fig, path)


def plot_caption_stats(stats: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = Counter(stats["_clean_counts"])
    axes[0].bar([str(k) for k in sorted(counts)], [counts[k] for k in sorted(counts)])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("captions per image (after cleaning)")
    axes[0].set_ylabel("images (log scale)")
    lengths = stats["_lengths"]
    axes[1].hist(lengths, bins=range(0, max(lengths) + 2), color="#c44e52")
    axes[1].axvline(stats["caption_words_p95"], color="k", ls="--", lw=0.8, label="95th percentile")
    axes[1].set_xlabel("words per caption")
    axes[1].legend()
    return _save(fig, path)


def plot_top_words(words: Counter, path: Path, n: int = 30) -> Path:
    top = [(w, c) for w, c in words.most_common() if w not in STOPWORDS][:n]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh([w for w, _ in reversed(top)], [c for _, c in reversed(top)], color="#8172b2")
    ax.set_title(f"top {n} words (stopwords removed)")
    ax.set_xlabel("occurrences")
    return _save(fig, path)


def plot_augmentation_preview(image: Image.Image, cfg: dict, path: Path, n: int = 8) -> Path:
    tf = get_train_transforms(cfg)
    mean, std = get_norm_stats(cfg)
    fig, axes = plt.subplots(1, n + 1, figsize=(2.2 * (n + 1), 2.6))
    axes[0].imshow(image)
    axes[0].set_title("original")
    for ax in axes[1:]:
        ax.imshow(denormalize(tf(image), mean, std).permute(1, 2, 0).numpy())
    for ax in axes:
        ax.axis("off")
    return _save(fig, path)


# --------------------------------------------------------------------- main

def run_eda(cfg: dict, out_dir: str | Path | None = None) -> tuple[dict, list[Path]]:
    """Compute every statistic and figure. Returns ``(summary, figure_paths)``."""
    data_cfg = cfg["data"]
    out = Path(out_dir or Path(cfg.get("output_dir", "outputs")) / "figures" / "eda")
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 42)

    hf = {"train": load_hf_split("train", data_cfg["train_dataset"], data_cfg.get("cache_dir")),
          "test": load_hf_split("test", data_cfg["test_dataset"], data_cfg.get("cache_dir"))}
    class_names = hf["train"].features["label"].names
    labels = {s: list(hf[s]["label"]) for s in hf}
    train_idx, val_idx = get_train_val_indices(cfg, labels["train"])
    figures: list[Path] = []

    # Class balance
    split_labels = {"train": [labels["train"][i] for i in train_idx],
                    "val": [labels["train"][i] for i in val_idx],
                    "test": labels["test"]}
    figures.append(plot_class_distribution(split_labels, out / "class_distribution.png"))
    balance = {}
    for name, ls in split_labels.items():
        counts = np.bincount(ls, minlength=len(class_names))
        balance[name] = {"images": len(ls), "min_per_class": int(counts.min()), "max_per_class": int(counts.max()),
                         "imbalance_ratio": round(float(counts.max() / counts.min()), 3),
                         "smallest_classes": [class_names[c] for c in np.argsort(counts, kind="stable")[:3]]}

    # Image scan: sizes, modes, decode errors, hashes; channel stats on train only (val excluded)
    df_train, channel = scan_images(hf["train"], "train", set(train_idx))
    df_test, _ = scan_images(hf["test"], "test")
    df = pd.concat([df_train, df_test], ignore_index=True)
    df.to_csv(out / "image_stats.csv", index=False)
    figures.append(plot_image_stats(df, out / "image_stats.png"))
    figures.append(plot_channel_stats(channel, data_cfg["imagenet_mean"], data_cfg["imagenet_std"],
                                      out / "channel_stats.png"))
    ok = df[df.ok]
    aspect = ok.width / ok.height
    images = {
        "failed_to_decode": df.loc[~df.ok, ["split", "file_name", "error"]].to_dict("records"),
        "modes": {f"{split}/{mode}": int(n) for (split, mode), n in ok.groupby(["split", "mode"]).size().items()},
        "non_rgb_files": ok.loc[ok["mode"] != "RGB", ["split", "file_name", "mode"]].to_dict("records"),
        "width": ok.width.describe().round(1).to_dict(),
        "height": ok.height.describe().round(1).to_dict(),
        "aspect_ratio": {"min": round(float(aspect.min()), 3), "median": round(float(aspect.median()), 3),
                         "max": round(float(aspect.max()), 3)},
        "file_kb": ok.file_kb.describe().round(1).to_dict(),
        "smaller_than_224": int(((ok.width < data_cfg["image_size"]) | (ok.height < data_cfg["image_size"])).sum()),
    }

    # Near-duplicates within train and across train/test
    ok_train, ok_test = df_train[df_train.ok], df_test[df_test.ok]
    dup_rows = []
    for (split_a, part_a), (split_b, part_b) in [(("train", ok_train), ("train", None)),
                                                 (("train", ok_train), ("test", ok_test))]:
        found = find_near_duplicates(list(part_a.phash), None if part_b is None else list(part_b.phash))
        part_b = part_a if part_b is None else part_b
        for i, j, d in found:
            a, b = part_a.iloc[i], part_b.iloc[j]
            dhash_dist = imagehash.hex_to_hash(a.dhash) - imagehash.hex_to_hash(b.dhash)
            dup_rows.append({"split_a": split_a, "index_a": int(a["index"]), "file_a": a.file_name,
                             "label_a": class_names[a.label], "split_b": split_b, "index_b": int(b["index"]),
                             "file_b": b.file_name, "label_b": class_names[b.label], "distance": d,
                             "dhash_distance": dhash_dist, "confirmed": dhash_dist <= DUP_CONFIRM_DHASH})
    dups = pd.DataFrame(dup_rows, columns=["split_a", "index_a", "file_a", "label_a", "split_b", "index_b",
                                           "file_b", "label_b", "distance", "dhash_distance", "confirmed"])
    dups = dups.sort_values(["confirmed", "distance"], ascending=[False, True])
    dups.to_csv(out / "duplicates.csv", index=False)
    figures.append(plot_duplicates(dups.to_dict("records"), hf, out / "duplicates.png"))
    confirmed = dups[dups.confirmed]
    duplicates = {
        "phash_max_hamming": DUP_MAX_HAMMING,
        "dhash_confirm_max": DUP_CONFIRM_DHASH,
        "candidate_pairs": len(dups),
        "train_train_pairs": int((confirmed.split_b == "train").sum()),
        "train_test_pairs": int((confirmed.split_b == "test").sum()),
        "cross_label_pairs": int((confirmed.label_a != confirmed.label_b).sum()),
        "confirmed": confirmed[["file_a", "split_b", "file_b"]].to_dict("records"),
    }

    # Captions
    cap, words, rare = caption_stats(list(hf["train"]["description"]))
    test_counts = Counter(len(split_description(d)) for d in hf["test"]["description"])
    rare.to_csv(out / "rare_words.csv", index=False)
    figures.append(plot_caption_stats(cap, out / "caption_stats.png"))
    figures.append(plot_top_words(words, out / "top_words.png"))
    cap = {k: v for k, v in cap.items() if not k.startswith("_")}
    cap["test_captions_per_image"] = dict(sorted(test_counts.items()))
    cap["rare_word_examples"] = rare.word.head(25).tolist()

    figures.append(plot_sample_grid(hf["train"], labels["train"], class_names, out / "sample_grid.png", seed))
    preview_idx = random.Random(seed).choice(train_idx)
    figures.append(plot_augmentation_preview(_decode(hf["train"], preview_idx), cfg, out / "augmentation_preview.png"))

    summary = {"class_balance": balance, "images": images, "channel_stats_train": channel,
               "imagenet": {"mean": data_cfg["imagenet_mean"], "std": data_cfg["imagenet_std"]},
               "duplicates": duplicates, "captions": cap}
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary, figures
