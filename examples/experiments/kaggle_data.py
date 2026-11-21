"""Download the public blood-smear dataset from Kaggle and split it, the same way in every Colab step.

Dataset: KAGGLE_DATASET (default mehradaria/leukemia, "Acute Lymphoblastic Leukemia (ALL) image dataset"; its
"Original" images, four classes: Benign, early Pre-B, Pre-B, Pro-B).  Every Colab experiment calls prepare(), which

  1. downloads it with the kaggle CLI (credentials: /content/kaggle.json uploaded by the pipeline, or the env vars
     KAGGLE_USERNAME / KAGGLE_KEY) unless it is already on the VM;
  2. renames the classes to the names the Jetson-side code uses (Benign, Malignant_Pre-B, Malignant_Pro-B,
     Malignant_early_Pre-B), which keeps the class order, and so the confusion-matrix labels, unchanged;
  3. drops every image whose bytes are identical to an earlier one, so no exact duplicate can sit on both sides of
     the split (the Jetson shard had 12 such images);
  4. splits each class with a seeded shuffle into TEST_FRACTION test images and the rest, the training pool;
  5. writes the training pool as 256x256 JPEGs (the size the trainer resizes to anyway) and links the test images
     untouched (original resolution), so every step sees the identical split.

The split is by image.  The dataset has no patient identifiers, so images of the same patient can fall on both sides;
the paper has to say so.

Settings (env): KAGGLE_DATASET  SPLIT_SEED=42  TEST_FRACTION=0.15  DATA_ROOT=/content/data
                KAGGLE_RAW_DIR (a folder that already holds the extracted dataset: no download; for tests)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DATASET = os.environ.get("KAGGLE_DATASET", "mehradaria/leukemia")
SEED = int(os.environ.get("SPLIT_SEED", "42"))
TEST_FRACTION = float(os.environ.get("TEST_FRACTION", "0.15"))
ROOT = Path(os.environ.get("DATA_ROOT", "/content/data"))
EXT = (".jpg", ".jpeg", ".png", ".bmp")
SIZE = 256


def canonical_class(name: str) -> str | None:
    """The Jetson-side class name for a Kaggle folder name such as '[Malignant] early Pre-B', or None."""
    words = re.sub(r"[^a-z]+", " ", name.lower()).split()
    if "benign" in words:
        return "Benign"
    if "early" in words:
        return "Malignant_early_Pre-B"
    if "pro" in words:
        return "Malignant_Pro-B"
    if "pre" in words:
        return "Malignant_Pre-B"
    return None


def _credentials() -> None:
    """Put the Kaggle token where the CLI looks for it."""
    target = Path.home() / ".kaggle" / "kaggle.json"
    uploaded = Path("/content/kaggle.json")
    if uploaded.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(uploaded, target)
    elif os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]}))
    if target.exists():
        target.chmod(0o600)


def _download() -> Path:
    override = os.environ.get("KAGGLE_RAW_DIR")
    if override:
        return Path(override)
    raw = ROOT / "raw"
    if (raw / ".done").exists():
        return raw
    _credentials()
    if shutil.which("kaggle") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir(parents=True)
    proc = subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", str(raw), "--unzip"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"kaggle download of {DATASET} failed: {(proc.stderr or proc.stdout).strip()[-400:]}")
    (raw / ".done").write_text(DATASET)
    return raw


def _images_in(d: Path) -> list[Path]:
    return sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in EXT and not f.name.startswith("."))


def find_class_folders(raw: Path) -> dict[str, Path]:
    """Canonical class name -> folder holding that class's original images."""
    best: dict[str, Path] = {}
    best_score = (-1, -1)
    for d in [raw, *[p for p in raw.rglob("*") if p.is_dir() and "__MACOSX" not in p.parts]]:
        kids = {canonical_class(k.name): k for k in d.iterdir() if k.is_dir() and _images_in(k)}
        kids.pop(None, None)
        if len(kids) < 3:
            continue
        # prefer the folder of original images over a "segmented" copy, then the larger one
        score = (0 if "segment" in str(d).lower() else 1, sum(len(_images_in(k)) for k in kids.values()))
        if score > best_score:
            best, best_score = kids, score
    if not best:
        raise RuntimeError(f"no class folders (Benign / early Pre-B / Pre-B / Pro-B) found under {raw}")
    return best


def prepare() -> dict:
    """Download, deduplicate and split once; returns {"train_dir", "test_dir", "info"}."""
    info_file = ROOT / "split_info.json"
    if info_file.exists():
        info = json.loads(info_file.read_text())
        return {"train_dir": str(ROOT / "train256"), "test_dir": str(ROOT / "test"), "info": info}
    import numpy as np
    from PIL import Image

    folders = find_class_folders(_download())
    classes = sorted(folders)
    seen: dict[str, str] = {}
    kept: dict[str, list[Path]] = {c: [] for c in classes}
    dropped = 0
    for c in classes:
        for f in sorted(p for p in folders[c].rglob("*") if p.suffix.lower() in EXT and not p.name.startswith(".")):
            h = hashlib.md5(f.read_bytes()).hexdigest()
            if h in seen:
                dropped += 1
                continue
            seen[h] = c
            kept[c].append(f)
    shutil.rmtree(ROOT / "train256", ignore_errors=True)
    shutil.rmtree(ROOT / "test", ignore_errors=True)
    rng = np.random.default_rng(SEED)
    counts = {"train": {}, "test": {}}
    for c in classes:
        files = kept[c]
        order = rng.permutation(len(files))
        n_test = max(1, int(round(TEST_FRACTION * len(files))))
        test_idx, train_idx = set(order[:n_test].tolist()), order[n_test:].tolist()
        (ROOT / "test" / c).mkdir(parents=True)
        (ROOT / "train256" / c).mkdir(parents=True)
        for i in sorted(test_idx):
            link = ROOT / "test" / c / files[i].name
            try:
                link.symlink_to(files[i].resolve())
            except OSError:
                shutil.copy(files[i], link)
        for i in sorted(train_idx):
            Image.open(files[i]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR).save(
                ROOT / "train256" / c / f"{i:05d}_{files[i].stem}.jpg", "JPEG", quality=95)
        counts["train"][c], counts["test"][c] = len(train_idx), len(test_idx)
    sample = Image.open(next(iter(kept[classes[0]])))
    info = {
        "dataset": DATASET, "split_seed": SEED, "test_fraction": TEST_FRACTION, "classes": classes,
        "images_per_class": {c: len(kept[c]) for c in classes},
        "train_images_per_class": counts["train"], "test_images_per_class": counts["test"],
        "n_train": sum(counts["train"].values()), "n_test": sum(counts["test"].values()),
        "exact_duplicates_dropped": dropped, "original_image_size": list(sample.size), "train_image_size": SIZE,
        "split_by": "image (the dataset has no patient identifiers)",
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    info_file.write_text(json.dumps(info, indent=2))
    return {"train_dir": str(ROOT / "train256"), "test_dir": str(ROOT / "test"), "info": info}
