"""One entry point for the dataset of any experiment: download, deduplicate, split, cache.

The dataset comes from the form (env vars, set by the pipeline):
  DATASET_SOURCE  kaggle | zenodo | local           (default kaggle)
  DATASET_ID      Kaggle slug, Zenodo record id, or (local) unused
  DATASET_RAW_DIR a folder that already holds the data (no download; also how `local` finds it, and for tests)
  SPLIT           official (use the dataset's own train/ and test/ folders) | random   (default official)
  GROUP_REGEX     a regular expression whose match in a file name is the capture session / subject (default: a date)
  TEST_FRACTION=0.15  SPLIT_SEED=42  CACHE_LONG_SIDE=512  DATA_ROOT=/content/data

kaggle goes through kaggle_data.py unchanged (the leukemia experiments).  zenodo and local produce:
  <DATA_ROOT>/cache/train/<class>/*.jpg and /test/<class>/*.jpg   images resized once to CACHE_LONG_SIDE (aspect kept)
  records                                                        one dict per image (split, class, group, original size, bytes)
  info                                                           counts, duplicates, groups shared by train and test

Exact duplicates are dropped (a train image identical to a test image is dropped from train).  The split is by image;
the groups let experiments keep a capture session together and let the audit report whether the official split shares
sessions between train and test.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

for _dir in (Path.cwd(), Path("/content")):  # a Colab kernel does not always put the script's folder on sys.path
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

SOURCE = os.environ.get("DATASET_SOURCE", "kaggle")
DATASET_ID = os.environ.get("DATASET_ID", "")
SPLIT = os.environ.get("SPLIT", "official")
ROOT = Path(os.environ.get("DATA_ROOT", "/content/data"))
LONG_SIDE = int(os.environ.get("CACHE_LONG_SIDE", "512"))
TEST_FRACTION = float(os.environ.get("TEST_FRACTION", "0.15"))
SEED = int(os.environ.get("SPLIT_SEED", "42"))
EXT = (".jpg", ".jpeg", ".png", ".bmp")


def group_of(name: str) -> str:
    """The capture session of a file: GROUP_REGEX if given, else a date in the name (20250714 or 2022-10-10)."""
    rx = os.environ.get("GROUP_REGEX")
    m = re.search(rx, name) if rx else (re.search(r"(\d{8})", name) or re.search(r"(\d{4})-(\d{2})-(\d{2})", name))
    return "".join(g for g in (m.groups() or (m.group(0),)) if g) if m else "unknown"


def _images(d: Path) -> list[Path]:
    return sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in EXT and not f.name.startswith("."))


def _class_dirs(d: Path) -> dict[str, Path]:
    return {k.name: k for k in sorted(d.iterdir()) if k.is_dir() and _images(k)}


def find_tree(raw: Path) -> tuple[str, dict[str, dict[str, Path]]]:
    """('official', {split: {class: folder}}) when a folder holds train/ and test/ class trees, else ('flat', {'all': ...})."""
    best = None
    for d in [raw, *[p for p in raw.rglob("*") if p.is_dir() and "__MACOSX" not in p.parts]]:
        if (d / "train").is_dir() and (d / "test").is_dir():
            tr, te = _class_dirs(d / "train"), _class_dirs(d / "test")
            if len(tr) >= 2 and set(te) <= set(tr):
                return "official", {"train": tr, "test": te}
        kids = _class_dirs(d)
        if len(kids) >= 2 and (best is None or sum(len(_images(k)) for k in kids.values()) > best[0]):
            best = (sum(len(_images(k)) for k in kids.values()), kids)
    if best is None:
        raise RuntimeError(f"no class folders with images found under {raw}")
    return "flat", {"all": best[1]}


def _raw() -> Path:
    override = os.environ.get("DATASET_RAW_DIR")
    if override:
        return Path(override)
    if SOURCE == "zenodo":
        raw = ROOT / "raw"
        if not (raw / ".done").exists():
            import zenodo_data

            shutil.rmtree(raw, ignore_errors=True)
            zenodo_data.download(DATASET_ID, raw)
            (raw / ".done").write_text(DATASET_ID)
        return raw
    raise RuntimeError(f"DATASET_SOURCE={SOURCE!r} needs DATASET_RAW_DIR (a folder with the data)")


def prepare() -> dict:
    """Download, deduplicate, split and cache once; returns {"train_dir", "test_dir", "info", "records"}."""
    if SOURCE == "kaggle":
        import kaggle_data

        out = kaggle_data.prepare()
        return {**out, "records": []}
    info_file = ROOT / "split_info.json"
    if info_file.exists():
        return {"train_dir": str(ROOT / "cache" / "train"), "test_dir": str(ROOT / "cache" / "test"),
                "info": json.loads(info_file.read_text()), "records": json.loads((ROOT / "records.json").read_text())}
    import numpy as np
    from PIL import Image, ImageOps

    layout, tree = find_tree(_raw())
    classes = sorted({c for part in tree.values() for c in part})
    seen: dict[str, str] = {}
    dropped = {"duplicate_in_train": 0, "train_identical_to_test": 0}
    items: list[dict] = []
    order = ["test", "train"] if layout == "official" else ["all"]  # test first, so a train copy of a test image is the one dropped
    for part in order:
        for cls in classes:
            for f in _images(tree[part][cls]) if cls in tree[part] else []:
                h = hashlib.md5(f.read_bytes()).hexdigest()
                if h in seen:
                    dropped["train_identical_to_test" if seen[h] == "test" and part == "train" else "duplicate_in_train"] += 1
                    continue
                seen[h] = part
                items.append({"split": part, "class": cls, "file": f})
    if layout == "flat":  # no official split: a seeded stratified random one
        rng = np.random.default_rng(SEED)
        for cls in classes:
            idx = [i for i, it in enumerate(items) if it["class"] == cls]
            for i in rng.permutation(idx)[: max(1, round(TEST_FRACTION * len(idx)))]:
                items[i]["split"] = "test"
        for it in items:
            it["split"] = "test" if it["split"] == "test" else "train"
    shutil.rmtree(ROOT / "cache", ignore_errors=True)
    records = []
    for it in items:
        f = it["file"]
        im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        w, h = im.size
        s = LONG_SIDE / max(w, h)
        small = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR) if s < 1 else im
        dest_dir = ROOT / "cache" / it["split"] / it["class"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", f.stem)
        dest = dest_dir / f"{stem}.jpg"
        k = 1
        while dest.exists():
            dest = dest_dir / f"{stem}_{k}.jpg"
            k += 1
        small.save(dest, "JPEG", quality=92)
        records.append({"path": f"{it['split']}/{it['class']}/{dest.name}", "split": it["split"], "class": it["class"],
                        "group": group_of(f.name), "orig_w": w, "orig_h": h, "bytes": f.stat().st_size, "name": f.name})
    train_groups = collections.Counter(r["group"] for r in records if r["split"] == "train")
    test_groups = collections.Counter(r["group"] for r in records if r["split"] == "test")
    shared = sorted(set(train_groups) & set(test_groups) - {"unknown"})
    per = lambda sp: {c: sum(1 for r in records if r["split"] == sp and r["class"] == c) for c in classes}
    info = {
        "dataset": f"{SOURCE}:{DATASET_ID}" if DATASET_ID else SOURCE, "layout": layout,
        "split": "official (the dataset's own train/ and test/ folders)" if layout == "official" else f"random, seed {SEED}, {int(TEST_FRACTION * 100)}% test",
        "classes": classes, "train_images_per_class": per("train"), "test_images_per_class": per("test"),
        "n_train": sum(per("train").values()), "n_test": sum(per("test").values()),
        "exact_duplicates_dropped": dropped, "cache_long_side": LONG_SIDE,
        "original_sizes": [[list(k), v] for k, v in collections.Counter((r["orig_w"], r["orig_h"]) for r in records).most_common(5)],
        "groups": {"n_train_groups": len(train_groups), "n_test_groups": len(test_groups), "shared_by_train_and_test": shared,
                   "test_images_on_a_shared_group": sum(test_groups[g] for g in shared)},
        "split_by": "image; capture sessions (groups) are recorded, and the official split " +
                    ("shares sessions between train and test" if shared else "does not share sessions between train and test"),
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "records.json").write_text(json.dumps(records))
    info_file.write_text(json.dumps(info, indent=2))
    return {"train_dir": str(ROOT / "cache" / "train"), "test_dir": str(ROOT / "cache" / "test"), "info": info, "records": records}
