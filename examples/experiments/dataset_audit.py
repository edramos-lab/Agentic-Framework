"""What does a photograph carry besides the subject?  An audit of the training images before any model is trained (Colab).

For every training image (the test images are not looked at):
  background  mean and std of L*, a*, b* of the pixels outside the person mask
  person      mean L*, a*, b* inside the mask, the share of the image it covers, the bounding box's height/width
              and the vertical position of its centre
  metadata    the original width and height, the file size, and the capture date from the file name
Per class and feature: mean, std, and a Kruskal-Wallis test across the classes with eta^2 (the share of rank variance the
class explains).  Then linear classifiers (cross-validated with session-aware folds) are given ONLY the background features,
ONLY the person features, ONLY the metadata, or all of them, each against the majority baseline and a label-permutation
null.  Accuracy clearly above the baseline from the background or the metadata means a network could classify these images
without looking at the subject.  The report also says whether the official split shares capture sessions between train and
test.  No photograph is drawn: the figure shows statistics only.

The person mask is a pretrained segmentation network's (person_mask.py).

Settings (env): PERMUTATIONS=500  K=5  SEED=7, and the data settings of data_source.py.
Writes dataset_audit.json and dataset_audit.png.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

for _dir in (Path.cwd(), Path("/content")):  # a Colab kernel does not always put the script's folder on sys.path
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_source
import person_mask
import stats_utils as su

PERMUTATIONS, K, SEED = int(os.environ.get("PERMUTATIONS", "500")), int(os.environ.get("K", "5")), int(os.environ.get("SEED", "7"))
BG = ["bg_L", "bg_a", "bg_b", "bg_L_std", "bg_a_std", "bg_b_std"]
PERSON = ["person_L", "person_a", "person_b", "person_fraction", "person_box_aspect", "person_center_y"]
META = ["orig_width", "orig_height", "log_file_bytes", "capture_day"]


def features(rgb: np.ndarray, mask: np.ndarray, rec: dict) -> dict[str, float]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 1:] -= 128
    bg, fg = lab[~mask], lab[mask]
    out = {}
    for j, n in enumerate("Lab"):
        out[f"bg_{n}"] = float(bg[:, j].mean()) if len(bg) else float("nan")
        out[f"bg_{n}_std"] = float(bg[:, j].std()) if len(bg) else float("nan")
    for j, n in enumerate("Lab"[:3]):
        out[f"person_{n}"] = float(fg[:, j].mean()) if len(fg) else float("nan")
    ys, xs = np.where(mask)
    out["person_fraction"] = float(mask.mean())
    out["person_box_aspect"] = float((ys.max() - ys.min() + 1) / (xs.max() - xs.min() + 1)) if len(ys) else float("nan")
    out["person_center_y"] = float(ys.mean() / mask.shape[0]) if len(ys) else float("nan")
    out["orig_width"], out["orig_height"] = float(rec["orig_w"]), float(rec["orig_h"])
    out["log_file_bytes"] = float(np.log(rec["bytes"]))
    g = rec["group"]
    try:
        out["capture_day"] = float(datetime.date(int(g[:4]), int(g[4:6]), int(g[6:8])).toordinal()) if g.isdigit() and len(g) == 8 else float("nan")
    except ValueError:
        out["capture_day"] = float("nan")
    return out


def main() -> None:
    data = data_source.prepare()
    info, base = data["info"], Path(data["train_dir"]).parent
    classes = info["classes"]
    train = sorted((r for r in data["records"] if r["split"] == "train"), key=lambda r: r["path"])
    y = np.array([classes.index(r["class"]) for r in train])
    groups = np.array([r["group"] for r in train])
    masks = person_mask.person_masks([r["path"] for r in train], base)
    names = BG + PERSON + META
    rows = [features(np.asarray(Image.open(base / r["path"]).convert("RGB")), masks[r["path"]], r) for r in train]
    feats = np.array([[row[n] for n in names] for row in rows])
    # a feature that is missing for some image (no capture date in the name, no person found) takes the column median
    n_missing = {n: int(np.isnan(feats[:, j]).sum()) for j, n in enumerate(names) if np.isnan(feats[:, j]).any()}
    feats = np.where(np.isnan(feats), np.nanmedian(feats, axis=0), feats)
    feats = np.nan_to_num(feats, nan=0.0)  # a feature missing for every image (no date in any name) is constant
    day = names.index("capture_day")
    feats[:, day] -= feats[:, day].min()  # days since the earliest capture in the training images
    try:
        fold_of = su.group_stratified_folds(y, groups, K, SEED)
        grouped = True
    except ValueError:
        fold_of, grouped = su.group_stratified_folds(y, np.arange(len(y)).astype(str), K, SEED), False
    sets = {"background_only": BG, "person_only": PERSON, "metadata_only": META, "all_features": names}
    report: dict = {
        "n_images": len(y), "classes": classes, "images_per_class": {c: int((y == i).sum()) for i, c in enumerate(classes)},
        "n_capture_sessions": len(set(groups)), "session_aware_folds": grouped, "images_without_person_mask": int(sum(masks[r["path"]].mean() < 0.02 for r in train)),
        "features_missing_and_filled_with_median": n_missing, "split": info, "per_class": {}, "features": {},
    }
    for i, c in enumerate(classes):
        report["per_class"][c] = {n: {"mean": round(float(feats[y == i, j].mean()), 3), "std": round(float(feats[y == i, j].std()), 3)} for j, n in enumerate(names)}
    for j, n in enumerate(names):
        report["features"][n] = su.kruskal(feats[:, j], y)
    report["classifiers"] = {k: su.classifier_report(feats[:, [names.index(n) for n in cols]], y, fold_of, PERMUTATIONS, SEED) for k, cols in sets.items()}
    base_acc = report["classifiers"]["background_only"]["majority_baseline"]
    flagged = [k for k in ("background_only", "metadata_only") if report["classifiers"][k]["cv_accuracy"] > base_acc + 0.10 and report["classifiers"][k]["permutation_p"] < 0.05]
    report["verdict"] = ("class information is available without the subject: " + " and ".join(k.replace("_", " ") for k in flagged)
                         if flagged else "the background and the metadata carry little class information beyond the majority baseline")
    g = info.get("groups", {})
    report["split_note"] = (f"{g.get('test_images_on_a_shared_group', 0)} test images were captured on a day that also has training images "
                            f"({len(g.get('shared_by_train_and_test', []))} shared days): the official split is not session-disjoint" if g.get("shared_by_train_and_test")
                            else "the official split does not share capture sessions between train and test")
    Path("dataset_audit.json").write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(1, 5, figsize=(12, 3.0), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.5], "wspace": 0.5})
    short = [c.replace("Genu-", "").replace("Asymmetric", "Asym.") for c in classes]
    for ax, (n, label) in zip(axes[:4], (("bg_L", "background L*"), ("person_fraction", "person area share"), ("log_file_bytes", "log file size"), ("capture_day", "days since first capture"))):
        ax.boxplot([feats[y == i, names.index(n)] for i in range(len(classes))], showfliers=False)
        ax.set_xticks(range(1, len(classes) + 1), short, rotation=45, ha="right", fontsize=7)
        p = report["features"][n]["p"]
        ax.set_title(f"{label}\nKruskal p={p:.1e}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    ax = axes[4]
    cl = report["classifiers"]
    labels = ["majority", "background", "person", "metadata", "all"]
    vals = [base_acc] + [cl[k]["cv_accuracy"] for k in sets]
    ax.bar(range(5), vals, color=["#999999", "#dd8452", "#4c72b0", "#8172b3", "#55a868"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(5), labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("CV accuracy (linear classifier)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig("dataset_audit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"classifiers": {k: (v["cv_accuracy"], v["permutation_p"]) for k, v in cl.items()}, "majority": base_acc, "verdict": report["verdict"], "split_note": report["split_note"]}, indent=2))


if __name__ == "__main__":
    main()
