"""Leg geometry from body keypoints, before any model is trained: does it separate the classes, and does detection itself?  (Colab)

A pretrained keypoint detector (torchvision Keypoint R-CNN, the COCO 17-keypoint family that PoseNet belongs to) is run on every
TRAINING image.  The test images are not looked at.  From the most confident person it keeps the hips, knees and ankles that it
finds (confidence >= KP_SCORE) and measures, normalised by the lower-leg length so that the camera distance and position drop out:
  knee_gap_over_leg, ankle_gap_over_leg   horizontal distance between the knees, between the ankles
  knee_over_ankle_gap                     large for knees far apart with ankles together (varus), small the other way (valgus)
  tibia_inward_tilt                       mean tilt of the knee->ankle line from the vertical, positive when the ankle is closer
                                          to the body's midline than the knee (varus > 0, valgus < 0)
  leg_length_asymmetry, tilt_asymmetry    left-right differences
  knee_angle_mean, knee_angle_asymmetry   the hip-knee-ankle angle, only when the hips are visible
Image left and right are used (a photograph from behind swaps the person's sides); every feature is symmetric in them.

Why also the detection: a close-up of the lower legs has no hips and may have no detectable knees, a full-body shot has all of them,
so WHETHER a keypoint is found can follow the framing of the photograph.  The audit therefore reports, per class, how often each
keypoint is found (with a chi-square test of class vs availability) and a classifier given only the availability, next to the
classifiers given the geometry (missing values filled with the median, and complete cases only).  Cross-validation uses the
session-aware folds of the other experiments; every classifier is compared with the majority baseline and a permutation null.

Verdict rule (written before the run): a classifier "separates the classes" when its cross-validated accuracy exceeds the
majority baseline OF THE IMAGES IT USED by more than 0.10 and its permutation p is below 0.05; detectability "depends on the
class" when the chi-square p is below 0.05.

Detector limits: it is trained on whole, upright bodies in everyday scenes, has no heel or foot points, and its keypoint error of a
few pixels is comparable to the few degrees by which varus and valgus differ from normal.  Nothing here is a clinical measurement.

Settings (env): KP_SCORE=1.0 (keypoint confidence, the detector's heat-map score)  PERSON_SCORE=0.5  PERMUTATIONS=500  K=5  SEED=7
Writes pose_audit.json and pose_audit.png (statistics only: no photograph is drawn).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

for _dir in (Path.cwd(), Path("/content")):  # a Colab kernel does not always put the script's folder on sys.path
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_source
import stats_utils as su

KP_SCORE, PERSON_SCORE = float(os.environ.get("KP_SCORE", "1.0")), float(os.environ.get("PERSON_SCORE", "0.5"))
PERMUTATIONS, K, SEED = int(os.environ.get("PERMUTATIONS", "500")), int(os.environ.get("K", "5")), int(os.environ.get("SEED", "7"))
HIP, KNEE, ANKLE = (11, 12), (13, 14), (15, 16)  # COCO keypoint indices: left/right hip, knee, ankle
GEOMETRY = ["knee_gap_over_leg", "ankle_gap_over_leg", "knee_over_ankle_gap", "tibia_inward_tilt", "leg_length_asymmetry", "tilt_asymmetry"]
WITH_HIPS = ["knee_angle_mean", "knee_angle_asymmetry"]
_model = None


def detector():
    global _model
    if _model is None:
        import torch
        from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights as W
        from torchvision.models.detection import keypointrcnn_resnet50_fpn

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _model = (keypointrcnn_resnet50_fpn(weights=W.DEFAULT).eval().to(dev), dev)
    return _model


def keypoints(img: Image.Image) -> dict | None:
    """Keypoints of the most confident person: {index: (x, y)} for those found, and the raw scores; None without a person."""
    import torch
    from torchvision.transforms.functional import to_tensor

    model, dev = detector()
    with torch.no_grad():
        out = model([to_tensor(img.convert("RGB")).to(dev)])[0]
    if len(out["scores"]) == 0 or float(out["scores"][0]) < PERSON_SCORE:
        return None
    pts, sc = out["keypoints"][0].cpu().numpy(), out["keypoints_scores"][0].cpu().numpy()
    return {"found": {i: (float(pts[i, 0]), float(pts[i, 1])) for i in range(17) if sc[i] >= KP_SCORE}, "scores": sc.tolist()}


def side(pair: tuple[int, int], found: dict) -> list | None:
    """The two points of a left/right pair ordered by image x (smaller x first), or None unless both were found."""
    if pair[0] in found and pair[1] in found:
        return sorted([found[pair[0]], found[pair[1]]], key=lambda p: p[0])
    return None


def geometry(found: dict) -> dict[str, float]:
    kn, an, hp = side(KNEE, found), side(ANKLE, found), side(HIP, found)
    out = {n: float("nan") for n in GEOMETRY + WITH_HIPS}
    if not (kn and an):
        return out
    # the knee and the ankle of one leg: the nearer pair of the two knees and two ankles in x
    legs = [(kn[0], an[0]), (kn[1], an[1])]
    lens = [math.hypot(k[0] - a[0], k[1] - a[1]) for k, a in legs]
    leg = float(np.mean(lens))
    if leg < 1e-6:
        return out
    knee_gap, ankle_gap = abs(kn[1][0] - kn[0][0]), abs(an[1][0] - an[0][0])
    # tilt of knee->ankle from the vertical; positive when the ankle is nearer the midline than the knee
    tilts = [math.degrees(math.atan2(a[0] - k[0], max(a[1] - k[1], 1e-6))) * (1 if i == 0 else -1) for i, (k, a) in enumerate(legs)]
    out.update({"knee_gap_over_leg": knee_gap / leg, "ankle_gap_over_leg": ankle_gap / leg,
                "knee_over_ankle_gap": knee_gap / max(ankle_gap, 0.05 * leg), "tibia_inward_tilt": float(np.mean(tilts)),
                "leg_length_asymmetry": abs(lens[0] - lens[1]) / leg, "tilt_asymmetry": abs(tilts[0] - tilts[1])})
    if hp:
        angles = []
        for h, (k, a) in zip(hp, legs):
            v1, v2 = np.array(h) - np.array(k), np.array(a) - np.array(k)
            c = float(v1 @ v2 / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-9))
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
        out["knee_angle_mean"], out["knee_angle_asymmetry"] = float(np.mean(angles)), abs(angles[0] - angles[1])
    return out


def chi_square(table: np.ndarray) -> dict:
    """Pearson chi-square test of independence of a classes x outcome table."""
    exp = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    mask = exp > 0
    stat = float(((table - exp) ** 2 / np.where(mask, exp, 1))[mask].sum())
    df = (table.shape[0] - 1) * (table.shape[1] - 1)
    return {"chi2": round(stat, 3), "df": df, "p": su.chi2_sf(stat, df), "min_expected": round(float(exp.min()), 2)}


def main() -> None:
    data = data_source.prepare()
    base, classes = Path(data["train_dir"]).parent, data["info"]["classes"]
    train = sorted((r for r in data["records"] if r["split"] == "train"), key=lambda r: r["path"])
    y = np.array([classes.index(r["class"]) for r in train])
    groups = np.array([r["group"] for r in train])
    dets = [keypoints(Image.open(base / r["path"]).convert("RGB")) for r in train]
    found = [d["found"] if d else {} for d in dets]
    feats = np.array([[geometry(f)[n] for n in GEOMETRY + WITH_HIPS] for f in found])
    names = GEOMETRY + WITH_HIPS
    avail = np.array([[d is not None, side(KNEE, f) is not None, side(ANKLE, f) is not None, side(HIP, f) is not None]
                      for d, f in zip(dets, found)], dtype=float)
    try:
        fold_of, grouped = su.group_stratified_folds(y, groups, K, SEED), True
    except ValueError:
        fold_of, grouped = su.group_stratified_folds(y, np.arange(len(y)).astype(str), K, SEED), False

    labels = ["person_detected", "both_knees", "both_ankles", "both_hips"]
    report: dict = {
        "detector": "torchvision keypointrcnn_resnet50_fpn (COCO 17 keypoints)", "keypoint_score_threshold": KP_SCORE, "person_score_threshold": PERSON_SCORE,
        "n_images": len(y), "classes": classes, "images_per_class": {c: int((y == i).sum()) for i, c in enumerate(classes)}, "session_aware_folds": grouped,
        "keypoint_score_quantiles": {n: [round(float(q), 2) for q in np.quantile([d["scores"][i] for d in dets if d], [0.1, 0.5, 0.9])]
                                     for n, i in (("hip", 11), ("knee", 13), ("ankle", 15))} if any(dets) else {},
        "availability": {c: {l: round(float(avail[y == i, j].mean()), 3) for j, l in enumerate(labels)} for i, c in enumerate(classes)},
    }
    lower = ((avail[:, 1] > 0) & (avail[:, 2] > 0)).astype(int)
    report["availability_test"] = {"outcome": "knees and ankles both found", **chi_square(np.array([[int(((y == i) & (lower == v)).sum()) for v in (0, 1)] for i in range(len(classes))], float))}
    report["per_class"], report["features"] = {}, {}
    for i, c in enumerate(classes):
        report["per_class"][c] = {n: ({"mean": round(float(np.nanmean(feats[y == i, j])), 3), "std": round(float(np.nanstd(feats[y == i, j])), 3), "n": int((~np.isnan(feats[y == i, j])).sum())}
                                       if (~np.isnan(feats[y == i, j])).any() else {"mean": None, "std": None, "n": 0}) for j, n in enumerate(names)}
    for j, n in enumerate(names):
        ok = ~np.isnan(feats[:, j])
        if len(np.unique(y[ok])) >= 2 and ok.sum() >= 10:
            report["features"][n] = {**su.kruskal(feats[ok, j], y[ok]), "n": int(ok.sum())}
    cls: dict = {}
    per_class_n = lambda mask: {c: int(((y == i) & mask).sum()) for i, c in enumerate(classes)}
    cls["availability_only"] = {**su.classifier_report(avail, y, fold_of, PERMUTATIONS, SEED), "n": len(y), "n_per_class": per_class_n(np.ones(len(y), bool))}
    geo = [names.index(n) for n in GEOMETRY]
    filled = np.where(np.isnan(feats[:, geo]), np.nanmedian(feats[:, geo], axis=0), feats[:, geo]) if (~np.isnan(feats[:, geo])).any() else None
    if filled is not None and not np.isnan(filled).any():
        cls["geometry_missing_filled_with_median"] = {**su.classifier_report(filled, y, fold_of, PERMUTATIONS, SEED), "n": len(y), "n_per_class": per_class_n(np.ones(len(y), bool))}
    complete = lower > 0
    if complete.sum() >= 30 and len(np.unique(y[complete])) == len(classes):
        for k in (K, 3):
            try:
                f2 = su.group_stratified_folds(y[complete], groups[complete], k, SEED)
                cls["geometry_complete_cases"] = {**su.classifier_report(feats[complete][:, geo], y[complete], f2, PERMUTATIONS, SEED), "n": int(complete.sum()), "folds": k, "n_per_class": per_class_n(complete)}
                break
            except ValueError:
                continue
    report["classifiers"] = cls
    base_rate = report["classifiers"]["availability_only"]["majority_baseline"]
    flag = lambda r: r["cv_accuracy"] > r["majority_baseline"] + 0.10 and r["permutation_p"] < 0.05
    parts = []
    for key, label in (("geometry_complete_cases", "among the images with all lower-leg keypoints"), ("geometry_missing_filled_with_median", "with missing values filled")):
        if key in cls and flag(cls[key]):
            counts = ", ".join(f"{c} {n}" for c, n in cls[key]["n_per_class"].items())
            parts.append(f"the leg geometry separates the classes beyond the majority baseline {label} (images per class: {counts})")
            break
    if report["availability_test"]["p"] < 0.05:
        parts.append(f"whether the knees and ankles are detected depends on the class (chi-square p={report['availability_test']['p']:.3g}), which follows the framing of the photograph")
    if flag(cls["availability_only"]):
        parts.append("a classifier given only which keypoints were found beats the majority baseline")
    report["verdict"] = "; ".join(parts) if parts else "neither the leg geometry nor the detectability of the keypoints separates the classes beyond the majority baseline"
    Path("pose_audit.json").write_text(json.dumps(report, indent=2))

    short = [c.replace("Genu-", "").replace("Asymmetric", "Asym.") for c in classes]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.0), gridspec_kw={"wspace": 0.5})
    w = 0.2
    for j, l in enumerate(labels):
        axes[0].bar(np.arange(len(classes)) + (j - 1.5) * w, [report["availability"][c][l] for c in classes], w, label=l.replace("_", " "))
    axes[0].set_xticks(range(len(classes)), short, fontsize=7)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("keypoints found", fontsize=8)
    axes[0].legend(fontsize=5.5, loc="lower right")
    for ax, n, title in ((axes[1], "knee_over_ankle_gap", "knee gap / ankle gap"), (axes[2], "tibia_inward_tilt", "tibia inward tilt (deg)")):
        ok = ~np.isnan(feats[:, names.index(n)])
        ax.boxplot([feats[(y == i) & ok, names.index(n)] if ((y == i) & ok).any() else [np.nan] for i in range(len(classes))], showfliers=False)
        ax.set_xticks(range(1, len(classes) + 1), short, fontsize=7)
        p = report["features"].get(n, {}).get("p")
        ax.set_title(title + (f"\nKruskal p={p:.1e}" if p is not None else ""), fontsize=8)
    names_c = ["majority"] + list(cls)
    vals = [base_rate] + [cls[k]["cv_accuracy"] for k in cls]
    axes[3].bar(range(len(vals)), vals, color=["#999999"] + ["#dd8452", "#4c72b0", "#55a868"][: len(cls)])
    for i, v in enumerate(vals):
        axes[3].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    axes[3].set_xticks(range(len(vals)), [n.replace("geometry_", "geom. ").replace("_", " ")[:16] for n in names_c], rotation=35, ha="right", fontsize=6)
    axes[3].set_ylim(0, 1.05)
    axes[3].set_title("CV accuracy", fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    fig.savefig("pose_audit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"availability": report["availability"], "availability_test": report["availability_test"],
                      "classifiers": {k: (v["cv_accuracy"], v["permutation_p"], v["n"]) for k, v in cls.items()}, "majority": base_rate, "verdict": report["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
