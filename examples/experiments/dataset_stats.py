"""Does the background of a blood-smear image carry the class?  Statistics of the training images before any training (Colab).

Reads the training pool of kaggle_data.py (or the zip of class folders named by TRAIN_ZIP).  Each image is segmented into cells and
background (stain_render.segment) and described by
  background   mean and std of L*, a*, b* of the pixels between the cells, and of the scale-bar corner
  cells        mean L*, a*, b* of the cell pixels, and the fraction of the image they cover
Per class and feature it reports the mean and std and a Kruskal-Wallis test across the classes (with eta^2, the share
of variance explained by the class).  Then two classifiers (linear discriminant, 5-fold stratified cross-validation)
are given ONLY the background features, or ONLY the cell features; each accuracy is compared with the majority-class
baseline and with a permutation test (PERMUTATIONS shuffles of the labels).  A background-only accuracy clearly above
the baseline means a network can classify these images without looking at the cells.

Also writes augmentation_examples.png: what the background augmentation (stain_render.TRAIN_PROTOCOLS) does to one
image per class, next to a held-out protocol.

Settings (env): TRAIN_ZIP=""  PERMUTATIONS=500  SEED=7
Writes dataset_stats.json, dataset_stats.png and augmentation_examples.png.
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

for _dir in (Path.cwd(), Path("/content")):  # a Colab kernel does not always put the script's folder on sys.path
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import cv2  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

import stain_render as sr  # noqa: E402

ZIP = os.environ.get("TRAIN_ZIP", "")
PERMUTATIONS = int(os.environ.get("PERMUTATIONS", "500"))
SEED = int(os.environ.get("SEED", "7"))
EXT = (".jpg", ".jpeg", ".png", ".bmp")
BG_FEATURES = ["bg_L", "bg_a", "bg_b", "bg_L_std", "bg_a_std", "bg_b_std", "bar_L", "bar_a", "bar_b"]
CELL_FEATURES = ["cell_L", "cell_a", "cell_b", "cell_coverage"]


def pretty(c: str) -> str:
    return c.replace("Malignant_", "").replace("_", " ")


def load_images() -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    if ZIP:
        root = Path("stats_images")
        if not root.exists():
            with zipfile.ZipFile(ZIP) as z:
                z.extractall(root)
    else:
        import kaggle_data

        root = Path(kaggle_data.prepare()["train_dir"])
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    imgs, labels = [], []
    for i, c in enumerate(classes):
        for f in sorted((root / c).iterdir()):
            if f.suffix.lower() in EXT:
                imgs.append(np.asarray(Image.open(f).convert("RGB")))
                labels.append(i)
    return imgs, np.asarray(labels), classes


def features(rgb: np.ndarray, cells: np.ndarray) -> dict[str, float]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 1:] -= 128  # a*, b* centred on 0
    bg, fg = lab[~cells], lab[cells]
    h, w = cells.shape
    bar = lab[int(sr.SCALE_BAR[1] * h):, int(sr.SCALE_BAR[0] * w):].reshape(-1, 3)
    out = {}
    for name, v in (("L", 0), ("a", 1), ("b", 2)):
        out[f"bg_{name}"] = float(bg[:, v].mean()) if len(bg) else float("nan")
        out[f"bg_{name}_std"] = float(bg[:, v].std()) if len(bg) else float("nan")
        out[f"bar_{name}"] = float(bar[:, v].mean())
        out[f"cell_{name}"] = float(fg[:, v].mean()) if len(fg) else float("nan")
    out["cell_coverage"] = float(cells.mean())
    return out


# ----- statistics ------------------------------------------------------------

def folds(y: np.ndarray, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    parts: list[list[int]] = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for j, i in enumerate(idx):
            parts[j % k].append(int(i))
    return [np.asarray(p) for p in parts]


def lda_cv_accuracy(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, k: int = 5) -> float:
    """Cross-validated accuracy of a linear discriminant (class means, pooled covariance, class priors)."""
    correct = 0
    classes = np.unique(y)
    for test in folds(y, k, rng):
        train = np.setdiff1d(np.arange(len(y)), test)
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-9
        Xt, Xs = (X[train] - mu) / sd, (X[test] - mu) / sd
        means = np.stack([Xt[y[train] == c].mean(0) for c in classes])
        resid = Xt - means[np.searchsorted(classes, y[train])]
        inv = np.linalg.inv(resid.T @ resid / max(1, len(Xt) - len(classes)) + 1e-3 * np.eye(X.shape[1]))
        prior = np.log(np.array([(y[train] == c).mean() for c in classes]))
        score = Xs @ inv @ means.T - 0.5 * np.einsum("kd,de,ke->k", means, inv, means) + prior
        correct += int((classes[score.argmax(1)] == y[test]).sum())
    return correct / len(y)


def classifier_report(X: np.ndarray, y: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    acc = lda_cv_accuracy(X, y, np.random.default_rng(SEED))
    null = np.array([lda_cv_accuracy(X, rng.permutation(y), np.random.default_rng(SEED)) for _ in range(PERMUTATIONS)])
    return {"cv_accuracy": round(acc, 4), "majority_baseline": round(float(np.bincount(y).max() / len(y)), 4),
            "permutation_null_mean": round(float(null.mean()), 4),
            "permutation_p": round(float((1 + (null >= acc).sum()) / (1 + PERMUTATIONS)), 4), "permutations": PERMUTATIONS}


def chi2_sf(x: float, df: int) -> float:
    """P(chi-square with df degrees of freedom > x): the regularised upper incomplete gamma function Q(df/2, x/2)."""
    import math

    a, x = df / 2.0, x / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1:  # series for the lower function
        term = total = 1.0 / a
        for n in range(1, 500):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    b, c, d = x + 1 - a, 1e300, 1.0 / (x + 1 - a)  # continued fraction (modified Lentz) for the upper function
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        if abs(d * c - 1) < 1e-14:
            break
    return min(1.0, h * math.exp(-x + a * math.log(x) - math.lgamma(a)))


def kruskal(values: np.ndarray, y: np.ndarray) -> dict:
    """Kruskal-Wallis H test across the classes (ties corrected) and eta^2, the share of rank variance the class explains."""
    groups = [values[y == c] for c in np.unique(y)]
    groups = [g[~np.isnan(g)] for g in groups]
    allv = np.concatenate(groups)
    n, k = len(allv), len(groups)
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(n)
    sorted_v = allv[order]
    i = 0
    while i < n:  # average ranks for ties
        j = i
        while j + 1 < n and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    h, start = 0.0, 0
    for g in groups:
        r = ranks[start:start + len(g)]
        h += r.sum() ** 2 / len(g)
        start += len(g)
    h = 12.0 / (n * (n + 1)) * h - 3 * (n + 1)
    _, counts = np.unique(allv, return_counts=True)
    h /= 1 - (counts ** 3 - counts).sum() / (n ** 3 - n)
    return {"H": round(float(h), 3), "p": chi2_sf(h, k - 1), "eta_squared": round(float(max(0.0, (h - k + 1) / (n - k))), 4)}


# ----- figures ---------------------------------------------------------------

def stats_figure(feats: np.ndarray, names: list[str], y: np.ndarray, classes: list[str], report: dict, out: Path) -> None:
    fig = plt.figure(figsize=(10.5, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5], wspace=0.55)
    for j, (feat, title) in enumerate((("bg_L", "background L*"), ("bg_a", "background a*"), ("bg_b", "background b*"))):
        ax = fig.add_subplot(gs[0, j])
        col = feats[:, names.index(feat)]
        ax.boxplot([col[y == c] for c in range(len(classes))], showfliers=False)
        ax.set_xticks(range(1, len(classes) + 1), [pretty(c) for c in classes], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{title}\nKruskal p={report['features'][feat]['p']:.1e}" if report["features"][feat]["p"] else title,
                     fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[0, 3])
    bars = [("majority\nbaseline", report["background_only"]["majority_baseline"], "#999999"),
            ("background\nonly", report["background_only"]["cv_accuracy"], "#dd8452"),
            ("cells\nonly", report["cells_only"]["cv_accuracy"], "#4c72b0")]
    ax.bar(range(3), [b[1] for b in bars], color=[b[2] for b in bars])
    for i, b in enumerate(bars):
        ax.text(i, b[1] + 0.01, f"{b[1]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(3), [b[0] for b in bars], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("5-fold CV accuracy", fontsize=8)
    ax.set_title(f"Linear classifier, permutation p (background) = {report['background_only']['permutation_p']:.3f}", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def augmentation_montage(imgs: list[np.ndarray], masks: list[np.ndarray], y: np.ndarray, classes: list[str], out: Path) -> None:
    cols = [("original", None, False)]
    cols += [(f"bg: {p}", p, False) for p in sr.TRAIN_PROTOCOLS[:2]]
    cols += [(f"stain: {p}", p, True) for p in sr.TRAIN_PROTOCOLS[2:4]]
    cols += [(f"held out: {p}", p, True) for p in sr.HELD_OUT_PROTOCOLS[:2]]
    rows = [int(np.where(y == c)[0][0]) for c in range(len(classes))]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.7 * len(cols), 1.7 * len(rows)))
    for r, i in enumerate(rows):
        for c, (label, proto, stain) in enumerate(cols):
            img = imgs[i] if proto is None else sr.render(imgs[i], masks[i], sr.PROTOCOLS[proto], 100 + i, stain)
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(label, fontsize=6)
        axes[r, 0].text(-0.05, 0.5, pretty(classes[y[i]]), transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=6)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    imgs, y, classes = load_images()
    masks = [sr.segment(im) for im in imgs]
    names = BG_FEATURES + CELL_FEATURES
    rows = [features(im, m) for im, m in zip(imgs, masks)]
    feats = np.array([[r[n] for n in names] for r in rows])
    ok = ~np.isnan(feats).any(axis=1)  # an image with no background or no cells cannot be described
    feats, y_ok = feats[ok], y[ok]
    report: dict = {
        "n_images": int(len(y)), "n_described": int(ok.sum()), "classes": classes,
        "images_per_class": {c: int((y == i).sum()) for i, c in enumerate(classes)},
        "image_size": list(imgs[0].shape[:2]),
        "split": __import__("kaggle_data").prepare()["info"] if not ZIP else None,
        "mean_cell_coverage": round(float(np.mean([m.mean() for m in masks])), 4),
        "per_class": {}, "features": {},
        "protocols": {"train": sr.TRAIN_PROTOCOLS, "held_out": sr.HELD_OUT_PROTOCOLS, "parameters": sr.PROTOCOLS},
    }
    for i, c in enumerate(classes):
        report["per_class"][c] = {n: {"mean": round(float(feats[y_ok == i, j].mean()), 3),
                                      "std": round(float(feats[y_ok == i, j].std()), 3)} for j, n in enumerate(names)}
    for j, n in enumerate(names):
        report["features"][n] = kruskal(feats[:, j], y_ok)
    report["background_only"] = classifier_report(feats[:, : len(BG_FEATURES)], y_ok)
    report["cells_only"] = classifier_report(feats[:, len(BG_FEATURES):], y_ok)
    b, base = report["background_only"], report["background_only"]["majority_baseline"]
    report["verdict"] = (
        "the background alone predicts the class: a network could classify these images without the cells"
        if b["cv_accuracy"] > base + 0.10 and b["permutation_p"] < 0.05
        else "the background carries little class information beyond the majority baseline")
    Path("dataset_stats.json").write_text(json.dumps(report, indent=2))
    stats_figure(feats, names, y_ok, classes, report, Path("dataset_stats.png"))
    augmentation_montage(imgs, masks, y, classes, Path("augmentation_examples.png"))
    print(json.dumps({k: report[k] for k in ("n_images", "background_only", "cells_only", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
