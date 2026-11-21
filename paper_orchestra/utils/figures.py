"""Draw result figures straight from measured experiment data.

Each renderer reads one experiment's metrics and writes a PDF.  The numbers are never retyped by a model, so a
figure either shows exactly what was measured or is not made at all (a renderer raises NoData).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

VARIANT_LABELS = {
    "pt": "PyTorch", "onnx": "ONNX FP32", "trt_fp32": "TensorRT FP32",
    "trt_fp16": "TensorRT FP16", "trt_int8": "TensorRT INT8",
}
PROBE_LABELS = {
    "original": "original", "cells_masked": "cells masked", "cells_only": "cells only",
    "scale_bar_masked": "scale bar masked", "person_masked": "person masked", "person_only": "person only",
}
BACKBONE_NAMES = {
    "efficientvit_b0.r224_in1k": "EfficientViT-B0", "mobilenetv3_large_100": "MobileNetV3-L", "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt-T", "majority_class": "majority class",
}


class NoData(Exception):
    """The experiment has no measurements this figure needs."""


def pretty(name: str) -> str:
    return name.replace("Malignant_", "").replace("_", " ")


def _valid_variants(metrics: dict) -> dict[str, dict]:
    """Variants that have a trustworthy accuracy report, in a fixed order."""
    out = {}
    for key in VARIANT_LABELS:
        v = (metrics.get("variants") or {}).get(key) or {}
        acc = v.get("accuracy")
        if v.get("status") == "success" and acc and acc.get("valid", True):
            out[key] = acc
    if not out:
        raise NoData("no variant has a valid accuracy report")
    return out


def confusion_matrices(metrics: dict, out: Path) -> None:
    accs = _valid_variants(metrics)
    n = len(accs)
    cols = min(n, 3)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.9 * rows), squeeze=False)
    for ax, (key, acc) in zip(axes.flat, accs.items()):
        cm = np.array(acc["confusion_matrix"])
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        k = len(cm)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
        names = [pretty(c) for c in acc["classes"]]
        ax.set_xticks(range(k), names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(k), names, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.set_title(f"{VARIANT_LABELS[key]} (acc {acc['accuracy']:.3f})", fontsize=8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


STYLES = [("-", 2.4), ("--", 1.8), ("-.", 1.5), (":", 1.8), ((0, (5, 1)), 1.2)]  # identical curves stay visible
COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def _curve(fpr, tpr) -> tuple[np.ndarray, np.ndarray]:
    """A ROC curve always runs from (0, 0) to (1, 1); a grid-sampled one is missing its first vertical segment."""
    x, y = np.asarray(fpr, float), np.asarray(tpr, float)
    if x[0] > 0 or y[0] > 0:
        x, y = np.r_[0.0, x], np.r_[0.0, y]
    if x[-1] < 1 or y[-1] < 1:
        x, y = np.r_[x, 1.0], np.r_[y, 1.0]
    return x, y


def roc_curves(metrics: dict, out: Path) -> None:
    accs = _valid_variants(metrics)
    ref_key = next(iter(accs))
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 4.6))
    inset = a.inset_axes([0.42, 0.10, 0.52, 0.50])
    for n, (key, acc) in enumerate(accs.items()):
        roc = acc["roc"]
        x, y = _curve(roc["fpr"], roc["macro_tpr"])
        ls, lw = STYLES[n % len(STYLES)]
        for ax in (a, inset):
            ax.plot(x, y, ls=ls, lw=lw, color=COLOURS[n % len(COLOURS)],
                    label=f"{VARIANT_LABELS[key]} (AUC {acc['macro_auc']:.3f})")
    a.set_title("Macro-average ROC, every deployed variant", fontsize=10)
    inset.set_xlim(0, 0.2)
    inset.set_ylim(0.8, 1.005)
    inset.grid(alpha=0.3)
    inset.tick_params(labelsize=7)
    inset.set_xlabel("FPR (zoom)", fontsize=7, labelpad=1)
    a.indicate_inset_zoom(inset, edgecolor="0.4")
    roc = accs[ref_key]["roc"]
    for n, (c, d) in enumerate(roc["classes"].items()):
        x, y = _curve(d.get("fpr", roc["fpr"]), d["tpr"])
        b.plot(x, y, lw=2.0, color=COLOURS[n % len(COLOURS)], label=f"{pretty(c)} (AUC {d['auc']:.3f})")
    b.set_title(f"One-vs-rest ROC per class, {VARIANT_LABELS[ref_key]}", fontsize=10)
    for ax in (a, b):
        ax.plot([0, 1], [0, 1], "k:", linewidth=0.9, label="chance")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.03)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("False positive rate", fontsize=9)
        ax.set_ylabel("True positive rate", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    a.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    b.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def probe_bars(metrics: dict, out: Path) -> None:
    """Accuracy of each model under every probe condition, with 95% intervals; hatched bars are stains not used in training."""
    models = metrics.get("models")
    if not models:
        raise NoData("no probe conditions")
    first = next(iter(models.values()))["conditions"]
    held = lambda c: first[c].get("protocol_set") == "held_out"
    order = [c for c in PROBE_LABELS if c in first]
    order += sorted((c for c in first if c.startswith("bg_")), key=lambda c: (held(c), c))
    order += sorted((c for c in first if c.startswith("stain_")), key=lambda c: (held(c), c))
    order += [c for c in first if c not in order]  # conditions a domain adds (grayscale, blur, person_box_crop, ...)
    label = lambda c: PROBE_LABELS.get(c) or c.replace("bg_", "bg: ").replace("stain_", "stain: ").replace("_", " ")
    n = len(models)
    width = 0.8 / n
    colours = ["#8c8c8c", "#4c72b0", "#dd8452", "#55a868"]
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    for k, (name, m) in enumerate(models.items()):
        conds = m["conditions"]
        x = np.arange(len(order)) + (k - (n - 1) / 2) * width
        acc = np.array([conds[c]["accuracy"] for c in order])
        err = np.array([[conds[c]["accuracy"] - conds[c]["ci95"][0], conds[c]["ci95"][1] - conds[c]["accuracy"]] for c in order]).T
        bars = ax.bar(x, acc, width, yerr=err, capsize=1.5, color=colours[k % len(colours)], label=name,
                      error_kw={"linewidth": 0.7})
        for bar, c in zip(bars, order):
            if held(c):
                bar.set_hatch("///")
                bar.set_edgecolor("white")
    base = next(iter(models.values()))["summary"].get("majority_baseline")
    if base is not None:
        ax.axhline(base, color="k", linestyle="--", linewidth=0.9, label=f"majority-class baseline ({base:.2f})")
    ax.set_xticks(range(len(order)), [label(c) for c in order], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy (95% bootstrap interval)", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=n + 1, frameon=False,
              title="hatched bars: stain protocols never seen in training", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def cv_comparison(metrics: dict, out: Path) -> None:
    """Backbones and reference rows from the cross-validation: balanced accuracy (95% interval) and macro-F1."""
    rows = {**metrics.get("backbones", {}), **metrics.get("references", {})}
    if not rows:
        raise NoData("no cross-validation results")
    pretty_name = lambda n: BACKBONE_NAMES.get(n) or ("linear probe, frozen " + BACKBONE_NAMES.get(n.split("frozen_", 1)[-1], n.split("frozen_", 1)[-1]) if n.startswith("linear_probe") else n.replace("_", " "))
    names = [pretty_name(n) + (" *" if n == metrics.get("selected") else "") for n in rows]
    bal = np.array([d["balanced_accuracy"]["mean"] for d in rows.values()])
    err = np.array([[d["balanced_accuracy"]["mean"] - d["balanced_accuracy"]["ci95"][0], d["balanced_accuracy"]["ci95"][1] - d["balanced_accuracy"]["mean"]]
                    if "ci95" in d["balanced_accuracy"] else [0, 0] for d in rows.values()]).T
    f1 = np.array([d["macro_f1"]["mean"] for d in rows.values()])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.0, 0.55 * len(rows) + 1.6))
    ax.barh(y - 0.19, bal, 0.36, xerr=err, capsize=2, color="#4c72b0", label="balanced accuracy (95% CI)", error_kw={"linewidth": 0.8})
    ax.barh(y + 0.19, f1, 0.36, color="#dd8452", label="macro-F1")
    k = len(metrics.get("classes", [])) or 4
    ax.axvline(1 / k, color="k", linestyle="--", linewidth=0.9, label=f"chance ({1 / k:.2f})")
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("out-of-fold score (selected model marked *)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)


def training_curves(metrics: dict, out: Path) -> None:
    h = metrics.get("history", metrics)
    loss = [k for k in ("train_loss", "val_loss") if h.get(k)]
    acc = [k for k in ("train_acc", "val_acc") if h.get(k)]
    if not loss and not acc:
        raise NoData("no training history")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, keys, title in ((axes[0], loss, "Loss"), (axes[1], acc, "Accuracy")):
        for k in keys:
            ys = h[k]
            ax.plot(h.get("epochs") or list(range(1, len(ys) + 1)), ys, label=k.replace("_", " "))
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if keys:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# figure_type -> (renderer, test for "this experiment's metrics can feed it")
RENDERERS: dict[str, tuple[Callable[[dict, Path], None], Callable[[dict], bool]]] = {
    "confusion_matrix": (confusion_matrices, lambda m: any(
        (v or {}).get("accuracy") for v in (m.get("variants") or {}).values())),
    "roc_curve": (roc_curves, lambda m: any(
        (v or {}).get("accuracy") for v in (m.get("variants") or {}).values())),
    "morphology_probe": (probe_bars, lambda m: bool(m.get("models"))),
    "training_curves": (training_curves, lambda m: any(
        k in (m.get("history", m)) for k in ("train_loss", "val_loss", "train_acc", "val_acc"))),
}
RENDERERS["training_curves_baseline"] = RENDERERS["training_curves"]
RENDERERS["shortcut_probe"] = RENDERERS["morphology_probe"]  # the photographs domain's name for the same chart
RENDERERS["cv_comparison"] = (cv_comparison, lambda m: bool(m.get("backbones")))
