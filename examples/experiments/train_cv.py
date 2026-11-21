"""Compare backbones by repeated, session-aware cross-validation on the TRAINING images; fit the best on all of them and
score it once on the test images (Colab).

Protocol (written before the run):
  - Cross-validation uses the training pool only.  The test images take no part in any choice.
  - Folds keep the images of one capture session together (stats_utils.group_stratified_folds) and put every class in
    every fold; K folds, REPEATS repeats with different seeds.  If that is impossible the images are split without the
    grouping and the results say so.
  - Per backbone: out-of-fold probabilities averaged over the repeats give the metrics and their 95% bootstrap
    intervals (resampling images); the spread over repeats is reported too.
  - Reference rows: the majority class, and a linear classifier on the frozen embeddings of the first backbone.
  - Selection: the highest mean balanced accuracy over the repeats; ties go to macro-F1.
  - Final model: the selected backbone fitted on all training images (EPOCHS), exported (model.pt, model.onnx), and
    scored once on the test images (test_results.json, with every prediction so the Jetson can be compared with it).

Settings (env): BACKBONES (timm names, comma separated)  K=5  REPEATS=3  EPOCHS=15  BATCH=16  LR=0.0005  IMGSZ=224
                SEED=42  BOOTSTRAP=1000, and the data settings of data_source.py.
Writes cv_results.json, selected.json, test_results.json, training_history.json, model.pt, model.onnx.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _dir in (Path.cwd(), Path("/content")):  # a Colab kernel does not always put the script's folder on sys.path
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import numpy as np
from PIL import Image

import cvlib
import data_source
import stats_utils as su

ENV = os.environ.get
BACKBONES = [b.strip() for b in ENV("BACKBONES", "efficientvit_b0.r224_in1k,mobilenetv3_large_100,resnet50,convnext_tiny").split(",") if b.strip()]
K, REPEATS, EPOCHS = int(ENV("K", "5")), int(ENV("REPEATS", "3")), int(ENV("EPOCHS", "15"))
BATCH, LR, IMGSZ, SEED, BOOT = int(ENV("BATCH", "16")), float(ENV("LR", "0.0005")), int(ENV("IMGSZ", "224")), int(ENV("SEED", "42")), int(ENV("BOOTSTRAP", "1000"))


def load(records, base: Path):
    return [Image.open(base / r["path"]).convert("RGB") for r in records]


def boot_metric(y, probs, fn, rng) -> list[float]:
    """95% bootstrap interval of a metric of (y, probs), resampling images."""
    vals = []
    for _ in range(BOOT):
        i = rng.integers(0, len(y), len(y))
        v = fn(y[i], probs[i])
        if not np.isnan(v):
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def summarise(y, rep_probs: list[np.ndarray], rng) -> dict:
    per_rep = [cvlib.metrics(y, p) for p in rep_probs]
    avg = np.mean(rep_probs, axis=0)
    pooled = cvlib.metrics(y, avg)
    row = {}
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "ece"):
        vals = [m[key] for m in per_rep]
        row[key] = {"mean": round(float(np.mean(vals)), 4), "sd_over_repeats": round(float(np.std(vals)), 4)}
    row["accuracy"]["ci95"] = boot_metric(y, avg, lambda a, b: float((b.argmax(1) == a).mean()), rng)
    row["balanced_accuracy"]["ci95"] = boot_metric(y, avg, lambda a, b: cvlib.metrics(a, b)["balanced_accuracy"], rng)
    row["per_class_recall"] = [round(r, 4) for r in pooled["per_class_recall"]]
    row["confusion_matrix"] = pooled["confusion_matrix"]
    return row


def main() -> None:
    data = data_source.prepare()
    info, base = data["info"], Path(data["train_dir"]).parent
    classes = info["classes"]
    train = sorted((r for r in data["records"] if r["split"] == "train"), key=lambda r: r["path"])
    test = sorted((r for r in data["records"] if r["split"] == "test"), key=lambda r: r["path"])
    y = np.array([classes.index(r["class"]) for r in train])
    y_test = np.array([classes.index(r["class"]) for r in test])
    groups = np.array([r["group"] for r in train])
    imgs, imgs_test = load(train, base), load(test, base)
    rng = np.random.default_rng(SEED)
    print(f"{len(train)} training images, {len(test)} test images, {len(set(groups))} capture sessions, device {cvlib.DEVICE}", flush=True)

    grouped = True

    def folds(rep: int) -> np.ndarray:
        nonlocal grouped
        try:
            return su.group_stratified_folds(y, groups, K, SEED + rep) if grouped else su.group_stratified_folds(y, np.arange(len(y)).astype(str), K, SEED + rep)
        except ValueError:
            grouped = False
            print("session-aware folds are impossible here: splitting by image", flush=True)
            return su.group_stratified_folds(y, np.arange(len(y)).astype(str), K, SEED + rep)

    fold_sets = [folds(r) for r in range(REPEATS)]
    fold_info = [{"repeat": r, "class_counts_per_fold": [np.bincount(y[f == k], minlength=len(classes)).tolist() for k in range(K)],
                  "sessions_split_across_folds": int(sum(len(set(f[groups == g])) > 1 for g in set(groups)))} for r, f in enumerate(fold_sets)]
    results: dict = {"classes": classes, "n_train": len(train), "n_test": len(test), "k": K, "repeats": REPEATS, "epochs": EPOCHS,
                     "session_aware_folds": grouped, "folds": fold_info, "backbones": {}, "references": {}}
    curves: dict[str, dict] = {}
    for arch in BACKBONES:
        t0, oof, cur = time.time(), [np.zeros((len(y), len(classes))) for _ in range(REPEATS)], []
        for r, f in enumerate(fold_sets):
            for k in range(K):
                tr, va = np.where(f != k)[0], np.where(f == k)[0]
                model = cvlib.make_model(arch, len(classes))
                hist = cvlib.fit(model, cvlib.ImageList([imgs[i] for i in tr], y[tr], IMGSZ, train=True, seed=SEED + r),
                                 cvlib.ImageList([imgs[i] for i in va], y[va], IMGSZ), EPOCHS, LR, BATCH, SEED + r * 100 + k)
                oof[r][va] = cvlib.predict(model, [imgs[i] for i in va], IMGSZ)
                cur.append(hist)
                del model
        results["backbones"][arch] = {**summarise(y, oof, rng), "seconds": round(time.time() - t0)}
        curves[arch] = {key: np.mean([h[key] for h in cur], axis=0).round(5).tolist() for key in cur[0]}
        s = results["backbones"][arch]
        print(f"{arch}: balanced accuracy {s['balanced_accuracy']['mean']:.3f} {s['balanced_accuracy']['ci95']}  macro-F1 {s['macro_f1']['mean']:.3f}  ({s['seconds']} s)", flush=True)

    majority = np.zeros((len(y), len(classes)))
    majority[:, np.bincount(y).argmax()] = 1.0
    results["references"]["majority_class"] = summarise(y, [majority], rng)
    emb = cvlib.embeddings(BACKBONES[0], imgs, IMGSZ)
    lp = np.zeros((len(y), len(classes)))
    for k in range(K):
        tr, va = np.where(fold_sets[0] != k)[0], np.where(fold_sets[0] == k)[0]
        lp[va] = cvlib.linear_probe(emb[tr], y[tr], emb[va], len(classes))
    results["references"][f"linear_probe_on_frozen_{BACKBONES[0]}"] = summarise(y, [lp], rng)

    best = max(BACKBONES, key=lambda a: (results["backbones"][a]["balanced_accuracy"]["mean"], results["backbones"][a]["macro_f1"]["mean"]))
    results["selected"] = best
    results["selection_rule"] = "highest mean balanced accuracy over the repeats, ties to macro-F1; the test images were not used"
    Path("cv_results.json").write_text(json.dumps(results, indent=2))
    Path("selected.json").write_text(json.dumps({"arch": best, "classes": classes, "epochs": EPOCHS, "imgsz": IMGSZ, "preprocess": os.environ.get("PREPROCESS", "squash")}, indent=2))
    print("selected:", best, flush=True)

    model = cvlib.make_model(best, len(classes))
    cvlib.fit(model, cvlib.ImageList(imgs, y, IMGSZ, train=True, seed=SEED), None, EPOCHS, LR, BATCH, SEED)
    cvlib.export(model, IMGSZ)
    probs = cvlib.predict(model, imgs_test, IMGSZ)  # the one and only look at the test images
    m = cvlib.metrics(y_test, probs)
    Path("test_results.json").write_text(json.dumps({
        "backbone": best, "n_test": len(test), "classes": classes, "files": [r["name"] for r in test],
        "true": y_test.tolist(), "predictions": probs.argmax(1).tolist(), "probabilities": probs.round(5).tolist(),
        "accuracy": round(m["accuracy"], 4), "accuracy_ci95": boot_metric(y_test, probs, lambda a, b: float((b.argmax(1) == a).mean()), rng),
        "balanced_accuracy": round(m["balanced_accuracy"], 4), "macro_f1": round(m["macro_f1"], 4),
        "per_class_recall": [None if np.isnan(r) else round(r, 4) for r in m["per_class_recall"]], "confusion_matrix": m["confusion_matrix"],
        "note": "scored once, after the backbone was chosen by cross-validation on the training images"}, indent=2))
    c = curves[best]
    best_epoch = int(np.argmax(c["val_acc"])) + 1
    Path("training_history.json").write_text(json.dumps({
        "epochs": list(range(1, EPOCHS + 1)), **c, "best_val_acc": round(max(c["val_acc"]), 5), "best_epoch": best_epoch,
        "classes": classes, "arch": best, "n_train": len(train), "n_val": round(len(train) / K), "device": str(cvlib.DEVICE),
        "curves_are": f"means over {K} folds x {REPEATS} repeats of the cross-validation of the selected backbone"}, indent=2))
    print(json.dumps({"selected": best, "test_accuracy": m["accuracy"], "test_balanced_accuracy": m["balanced_accuracy"]}))


if __name__ == "__main__":
    main()
