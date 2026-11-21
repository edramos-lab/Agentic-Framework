"""Probe whether the classifier reads cell morphology or the stain/background, with and without background augmentation.

Every test image is turned into variants and each model re-classifies them:

  original            the image as is
  cells_masked        every cell erased and inpainted: background only, no morphology
  cells_only          cells kept on a neutral grey background: no stain background
  scale_bar_masked    the scale-bar corner erased (a possible shortcut: it is the same on every image)
  bg_<protocol>       cells kept, background replaced by that protocol's typical background
  stain_<protocol>    as bg_<protocol>, and the cells also colour-shifted towards that protocol

The seven protocols are SYNTHETIC approximations of Romanowsky stains (stain_render.py).  Four were used to augment
the training of the "augmented" model (stain_render.TRAIN_PROTOCOLS); three were never used for training
(HELD_OUT_PROTOCOLS).  Only the held-out ones count for robustness verdicts, otherwise the test would measure what the
augmented model was trained on instead of what it learned.

Models (torch backend, Colab): "baseline" = model_baseline.pt (trained without the augmentation) and "augmented" =
model.pt (trained with it); with only model.pt present there is one model, "model".  The triton backend (Jetson,
PROBE_BACKEND=triton) probes one deployed variant (BENCH_PROBE_VARIANT) through Triton.

Decision rule per model (written down before the run).  Let orig = accuracy on the original images, base = the
majority-class share, cells_only / held_stain = accuracy on cells_only / mean over the held-out stain_* conditions:
  inconclusive        orig <= base + 0.10 (nothing to probe)
  morphology-driven   cells_only >= 0.9 * orig AND held_stain >= 0.9 * orig AND cells_masked <= base + 0.10
  background/stain-driven   cells_masked > base + 0.15 OR held_stain < 0.7 * orig
  mixed               otherwise
Comparison of the two models: the augmentation "removed the background dependence" when the baseline is
background/stain-driven and the augmented model is not, or when the augmented model keeps at least 0.10 more of its
accuracy on the held-out stains than the baseline does with a paired-bootstrap interval that excludes 0.

Statistics: 95% bootstrap intervals of every accuracy (BOOTSTRAP resamples of the images); McNemar exact tests of the
paired per-image outcomes, original vs each condition within a model and baseline vs augmented within a condition.

Settings: PROBE_BACKEND (torch | triton; default triton when docker is installed, else torch), the torch backend reads
the test split of kaggle_data.py (or the zip named by PROBE_TEST_ZIP), BENCH_TEST_DIR (triton), PROBE_MODEL / PROBE_BASELINE (torch file names),
BENCH_PROBE_VARIANT (triton), BENCH_PROBE_LIMIT (images per class, default all), BOOTSTRAP=1000.
Writes probe_results.json and probe_examples.png.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

for _dir in (Path.cwd(), Path("/content")):  # a Colab kernel does not always put the script's folder on sys.path
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import stain_render as sr  # noqa: E402
import triton_benchmark as tb  # noqa: E402  (shared helpers: image loading, preprocessing, metrics)
from stain_render import HELD_OUT_PROTOCOLS, NEUTRAL_GREY, PROTOCOLS, SCALE_BAR, TRAIN_PROTOCOLS  # noqa: E402

BACKEND = os.environ.get("PROBE_BACKEND") or ("triton" if shutil.which("docker") else "torch")
PROBE_VARIANT = os.environ.get("BENCH_PROBE_VARIANT", "onnx")
PER_CLASS_LIMIT = int(os.environ.get("BENCH_PROBE_LIMIT", "0"))
BOOTSTRAP = int(os.environ.get("BOOTSTRAP", "1000"))
RNG_SEED = 7


# ----- conditions ------------------------------------------------------------

def conditions(rgb: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    cells = sr.segment(rgb)
    h, w = rgb.shape[:2]
    out: dict[str, np.ndarray] = {"original": rgb}
    out["cells_masked"] = sr.erase(rgb, cells)
    out["cells_only"] = sr.blend(rgb.astype(np.float32), sr.flat_background(rgb.shape, NEUTRAL_GREY, seed), cells)
    bar = np.zeros((h, w), bool)
    bar[int(SCALE_BAR[1] * h):, int(SCALE_BAR[0] * w):] = True
    out["scale_bar_masked"] = sr.erase(rgb, bar)
    for name, p in PROTOCOLS.items():
        out[f"bg_{name}"] = sr.render(rgb, cells, p, seed, stain=False)
        out[f"stain_{name}"] = sr.render(rgb, cells, p, seed, stain=True)
    return out


def protocol_set(condition: str) -> str:
    """'train' or 'held_out' for a bg_*/stain_* condition, '' for the others."""
    kind, _, proto = condition.partition("_")
    if kind not in ("bg", "stain"):
        return ""
    return "held_out" if proto in HELD_OUT_PROTOCOLS else "train" if proto in TRAIN_PROTOCOLS else ""


def montage(rows: list[dict[str, np.ndarray]], names: list[str], labels: list[str], tile: int = 170) -> np.ndarray:
    """A grid with one row per example image and one column per condition; each row is titled with its class."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    th, strip = int(tile * 0.75), 18
    canvas = np.full((len(rows) * (th + strip) + strip, len(names) * tile, 3), 255, np.uint8)
    for j, n in enumerate(names):
        cv2.putText(canvas, n[:22], (j * tile + 4, 13), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    for i, row in enumerate(rows):
        top = strip + i * (th + strip)
        cv2.putText(canvas, "class: " + labels[i], (4, top + 13), font, 0.4, (160, 0, 0), 1, cv2.LINE_AA)
        for j, n in enumerate(names):
            canvas[top + strip - 4: top + strip - 4 + th, j * tile: (j + 1) * tile] = cv2.resize(
                row[n], (tile, th), interpolation=cv2.INTER_AREA)
    return canvas


# ----- statistics ------------------------------------------------------------

def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Exact McNemar test of two paired sets of correct/incorrect outcomes."""
    only_a, only_b = int((a_correct & ~b_correct).sum()), int((~a_correct & b_correct).sum())
    n = only_a + only_b
    if n == 0:
        return {"only_first_correct": 0, "only_second_correct": 0, "p": 1.0}
    import math

    k = min(only_a, only_b)  # exact two-sided binomial test of the discordant pairs against p = 0.5
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p": round(p, 5)}


def boot_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95% interval of the mean of per-image values."""
    idx = rng.integers(0, len(values), (BOOTSTRAP, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(acc: dict[str, float], baseline: float) -> dict:
    orig = acc["original"]
    held_stain = float(np.mean([acc[f"stain_{p}"] for p in HELD_OUT_PROTOCOLS]))
    held_bg = float(np.mean([acc[f"bg_{p}"] for p in HELD_OUT_PROTOCOLS]))
    seen_stain = float(np.mean([acc[f"stain_{p}"] for p in TRAIN_PROTOCOLS]))
    keeps = {"cells_only": acc["cells_only"] / orig, "held_out_stain": held_stain / orig, "held_out_bg": held_bg / orig}
    weak = orig <= baseline + 0.10
    morph = keeps["cells_only"] >= 0.9 and keeps["held_out_stain"] >= 0.9 and acc["cells_masked"] <= baseline + 0.10
    shortcut = acc["cells_masked"] > baseline + 0.15 or keeps["held_out_stain"] < 0.7
    return {
        "majority_baseline": round(baseline, 4), "original_accuracy": round(orig, 4),
        "cells_masked_accuracy": round(acc["cells_masked"], 4), "cells_only_accuracy": round(acc["cells_only"], 4),
        "scale_bar_masked_accuracy": round(acc["scale_bar_masked"], 4),
        "mean_held_out_bg_accuracy": round(held_bg, 4), "mean_held_out_stain_accuracy": round(held_stain, 4),
        "mean_train_protocol_stain_accuracy": round(seen_stain, 4),
        "retained_fraction": {k: round(v, 4) for k, v in keeps.items()},
        "verdict": "inconclusive (original accuracy is not above the majority baseline)" if weak
        else "morphology-driven" if morph else ("background/stain-driven" if shortcut else "mixed"),
    }


def held_out_score(correct: dict[str, np.ndarray], kind: str) -> np.ndarray:
    """Per-image share of the held-out protocols classified correctly (kind: 'stain' or 'bg')."""
    return np.mean([correct[f"{kind}_{p}"] for p in HELD_OUT_PROTOCOLS], axis=0)


def compare(models: dict[str, dict], correct: dict[str, dict[str, np.ndarray]], rng: np.random.Generator) -> dict:
    """Baseline vs augmented: per-condition McNemar tests and the retained accuracy on the held-out stains."""
    a, b = correct["baseline"], correct["augmented"]
    per_condition = {c: {"baseline": models["baseline"]["conditions"][c]["accuracy"],
                         "augmented": models["augmented"]["conditions"][c]["accuracy"], **mcnemar(a[c], b[c])}
                     for c in a}
    out = {"per_condition": per_condition}
    keep_a = held_out_score(a, "stain") / max(float(a["original"].mean()), 1e-9)
    keep_b = held_out_score(b, "stain") / max(float(b["original"].mean()), 1e-9)
    diff = keep_b - keep_a
    idx = rng.integers(0, len(diff), (BOOTSTRAP, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    out["held_out_stain_retained"] = {"baseline": round(float(keep_a.mean()), 4), "augmented": round(float(keep_b.mean()), 4),
                                      "difference": round(float(diff.mean()), 4),
                                      "difference_ci95": [round(float(lo), 4), round(float(hi), 4)]}
    vb, va = models["baseline"]["summary"]["verdict"], models["augmented"]["summary"]["verdict"]
    removed = ("background" in vb and "background" not in va) or (diff.mean() >= 0.10 and lo > 0)
    if vb.startswith("inconclusive") or va.startswith("inconclusive"):
        out["verdict"] = "inconclusive: a model does not beat the majority baseline"
    elif removed:
        out["verdict"] = "the augmentation removed a background/stain dependence that the baseline had"
    elif vb == va == "morphology-driven":
        out["verdict"] = "both models read morphology; the augmentation was not needed"
    elif "background" in vb and "background" in va:
        out["verdict"] = "both models depend on background/stain; the augmentation did not remove it"
    else:
        out["verdict"] = "no clear difference between the models"
    return out


# ----- models ----------------------------------------------------------------

def find_input(name: str, required: bool = True) -> Path | None:
    """A file the previous steps left in the working directory or in /content."""
    for base in (Path.cwd(), Path("/content")):
        if (base / name).is_file():
            return base / name
    if required:
        raise FileNotFoundError(f"{name} not found in {Path.cwd()} or /content")
    return None


def torch_models() -> tuple[dict, str]:
    """{name: classify(list of 1x3xSxS arrays) -> logits} for the trained TorchScript models, and a device label."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base_file = find_input(os.environ.get("PROBE_BASELINE", "model_baseline.pt"), required=False)
    aug_file = find_input(os.environ.get("PROBE_MODEL", "model.pt"))
    files = {"baseline": base_file, "augmented": aug_file} if base_file else {"model": aug_file}

    def make(path: Path):
        model = torch.jit.load(str(path), map_location=dev).eval()

        def classify(batch: list[np.ndarray]) -> np.ndarray:
            with torch.no_grad():
                return model(torch.from_numpy(np.concatenate(batch)).to(dev)).float().cpu().numpy()

        return classify

    label = f"Google Colab, {torch.cuda.get_device_name(0)}" if dev == "cuda" else "Google Colab (CPU)"
    return {name: make(path) for name, path in files.items()}, label


def test_directory() -> str:
    if BACKEND == "torch":
        import zipfile

        if os.environ.get("PROBE_TEST_ZIP"):
            out = Path("test_data")
            shutil.rmtree(out, ignore_errors=True)
            zipfile.ZipFile(find_input(os.environ["PROBE_TEST_ZIP"])).extractall(out)
            return str(out)
        import kaggle_data

        return kaggle_data.prepare()["test_dir"]
    tb.unpack_data()
    if not tb.TEST_DIR:
        sys.exit("set BENCH_TEST_DIR to an ImageFolder-style test directory")
    return tb.TEST_DIR


def main() -> None:
    test_dir = test_directory()
    if BACKEND == "triton":
        tb.stage_models()  # BENCH_MODEL_DIR -> this experiment's working directory
    paths, y, classes = tb.load_test_set(test_dir)
    if PER_CLASS_LIMIT:
        keep = np.concatenate([np.where(y == i)[0][:PER_CLASS_LIMIT] for i in range(len(classes))])
        paths, y = [paths[i] for i in keep], y[keep]
    work = Path.cwd()
    out_dim = None
    baseline = float(np.bincount(y, minlength=len(classes)).max() / len(y))
    results: dict = {
        "backend": BACKEND, "classes": classes, "n_images": len(paths),
        "protocols": {"train": TRAIN_PROTOCOLS, "held_out": HELD_OUT_PROTOCOLS, "parameters": PROTOCOLS},
        "scale_bar_region_from": SCALE_BAR, "bootstrap_resamples": BOOTSTRAP, "models": {},
    }
    names: list[str] = []
    logits: dict[str, dict[str, list[np.ndarray]]] = {}
    examples: list[tuple[str, dict[str, np.ndarray]]] = []
    shown: set[int] = set()
    try:
        if BACKEND == "torch":
            classifiers, results["device"] = torch_models()
        else:
            shutil.rmtree(tb.REPO, ignore_errors=True)
            tb.REPO.mkdir()
            results["device"] = tb.device_label()
            tb.start_triton()
            name = tb.stage(PROBE_VARIANT, out_dim)
            status, _, body = tb.http_call("POST", f"/v2/repository/models/{name}/load")
            if status != 200:
                raise RuntimeError(body.decode()[:600])
            in_name, out_name, _ = tb.model_io(name)
            classifiers = {"model": lambda batch: np.stack([tb.infer_logits(name, in_name, out_name, x) for x in batch])}
            results["triton_variant"] = PROBE_VARIANT
        logits = {m: {} for m in classifiers}
        for i, p in enumerate(paths):
            conds = conditions(tb.load_rgb(p), RNG_SEED + i)
            names = list(conds)
            batch = [tb.preprocess_rgb(conds[c]) for c in names]
            for m, classify in classifiers.items():
                for c, row in zip(names, classify(batch)):
                    logits[m].setdefault(c, []).append(np.asarray(row).reshape(-1))
            if int(y[i]) not in shown:  # one example per class for the montage
                shown.add(int(y[i]))
                examples.append((classes[int(y[i])], conds))
            if (i + 1) % 50 == 0:
                print(f"probed {i + 1}/{len(paths)}", flush=True)
    finally:
        if BACKEND == "triton":
            tb.sh(["docker", "rm", "-f", tb.CONTAINER])

    rng = np.random.default_rng(RNG_SEED)
    correct: dict[str, dict[str, np.ndarray]] = {}
    for m in logits:
        correct[m], acc, conds_out = {}, {}, {}
        base_pred = np.stack(logits[m]["original"]).argmax(axis=1)
        for c in names:
            pred = np.stack(logits[m][c]).argmax(axis=1)
            correct[m][c] = pred == y
            acc[c] = float(correct[m][c].mean())
            cm = tb.confusion_matrix(y, pred, len(classes))
            lo, hi = boot_ci(correct[m][c].astype(float), rng)
            conds_out[c] = {
                "accuracy": round(acc[c], 4), "ci95": [round(lo, 4), round(hi, 4)], "macro_f1": round(tb.macro_f1(cm), 4),
                "agreement_with_original": round(float((pred == base_pred).mean()), 4),
                "predicted_class_counts": np.bincount(pred, minlength=len(classes)).tolist(),
                "protocol_set": protocol_set(c),
            }
            if c != "original":
                conds_out[c]["vs_original"] = mcnemar(correct[m]["original"], correct[m][c])
        results["models"][m] = {"conditions": conds_out, "summary": verdict(acc, baseline)}
    if {"baseline", "augmented"} <= set(logits):
        results["comparison"] = compare(results["models"], correct, rng)
    (work / "probe_results.json").write_text(json.dumps(results, indent=2))

    show = ["original", "cells_masked", "cells_only", "scale_bar_masked", "bg_giemsa", "bg_leishman", "stain_giemsa", "stain_leishman"]
    cv2.imwrite(str(work / "probe_examples.png"),
                cv2.cvtColor(montage([e[1] for e in examples], show, [e[0] for e in examples]), cv2.COLOR_RGB2BGR))
    print(json.dumps({"models": {m: v["summary"] for m, v in results["models"].items()},
                      "comparison": results.get("comparison", {}).get("verdict")}, indent=2))


if __name__ == "__main__":
    main()
