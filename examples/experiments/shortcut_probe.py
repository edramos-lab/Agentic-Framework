"""Does a photograph classifier read the person or the room?  Baseline vs background-randomised training, out-of-fold (Colab).

For the backbone chosen by train_cv.py (selected.json), two recipes are trained inside the same session-aware
cross-validation of the TRAINING images: "baseline" (no background change) and "augmented" (with probability AUG_P a
training image gets a new background: a flat colour, a gradient, or the scene of another training image).  Every
training image is classified by models that never saw it, under these conditions:

  original       the photograph as it is
  person_masked  the person replaced by a flat colour: the surroundings remain AND so does the person's silhouette (its size,
                 position and shape), so accuracy here can come from the silhouette, not from the room
  scene_only     the person inpainted away: the surroundings alone, with no silhouette
  person_only    the person on a flat grey (a kind of background used in training)
  bg_other       the person on the scene of a different image the model never trained on (held out)
  bg_texture     the person on coloured noise (held out)
  person_box_crop  cropped to the person's bounding box: the person's scale and position in the frame are removed; the
                 box's aspect ratio (a strong class cue in the audit) is NOT
  grayscale      the photograph without colour
  blur           the photograph blurred (Gaussian, sigma 4)

The person/background split comes from a pretrained segmentation network (person_mask.py).  The backgrounds are
synthetic (compositions), not real rooms.  person_box_crop, grayscale and blur are reported for every recipe but are not
part of the decision rule below.

Decision rule per recipe (written before the run).  orig = accuracy on the original images, base = the share of the largest
class in the training pool, held = mean accuracy on the held-out backgrounds (bg_other, bg_texture):
  inconclusive          orig <= base + 0.10 (nothing to probe)
  robust to the tested changes   person_only >= 0.9 * orig AND held >= 0.9 * orig AND bgonly <= base + 0.10
  background-reliant    bgonly > base + 0.15 OR held < 0.7 * orig
  mixed                 otherwise
where bgonly is scene_only when it was computed (rule v2) and person_masked otherwise (rule v1, which the first
lower-limb run used).  "Robust to the tested changes" does not mean the model reads the legs: the person's silhouette and the
box's aspect ratio stay in every condition except scene_only.
Comparison: the augmentation "removed a background dependence" when the baseline is background-reliant and the augmented
recipe is not, or when it keeps at least 0.10 more of its accuracy on the held-out backgrounds than the baseline with a
paired-bootstrap interval that excludes 0.  Statistics: 95% bootstrap intervals; exact McNemar tests (original vs each
condition within a recipe, baseline vs augmented within a condition).

Settings (env): K=5  EPOCHS (default: from selected.json)  AUG_P=0.7  BATCH=16  LR=0.0005  SEED=42  BOOTSTRAP=1000
                PROBE_ARCH (default: selected.json)  SHOW_IMAGES=0 (1 writes probe_examples.png: photographs of people, so off)
Writes probe_results.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _dir in (Path.cwd(), Path("/content")):  # a Colab kernel does not always put the script's folder on sys.path
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import numpy as np
from PIL import Image

import cvlib
import data_source
import person_mask
import photo_conditions as pc
import stats_utils as su

ENV = os.environ.get
K, AUG_P, BATCH, LR, SEED, BOOT = int(ENV("K", "5")), float(ENV("AUG_P", "0.7")), int(ENV("BATCH", "16")), float(ENV("LR", "0.0005")), int(ENV("SEED", "42")), int(ENV("BOOTSTRAP", "1000"))
HELD_OUT = ("bg_other", "bg_texture")
SEEN = ("person_only",)
RECIPES = ("baseline", "augmented")


def selected() -> dict:
    for base in (Path.cwd(), Path("/content")):
        if (base / "selected.json").is_file():
            return json.loads((base / "selected.json").read_text())
    return {"arch": ENV("PROBE_ARCH", "efficientvit_b0.r224_in1k"), "epochs": 15, "imgsz": 224}


def protocol_set(cond: str) -> str:
    return "held_out" if cond in HELD_OUT else "train" if cond in SEEN else ""


def verdict(acc: dict[str, float], base: float) -> dict:
    orig = acc["original"]
    held = float(np.mean([acc[c] for c in HELD_OUT]))
    keeps = {"person_only": acc["person_only"] / orig, "held_out_backgrounds": held / orig}
    weak = orig <= base + 0.10
    bgonly = acc.get("scene_only", acc["person_masked"])  # rule v2 uses the condition without the silhouette
    person = keeps["person_only"] >= 0.9 and keeps["held_out_backgrounds"] >= 0.9 and bgonly <= base + 0.10
    reliant = bgonly > base + 0.15 or keeps["held_out_backgrounds"] < 0.7
    return {"majority_baseline": round(base, 4), "original_accuracy": round(orig, 4), "person_masked_accuracy": round(acc["person_masked"], 4),
            "person_only_accuracy": round(acc["person_only"], 4), "mean_held_out_background_accuracy": round(held, 4),
            "scene_only_accuracy": round(acc["scene_only"], 4) if "scene_only" in acc else None,
            "person_box_crop_accuracy": round(acc["person_box_crop"], 4),
            "grayscale_accuracy": round(acc["grayscale"], 4), "blur_accuracy": round(acc["blur"], 4),
            "retained_fraction": {k: round(v, 4) for k, v in keeps.items()},
            "verdict": "inconclusive (original accuracy is not above the majority baseline)" if weak
            else "robust to the tested changes" if person else ("background-reliant" if reliant else "mixed")}


def main() -> None:
    sel = selected()
    arch, epochs, imgsz = sel["arch"], int(ENV("EPOCHS", sel.get("epochs", 15))), int(sel.get("imgsz", 224))
    data = data_source.prepare()
    classes, base_dir = data["info"]["classes"], Path(data["train_dir"]).parent
    train = sorted((r for r in data["records"] if r["split"] == "train"), key=lambda r: r["path"])
    y = np.array([classes.index(r["class"]) for r in train])
    groups = np.array([r["group"] for r in train])
    pil = [Image.open(base_dir / r["path"]).convert("RGB") for r in train]
    arr = [np.asarray(i) for i in pil]
    masks = person_mask.person_masks([r["path"] for r in train], base_dir)
    mask_list = [masks[r["path"]] for r in train]
    n_empty = int(sum(m.mean() < 0.02 for m in mask_list))
    canvases = [pc.canvas_of(a, m) for a, m in zip(arr, mask_list)]
    try:
        fold_of, grouped = su.group_stratified_folds(y, groups, K, SEED), True
    except ValueError:
        fold_of, grouped = su.group_stratified_folds(y, np.arange(len(y)).astype(str), K, SEED), False
    majority = float(np.bincount(y).max() / len(y))
    rng = np.random.default_rng(SEED)
    conds = list(pc.conditions(arr[0], mask_list[0], canvases[1], 0))
    probs = {m: {c: np.zeros((len(y), len(classes))) for c in conds} for m in RECIPES}
    print(f"{arch}: {len(y)} images, {K} folds, {epochs} epochs, {n_empty} images without a person mask, device {cvlib.DEVICE}", flush=True)
    for k in range(K):
        tr, va = np.where(fold_of != k)[0], np.where(fold_of == k)[0]
        pool = [canvases[i] for i in tr]  # the backgrounds the augmented recipe may use: training images only
        by_cond: dict[str, list] = {c: [] for c in conds}
        for n, i in enumerate(va):
            other = canvases[va[(n + 1) % len(va)]]  # the scene of another validation image: never seen in training
            for c, img in pc.conditions(arr[i], mask_list[i], other, SEED + int(i)).items():
                by_cond[c].append(Image.fromarray(img))
        for recipe in RECIPES:
            model = cvlib.make_model(arch, len(classes))
            ds = cvlib.ImageList([pil[i] for i in tr], y[tr], imgsz, train=True, seed=SEED + k,
                                 masks=[mask_list[i] for i in tr] if recipe == "augmented" else None, pool=pool, aug_p=AUG_P)
            cvlib.fit(model, ds, None, epochs, LR, BATCH, SEED + k)
            for c in conds:
                probs[recipe][c][va] = cvlib.predict(model, by_cond[c], imgsz)
            del model
        print(f"fold {k + 1}/{K} done", flush=True)

    results: dict = {"backend": "torch", "device": f"Google Colab, {cvlib.torch.cuda.get_device_name(0)}" if cvlib.DEVICE.type == "cuda" else "Google Colab (CPU)",
                     "classes": classes, "n_images": len(y), "backbone": arch, "epochs": epochs, "k": K, "session_aware_folds": grouped,
                     "images_without_person_mask": n_empty, "augmentation_probability": AUG_P, "bootstrap_resamples": BOOT,
                     "evaluation": "out-of-fold: every image is classified by models trained without it", "models": {}}
    correct: dict[str, dict[str, np.ndarray]] = {}
    for m in RECIPES:
        correct[m], acc, out = {}, {}, {}
        base_pred = probs[m]["original"].argmax(1)
        for c in conds:
            pred = probs[m][c].argmax(1)
            correct[m][c] = pred == y
            acc[c] = float(correct[m][c].mean())
            lo, hi = su.boot_ci(correct[m][c].astype(float), rng, BOOT)
            met = cvlib.metrics(y, probs[m][c])
            out[c] = {"accuracy": round(acc[c], 4), "ci95": [round(lo, 4), round(hi, 4)], "macro_f1": round(met["macro_f1"], 4),
                      "balanced_accuracy": round(met["balanced_accuracy"], 4), "agreement_with_original": round(float((pred == base_pred).mean()), 4),
                      "predicted_class_counts": np.bincount(pred, minlength=len(classes)).tolist(), "protocol_set": protocol_set(c)}
            if c != "original":
                out[c]["vs_original"] = su.mcnemar(correct[m]["original"], correct[m][c])
        results["models"][m] = {"conditions": out, "summary": verdict(acc, majority)}

    a, b = correct["baseline"], correct["augmented"]
    held = lambda cor: np.mean([cor[c] for c in HELD_OUT], axis=0) / max(float(cor["original"].mean()), 1e-9)
    diff = held(b) - held(a)
    idx = rng.integers(0, len(diff), (BOOT, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    vb, va_ = (results["models"][m]["summary"]["verdict"] for m in RECIPES)
    removed = (vb == "background-reliant" and va_ != "background-reliant") or (diff.mean() >= 0.10 and lo > 0)
    results["comparison"] = {
        "per_condition": {c: {"baseline": results["models"]["baseline"]["conditions"][c]["accuracy"],
                              "augmented": results["models"]["augmented"]["conditions"][c]["accuracy"], **su.mcnemar(a[c], b[c])} for c in conds},
        "held_out_background_retained": {"baseline": round(float(held(a).mean()), 4), "augmented": round(float(held(b).mean()), 4),
                                         "difference": round(float(diff.mean()), 4), "difference_ci95": [round(float(lo), 4), round(float(hi), 4)]},
        "verdict": ("inconclusive: a recipe does not beat the majority baseline" if "inconclusive" in vb + va_
                    else "the background randomisation removed a background dependence that the baseline had" if removed
                    else "both recipes are robust to the tested changes; the background randomisation was not needed" if vb == va_ == "robust to the tested changes"
                    else "both recipes depend on the background; the randomisation did not remove it" if vb == va_ == "background-reliant"
                    else "no clear difference between the recipes")}
    Path("probe_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"models": {m: v["summary"]["verdict"] for m, v in results["models"].items()}, "comparison": results["comparison"]["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
