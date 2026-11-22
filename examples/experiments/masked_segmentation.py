"""With every person masked out, can a segmentation network still tell the class? (Colab, out-of-fold)

Every training photograph is shown to the model in one of four forms, and the model is trained AND tested on that form
(unlike shortcut_probe.py, which trains on the original and only tests on changed images):

  original         the photograph as it is (control: what the model can do with everything visible)
  person_masked    every pixel the person network calls 'person' (dilated by MASK_GROW pixels) painted a flat colour: the
                   room remains AND so does the person's silhouette (its size, position and shape)
  scene_only       the person inpainted away: the room alone, no silhouette
  silhouette_only  the person mask as a white shape on black: no pixel of the photograph at all, only where the person stands

Models are segmentation networks used as encoders with a pooled linear head, fine-tuned end to end:
  deeplabv3_mnv3   torchvision DeepLabV3-MobileNetV3-Large backbone (segmentation weights)
  segformer_b0     SegFormer MiT-b0 encoder (transformers, nvidia/mit-b0)
A shuffled-label run of each (model, form) gives the accuracy that chance and the majority class alone produce.

Decision rule (written before the run), per (model, form): "detects the class" when out-of-fold accuracy > majority
baseline + 0.10 AND the lower end of its 95% bootstrap interval is above the baseline AND it also exceeds the shuffled-label
accuracy by more than 0.05; otherwise "does not detect the class".  Which cue it uses is read from the forms:
scene_only detecting the class means the room carries it; silhouette_only detecting it means the framing does.
Statistics: stratified session-aware K-fold of the TRAINING images only; the 14 test images are not used.

Settings (env): K=5 EPOCHS=15 BATCH=16 LR=0.0002 SEED=42 BOOTSTRAP=1000 MASK_GROW=15 MODELS=deeplabv3_mnv3,segformer_b0
Writes masked_segmentation.json.  No photograph is written or drawn.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

for _dir in (Path.cwd(), Path("/content")):  # a Colab kernel does not always put the script's folder on sys.path
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import cv2
import numpy as np
from PIL import Image

import cvlib
import data_source
import person_mask
import photo_conditions as pc
import stats_utils as su

ENV = os.environ.get
K, EPOCHS, BATCH, LR = int(ENV("K", "5")), int(ENV("EPOCHS", "15")), int(ENV("BATCH", "16")), float(ENV("LR", "0.0002"))
SEED, BOOT, GROW = int(ENV("SEED", "42")), int(ENV("BOOTSTRAP", "1000")), int(ENV("MASK_GROW", "15"))
MODELS = [m for m in ENV("MODELS", "deeplabv3_mnv3,segformer_b0").split(",") if m]
FORMS = ("original", "person_masked", "scene_only", "silhouette_only")
torch, nn = cvlib.torch, cvlib.nn


class DeepLabEncoder(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights as W
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

        self.backbone = deeplabv3_mobilenet_v3_large(weights=W.DEFAULT).backbone
        self.head = nn.Linear(960, n)  # the last backbone stage of MobileNetV3-Large has 960 channels

    def forward(self, x):
        return self.head(self.backbone(x)["out"].mean((2, 3)))


class SegformerEncoder(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        from transformers import SegformerModel

        self.encoder = SegformerModel.from_pretrained("nvidia/mit-b0")
        self.head = nn.Linear(self.encoder.config.hidden_sizes[-1], n)

    def forward(self, x):
        return self.head(self.encoder(pixel_values=x).last_hidden_state.mean((2, 3)))


def make(name: str, n: int) -> nn.Module:
    return (DeepLabEncoder if name == "deeplabv3_mnv3" else SegformerEncoder)(n).to(cvlib.DEVICE)


def forms_of(rgb: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    grown = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROW + 1, 2 * GROW + 1))).astype(bool)
    flat = rgb.copy()
    flat[grown] = (124, 116, 104)
    sil = np.zeros_like(rgb)
    sil[grown] = 255
    return {"original": rgb, "person_masked": flat, "scene_only": pc.canvas_of(rgb, grown), "silhouette_only": sil}


def cross_validate(name: str, images: list[Image.Image], y: np.ndarray, fold_of: np.ndarray, seed: int) -> np.ndarray:
    probs = np.zeros((len(y), int(y.max()) + 1))
    for k in range(K):
        tr, va = np.where(fold_of != k)[0], np.where(fold_of == k)[0]
        model = make(name, probs.shape[1])
        cvlib.fit(model, cvlib.ImageList([images[i] for i in tr], y[tr], 224, train=True, seed=seed + k), None, EPOCHS, LR, BATCH, seed + k)
        probs[va] = cvlib.predict(model, [images[i] for i in va], 224)
        del model
        torch.cuda.empty_cache() if cvlib.DEVICE.type == "cuda" else None
    return probs


def main() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers"])
    data = data_source.prepare()
    classes, base_dir = data["info"]["classes"], Path(data["train_dir"]).parent
    train = sorted((r for r in data["records"] if r["split"] == "train"), key=lambda r: r["path"])
    y = np.array([classes.index(r["class"]) for r in train])
    groups = np.array([r["group"] for r in train])
    arr = [np.asarray(Image.open(base_dir / r["path"]).convert("RGB")) for r in train]
    masks = person_mask.person_masks([r["path"] for r in train], base_dir)
    mask_list = [masks[r["path"]] for r in train]
    n_empty = int(sum(m.mean() < 0.02 for m in mask_list))
    built = [forms_of(a, m) for a, m in zip(arr, mask_list)]
    images = {f: [Image.fromarray(b[f]) for b in built] for f in FORMS}
    try:
        fold_of, grouped = su.group_stratified_folds(y, groups, K, SEED), True
    except ValueError:
        fold_of, grouped = su.group_stratified_folds(y, np.arange(len(y)).astype(str), K, SEED), False
    base = float(np.bincount(y).max() / len(y))
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(y)  # same fold structure, labels carry no information: the chance reference
    print(f"{len(y)} images, {K} folds, {EPOCHS} epochs, {n_empty} without a person mask, majority {base:.3f}, device {cvlib.DEVICE}", flush=True)
    results: dict = {"classes": classes, "n_images": len(y), "k": K, "epochs": EPOCHS, "lr": LR, "mask_grow_px": GROW, "session_aware_folds": grouped,
                     "images_without_person_mask": n_empty, "majority_baseline": round(base, 4), "bootstrap_resamples": BOOT,
                     "device": f"Google Colab, {torch.cuda.get_device_name(0)}" if cvlib.DEVICE.type == "cuda" else "Google Colab (CPU)",
                     "mean_person_share_of_frame": round(float(np.mean([m.mean() for m in mask_list])), 4),
                     "evaluation": "out-of-fold, trained and tested on the same form; the 14 test images are not used", "models": {}}
    for name in MODELS:
        res: dict = {}
        for form in FORMS:
            p = cross_validate(name, images[form], y, fold_of, SEED)
            ps = cross_validate(name, images[form], shuffled, fold_of, SEED)
            correct = (p.argmax(1) == y).astype(float)
            acc, chance = float(correct.mean()), float((ps.argmax(1) == shuffled).mean())
            lo, hi = su.boot_ci(correct, rng, BOOT)
            met = cvlib.metrics(y, p)
            detects = acc > base + 0.10 and lo > base and acc > chance + 0.05
            res[form] = {"accuracy": round(acc, 4), "ci95": [round(lo, 4), round(hi, 4)], "balanced_accuracy": round(met["balanced_accuracy"], 4),
                         "macro_f1": round(met["macro_f1"], 4), "shuffled_label_accuracy": round(chance, 4),
                         "predicted_class_counts": np.bincount(p.argmax(1), minlength=len(classes)).tolist(),
                         "confusion_matrix": cvlib.confusion(y, p.argmax(1), len(classes)).tolist(),
                         "verdict": "detects the class" if detects else "does not detect the class"}
            print(name, form, res[form]["accuracy"], res[form]["ci95"], "shuffled", res[form]["shuffled_label_accuracy"], res[form]["verdict"], flush=True)
        results["models"][name] = res
    Path("masked_segmentation.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({m: {f: v["verdict"] for f, v in r.items()} for m, r in results["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
