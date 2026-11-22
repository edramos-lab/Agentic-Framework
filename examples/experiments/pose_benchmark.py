"""Benchmark of body-keypoint detectors on the lower-limb photographs: MediaPipe Pose, RTMPose, ViTPose (Colab, training images only).

Detectors (all pretrained, none fitted here):
  keypointrcnn   torchvision Keypoint R-CNN, the detector of the earlier pose audit, kept as the reference row
  mediapipe      MediaPipe Pose (BlazePose, heavy model), single person, its own person finder
  rtmpose_m/_x   RTMPose through rtmlib (ONNX Runtime), with the YOLOX person detector rtmlib bundles ("balanced" / "performance")
  vitpose_b/_h   ViTPose (Hugging Face transformers, "simple" base and "plus" huge), a top-down model: the person box comes
                 from Keypoint R-CNN and its time is included in the ViTPose time

There is no keypoint ground truth for these photographs, so the benchmark does NOT measure keypoint accuracy.  It measures what the
audit needs from a detector, per detector, on the same 129 training images (resized to a 512-pixel long side):
  speed           median / 90th-percentile / mean time per image, person finding included, after warm-up
  availability    how often both hips, knees and ankles are found at the detector's own confidence threshold, per class, with a
                  chi-square test of class against "knees and ankles both found" (a framing cue when it is significant)
  agreement       between every pair of detectors, on the images where both find a joint: the median distance in pixels
                  between the two detectors' positions of the hip, knee and ankle, and the Spearman correlation of the leg
                  measures (knee gap / ankle gap, tibial tilt) that both can compute
  separability    the leg geometry (same features as pose_audit.py) per class with Kruskal-Wallis tests, and a session-aware
                  cross-validated linear classifier on the images with all lower-leg keypoints, against the majority baseline
                  of those images and a permutation null
Agreement between detectors shows that they locate the same joints, not that the joints are right: all of them were trained on
upright, whole bodies in everyday scenes and have no heel or foot points.  Nothing here is a clinical measurement.

Confidence scores have different meanings (heat-map logit, visibility, SimCC peak, heat-map peak), so each detector keeps its own
threshold, set before the run: keypointrcnn 1.0, mediapipe 0.5, rtmpose 0.3, vitpose 0.3 (the 0.3 is the usual keypoint threshold of
the RTMPose / ViTPose tooling).  Override with THR_<NAME>=value.  The score quantiles at the knees are reported next to them.

Verdict rule (written before the run): a detector "separates the classes" when the cross-validated accuracy of its geometry exceeds
the majority baseline of the images it used by more than 0.10 and its permutation p is below 0.05.

Settings (env): DETECTORS=keypointrcnn,mediapipe,rtmpose_m,rtmpose_x,vitpose_b,vitpose_h  PERMUTATIONS=500  K=5  SEED=7  WARMUP=3
Writes pose_benchmark.json and pose_benchmark.png (statistics only: no photograph is drawn).
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


def _pip(*packages: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], capture_output=True, text=True)


def _ensure(module: str, *packages: str) -> bool:
    """Import a module, installing its package first when missing; False (and a note) when it still cannot be imported."""
    try:
        importlib.import_module(module)
        return True
    except Exception:
        _pip(*packages)
        importlib.invalidate_caches()
        try:
            importlib.import_module(module)
            return True
        except Exception as exc:
            print(f"cannot import {module}: {exc}", flush=True)
            return False


WANTED = [d.strip() for d in os.environ.get("DETECTORS", "keypointrcnn,mediapipe,rtmpose_m,rtmpose_x,vitpose_b,vitpose_h").split(",") if d.strip()]
# installs happen before numpy and torch are imported, so a changed numpy is the one that gets loaded
if any(d.startswith("mediapipe") for d in WANTED):
    _ensure("mediapipe", "mediapipe")
if any(d.startswith("rtmpose") for d in WANTED):
    # rtmlib is installed without being imported: importing it loads the CPU build of ONNX Runtime, which would stay loaded after the swap
    if subprocess.run([sys.executable, "-c", "import rtmlib"], capture_output=True).returncode != 0:
        _pip("rtmlib")
    if shutil.which("nvidia-smi"):
        # asked in a child process: importing the CPU build here would keep it loaded after the swap below
        probe = subprocess.run([sys.executable, "-c", "import onnxruntime as o; print('CUDAExecutionProvider' in o.get_available_providers())"],
                               capture_output=True, text=True)
        if probe.stdout.strip() != "True":
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "onnxruntime"], capture_output=True, text=True)
            _pip("onnxruntime-gpu")
if any(d.startswith("vitpose") for d in WANTED):
    _ensure("transformers", "transformers")

for _dir in (Path.cwd(), Path("/content")):  # a Colab kernel does not always put the script's folder on sys.path
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_source
import pose_audit as pa
import stats_utils as su

PERMUTATIONS, K, SEED, WARMUP = (int(os.environ.get(k, d)) for k, d in (("PERMUTATIONS", "500"), ("K", "5"), ("SEED", "7"), ("WARMUP", "3")))
THRESHOLDS = {"keypointrcnn": 1.0, "mediapipe": 0.5, "rtmpose": 0.3, "vitpose": 0.3}
LOWER = [11, 12, 13, 14, 15, 16]
COCO_TO_MEDIAPIPE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
LABEL = {"keypointrcnn": "Keypoint R-CNN", "mediapipe": "MediaPipe Pose", "rtmpose_m": "RTMPose-m", "rtmpose_x": "RTMPose-x",
         "vitpose_b": "ViTPose-B", "vitpose_h": "ViTPose+-H"}
PERSON_FINDER = {"keypointrcnn": "itself", "mediapipe": "built in", "rtmpose_m": "YOLOX-m", "rtmpose_x": "YOLOX-x",
                 "vitpose_b": "Keypoint R-CNN", "vitpose_h": "Keypoint R-CNN"}


def threshold(name: str) -> float:
    family = "rtmpose" if name.startswith("rtmpose") else "vitpose" if name.startswith("vitpose") else name
    return float(os.environ.get(f"THR_{name.upper()}", THRESHOLDS[family]))


def best_person(xy: np.ndarray, sc: np.ndarray):
    """Of several detected people, the one whose lower-limb keypoints have the highest total confidence."""
    if len(xy) == 0:
        return None
    i = int(np.argmax(sc[:, LOWER].sum(1)))
    return np.asarray(xy[i], float), np.asarray(sc[i], float)


# ---- detectors: each returns a callable image -> (xy 17x2, scores 17) or None, plus a description of where it ran

def load_keypointrcnn():
    import torch
    from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights as W
    from torchvision.models.detection import keypointrcnn_resnet50_fpn
    from torchvision.transforms.functional import to_tensor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = keypointrcnn_resnet50_fpn(weights=W.DEFAULT).eval().to(dev)

    def run(img: Image.Image, want_box: bool = False):
        with torch.no_grad():
            out = model([to_tensor(img).to(dev)])[0]
        if len(out["scores"]) == 0 or float(out["scores"][0]) < 0.5:
            return (None, None) if want_box else None
        box = out["boxes"][0].cpu().numpy()
        pose = (out["keypoints"][0].cpu().numpy()[:, :2], out["keypoints_scores"][0].cpu().numpy())
        return (pose, box) if want_box else pose

    return run, dev


def load_mediapipe():
    import mediapipe as mp

    if hasattr(mp, "solutions"):
        pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.3)

        def landmarks(arr):
            res = pose.process(arr)
            return None if not res.pose_landmarks else [(l.x, l.y, l.visibility) for l in res.pose_landmarks.landmark]
    else:  # newer releases ship only the Tasks API, which needs the model file
        import urllib.request

        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        path = Path("pose_landmarker_heavy.task")
        if not path.exists():
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task", path)
        det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=str(path)), min_pose_detection_confidence=0.3))

        def landmarks(arr):
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
            return None if not res.pose_landmarks else [(l.x, l.y, l.visibility) for l in res.pose_landmarks[0]]

    def run(img: Image.Image):
        arr = np.ascontiguousarray(np.array(img.convert("RGB")))
        lm = landmarks(arr)
        if lm is None:
            return None
        h, w = arr.shape[:2]
        pts = np.array([lm[i] for i in COCO_TO_MEDIAPIPE], float)
        return np.stack([pts[:, 0] * w, pts[:, 1] * h], 1), pts[:, 2]

    return run, "cpu (MediaPipe runs on the CPU)"


def load_rtmpose(mode: str):
    import torch  # imported first so that ONNX Runtime finds the CUDA libraries that torch ships
    from rtmlib import Body

    want_cuda = torch.cuda.is_available()
    body = Body(mode=mode, to_openpose=False, backend="onnxruntime", device="cuda" if want_cuda else "cpu")
    try:
        providers = body.pose_model.session.get_providers()
    except Exception:
        providers = []
    dev = "cuda" if providers and providers[0].startswith("CUDA") else "cpu"

    def run(img: Image.Image):
        kps, sc = body(np.array(img.convert("RGB"))[:, :, ::-1].copy())  # rtmlib expects BGR
        return best_person(np.asarray(kps)[:, :17], np.asarray(sc)[:, :17]) if len(kps) else None

    return run, dev + (f" (providers {providers[:2]})" if providers else "")


def load_vitpose(model_name: str, plus: bool, rcnn):
    import torch
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_name)
    model = VitPoseForPoseEstimation.from_pretrained(model_name).eval().to(dev)

    def run(img: Image.Image):
        _, box = rcnn(img, want_box=True)  # the person box of Keypoint R-CNN; the time is included
        if box is None:
            return None
        boxes = [np.array([[box[0], box[1], box[2] - box[0], box[3] - box[1]]], dtype=np.float32)]  # COCO format x, y, w, h
        inputs = processor(img.convert("RGB"), boxes=boxes, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inputs, dataset_index=torch.tensor([0], device=dev)) if plus else model(**inputs)
        res = processor.post_process_pose_estimation(out, boxes=boxes)[0][0]
        return res["keypoints"].cpu().numpy(), res["scores"].cpu().numpy()

    return run, dev


# ---- analysis of one detector's keypoints (the same features and tests as pose_audit.py)

def analyse(name: str, preds: list, y: np.ndarray, groups: np.ndarray, classes: list[str], fold_of: np.ndarray, widths: list[int], heights: list[int]) -> dict:
    thr = threshold(name)
    found = []
    for p, w, h in zip(preds, widths, heights):
        found.append({} if p is None else {i: (float(p[0][i, 0]), float(p[0][i, 1])) for i in range(17)
                                            if p[1][i] >= thr and 0 <= p[0][i, 0] <= w and 0 <= p[0][i, 1] <= h})
    names = pa.GEOMETRY + pa.WITH_HIPS
    feats = np.array([[pa.geometry(f)[n] for n in names] for f in found])
    avail = np.array([[bool(f), pa.side(pa.KNEE, f) is not None, pa.side(pa.ANKLE, f) is not None, pa.side(pa.HIP, f) is not None] for f in found], float)
    labels = ["person_found", "both_knees", "both_ankles", "both_hips"]
    lower = ((avail[:, 1] > 0) & (avail[:, 2] > 0)).astype(int)
    out: dict = {"score_threshold": thr, "n_images": len(y),
                 "availability": {c: {l: round(float(avail[y == i, j].mean()), 3) for j, l in enumerate(labels)} for i, c in enumerate(classes)},
                 "availability_overall": {l: round(float(avail[:, j].mean()), 3) for j, l in enumerate(labels)},
                 "knees_and_ankles_found": int(lower.sum()),
                 "score_quantiles": {k: [round(float(q), 2) for q in np.quantile([p[1][i] for p in preds if p is not None], [0.1, 0.5, 0.9])]
                                     for k, i in (("hip", 11), ("knee", 13), ("ankle", 15))} if any(p is not None for p in preds) else {}}
    out["availability_test"] = {"outcome": "knees and ankles both found",
                                **pa.chi_square(np.array([[int(((y == i) & (lower == v)).sum()) for v in (0, 1)] for i in range(len(classes))], float))}
    out["per_class"], out["features"] = {}, {}
    for i, c in enumerate(classes):
        out["per_class"][c] = {n: ({"mean": round(float(np.nanmean(feats[y == i, j])), 3), "std": round(float(np.nanstd(feats[y == i, j])), 3), "n": int((~np.isnan(feats[y == i, j])).sum())}
                                   if (~np.isnan(feats[y == i, j])).any() else {"mean": None, "std": None, "n": 0}) for j, n in enumerate(names)}
    for j, n in enumerate(names):
        ok = ~np.isnan(feats[:, j])
        if len(np.unique(y[ok])) >= 2 and ok.sum() >= 10:
            out["features"][n] = {**su.kruskal(feats[ok, j], y[ok]), "n": int(ok.sum())}
    geo = [names.index(n) for n in pa.GEOMETRY]
    complete = lower > 0
    out["geometry_classifier"] = None
    if complete.sum() >= 30 and len(np.unique(y[complete])) == len(classes):
        for k in (K, 3):
            try:
                f2 = su.group_stratified_folds(y[complete], groups[complete], k, SEED)
            except ValueError:
                continue
            rep = su.classifier_report(feats[complete][:, geo], y[complete], f2, PERMUTATIONS, SEED)
            out["geometry_classifier"] = {**rep, "n": int(complete.sum()), "folds": k,
                                          "n_per_class": {c: int(((y == i) & complete).sum()) for i, c in enumerate(classes)},
                                          "separates": bool(rep["cv_accuracy"] > rep["majority_baseline"] + 0.10 and rep["permutation_p"] < 0.05)}
            break
    out["_found"], out["_feats"] = found, feats  # private: used for the agreement, removed before writing
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 10:
        return None
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return round(float(np.corrcoef(ra, rb)[0, 1]), 3)


def agreement(res: dict, names_ok: list[str]) -> dict:
    out: dict = {}
    names = pa.GEOMETRY + pa.WITH_HIPS
    for a_i, a in enumerate(names_ok):
        for b in names_ok[a_i + 1:]:
            fa, fb, ga, gb = res[a]["_found"], res[b]["_found"], res[a]["_feats"], res[b]["_feats"]
            row: dict = {}
            for joint, idx in (("hip", (11, 12)), ("knee", (13, 14)), ("ankle", (15, 16))):
                d = [np.hypot(x[i][0] - z[i][0], x[i][1] - z[i][1]) for x, z in zip(fa, fb) for i in idx if i in x and i in z]
                row[f"{joint}_px"], row[f"{joint}_n"] = (round(float(np.median(d)), 1), len(d)) if d else (None, 0)
            for feat in ("knee_over_ankle_gap", "tibia_inward_tilt"):
                j = names.index(feat)
                row[f"spearman_{feat}"] = spearman(ga[:, j], gb[:, j])
            row["images_both_complete"] = int(sum(1 for x, z in zip(ga[:, 0], gb[:, 0]) if not (np.isnan(x) or np.isnan(z))))
            out[f"{a}|{b}"] = row
    return out


def main() -> None:
    data = data_source.prepare()
    base, classes = Path(data["train_dir"]).parent, data["info"]["classes"]
    train = sorted((r for r in data["records"] if r["split"] == "train"), key=lambda r: r["path"])
    y = np.array([classes.index(r["class"]) for r in train])
    groups = np.array([r["group"] for r in train])
    images = [Image.open(base / r["path"]).convert("RGB") for r in train]
    widths, heights = [im.width for im in images], [im.height for im in images]
    try:
        fold_of, grouped = su.group_stratified_folds(y, groups, K, SEED), True
    except ValueError:
        fold_of, grouped = su.group_stratified_folds(y, np.arange(len(y)).astype(str), K, SEED), False

    import torch

    rcnn = None
    res: dict = {}
    failed: dict = {}
    order = [d for d in WANTED if d in LABEL]
    if any(d.startswith("vitpose") for d in order) and "keypointrcnn" not in order:
        order.insert(0, "keypointrcnn")  # ViTPose needs its person boxes
    for name in order:
        try:
            if name == "keypointrcnn":
                run, device = load_keypointrcnn()
                rcnn = run
            elif name == "mediapipe":
                run, device = load_mediapipe()
            elif name.startswith("rtmpose"):
                run, device = load_rtmpose("balanced" if name.endswith("_m") else "performance")
            else:
                if rcnn is None:
                    raise RuntimeError("Keypoint R-CNN (the person finder) did not load")
                run, device = load_vitpose("usyd-community/vitpose-base-simple" if name.endswith("_b") else "usyd-community/vitpose-plus-huge", name.endswith("_h"), rcnn)
            for im in images[:WARMUP]:
                run(im)
            times, preds = [], []
            for im in images:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                p = run(im)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
                preds.append(p)
            r = analyse(name, preds, y, groups, classes, fold_of, widths, heights)
            r.update({"status": "ok", "label": LABEL[name], "device": device, "person_finder": PERSON_FINDER[name],
                      "latency_ms": {"median": round(float(np.median(times)), 1), "p90": round(float(np.quantile(times, 0.9)), 1), "mean": round(float(np.mean(times)), 1), "n": len(times)}})
            res[name] = r
            print(f"{name}: {r['availability_overall']} median {r['latency_ms']['median']} ms on {device}", flush=True)
        except Exception as exc:  # one detector failing must not lose the others
            failed[name] = f"{type(exc).__name__}: {exc}"[:300]
            print(f"{name} FAILED: {failed[name]}\n{traceback.format_exc()[-600:]}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    ok = [n for n in order if n in res]
    if not ok:
        sys.exit("no detector ran")
    report: dict = {"classes": classes, "n_images": len(y), "images_per_class": {c: int((y == i).sum()) for i, c in enumerate(classes)},
                    "image_long_side_px": 512, "session_aware_folds": grouped, "detectors": {}, "failed": failed, "agreement": agreement(res, ok)}
    for n in ok:
        report["detectors"][n] = {k: v for k, v in res[n].items() if not k.startswith("_")}
    sep = [LABEL[n] for n in ok if (res[n]["geometry_classifier"] or {}).get("separates")]
    best_lower = max(ok, key=lambda n: res[n]["knees_and_ankles_found"])
    fastest = min(ok, key=lambda n: res[n]["latency_ms"]["median"])
    report["summary"] = {"most_complete_lower_leg": LABEL[best_lower], "knees_and_ankles_found": {LABEL[n]: res[n]["knees_and_ankles_found"] for n in ok},
                         "fastest": LABEL[fastest], "geometry_separates_classes": sep,
                         "availability_depends_on_class": [LABEL[n] for n in ok if res[n]["availability_test"]["p"] < 0.05]}
    report["verdict"] = (("the leg geometry of " + ", ".join(sep) + " separates the classes beyond the majority baseline") if sep
                         else "no detector's leg geometry separates the classes beyond the majority baseline")
    Path("pose_benchmark.json").write_text(json.dumps(report, indent=2))

    short = [c.replace("Genu-", "").replace("Asymmetric", "Asym.") for c in classes]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.2), gridspec_kw={"wspace": 0.45})
    w = 0.8 / len(ok)
    for j, n in enumerate(ok):
        axes[0].bar(np.arange(3) + (j - (len(ok) - 1) / 2) * w, [res[n]["availability_overall"][k] for k in ("both_hips", "both_knees", "both_ankles")], w, label=LABEL[n])
        axes[1].plot(range(len(classes)), [res[n]["availability"][c]["both_knees"] for c in classes], "o-", label=LABEL[n], ms=3)
    axes[0].set_xticks(range(3), ["hips", "knees", "ankles"], fontsize=7)
    axes[0].set_title("both found (all classes)", fontsize=8)
    axes[0].legend(fontsize=5, loc="lower left")
    axes[1].set_xticks(range(len(classes)), short, fontsize=7)
    axes[1].set_title("both knees found, by class", fontsize=8)
    axes[2].barh(range(len(ok)), [res[n]["latency_ms"]["median"] for n in ok], color="#4c72b0")
    axes[2].set_yticks(range(len(ok)), [LABEL[n] for n in ok], fontsize=6.5)
    axes[2].set_xscale("log")
    axes[2].set_title("median ms per image", fontsize=8)
    accs, base_acc = [], []
    for n in ok:
        g = res[n]["geometry_classifier"]
        accs.append(g["cv_accuracy"] if g else np.nan)
        base_acc.append(g["majority_baseline"] if g else np.nan)
    axes[3].bar(np.arange(len(ok)) - 0.2, accs, 0.4, label="geometry", color="#55a868")
    axes[3].bar(np.arange(len(ok)) + 0.2, base_acc, 0.4, label="majority", color="#999999")
    axes[3].set_xticks(range(len(ok)), [LABEL[n] for n in ok], rotation=40, ha="right", fontsize=6)
    axes[3].set_ylim(0, 1.0)
    axes[3].set_title("CV accuracy, complete cases", fontsize=8)
    axes[3].legend(fontsize=6)
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    fig.savefig("pose_benchmark.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"summary": report["summary"], "verdict": report["verdict"], "failed": failed}, indent=2))


if __name__ == "__main__":
    main()
