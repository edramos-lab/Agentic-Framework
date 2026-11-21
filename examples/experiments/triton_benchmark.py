"""Benchmark one model through Triton on the Jetson: PyTorch, ONNX, TensorRT FP32/FP16/INT8.

Runs on the Jetson host (python3 + numpy + docker, nothing else).  It starts its own
Triton container from the image already on the box, on a separate port and model
repo, so the fl-platform Triton and its models are never touched.  Models are
loaded one at a time because the Orin Nano has 7.6 GB shared between CPU and GPU.

Inputs in the working directory: model.pt (TorchScript) and model.onnx, both from
prepare_models.py.  Optional INT8 calibration cache: calib.cache.  Without one, and with BENCH_CALIB_DIR set, the
cache is generated first (int8_calibration.py, TensorRT's entropy calibrator) from training images.

Settings come from env vars so the pipeline can pass them:
    BENCH_IMGSZ=224  BENCH_RUNS=300  BENCH_WARMUP=50  BENCH_VARIANTS=pt,onnx,trt_fp32,trt_fp16,trt_int8
A test_data.zip / calib_data.zip next to this script (carried over from the Colab step) is unpacked and used when
BENCH_TEST_DIR / BENCH_CALIB_DIR are not set.
Optional accuracy pass: set BENCH_TEST_DIR to a folder of class sub-folders (ImageFolder layout, e.g.
fl-platform/data/shards/ALL/test).  Each variant then also classifies every test image through Triton and
reports accuracy, macro-F1, the confusion matrix and one-vs-rest ROC curves.  Needs PIL as well as numpy.
Class index = position of the folder name in sorted order, the same as torchvision's ImageFolder.
INT8 calibration (the INT8 engine is mixed INT8/FP16, see build_trt): BENCH_CALIB_DIR (ImageFolder of TRAINING
images), BENCH_CALIB_N images (default 256, spread
evenly over the classes; any image identical to a test image is skipped so the test set stays unseen).
Writes benchmark_results.json.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

IMAGE = "nvcr.io/nvidia/tritonserver:24.12-py3-igpu"
CONTAINER = "paper-orchestra-triton"
HTTP_PORT = 8100
WORK = Path.cwd()
REPO = WORK / "model_repo"

IMGSZ = int(os.environ.get("BENCH_IMGSZ", "224"))
RUNS = int(os.environ.get("BENCH_RUNS", "300"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "50"))
TEST_DIR = os.environ.get("BENCH_TEST_DIR", "")
CALIB_DIR = os.environ.get("BENCH_CALIB_DIR", "")
CALIB_N = int(os.environ.get("BENCH_CALIB_N", "256"))
CALIB: dict = {}  # what the INT8 calibration did, reported in benchmark_results.json
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet statistics, as in training
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ROC_POINTS = 201  # FPR grid for the stored macro-average ROC
VARIANTS = os.environ.get("BENCH_VARIANTS", "pt,onnx,trt_fp32,trt_fp16,trt_int8").split(",")
DOCKER_GPU = ["docker", "run", "--rm", "--runtime", "nvidia", "-v", f"{WORK}:/work", "-w", "/work"]


def sh(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ----- model repository ------------------------------------------------------

def write_config(name: str, body: str) -> Path:
    d = REPO / name
    (d / "1").mkdir(parents=True, exist_ok=True)
    (d / "config.pbtxt").write_text(body)
    return d


def io_block(in_name: str, out_name: str, out_dim: int) -> str:
    return (f'input [ {{ name: "{in_name}" data_type: TYPE_FP32 dims: [ 1, 3, {IMGSZ}, {IMGSZ} ] }} ]\n'
            f'output [ {{ name: "{out_name}" data_type: TYPE_FP32 dims: [ 1, {out_dim} ] }} ]\n')


def calibration_batches() -> tuple[list[np.ndarray], list[str]]:
    """CALIB_N preprocessed training images, spread evenly over the classes, none of them in the test set."""
    import hashlib

    digest = lambda f: hashlib.md5(f.read_bytes()).hexdigest()
    banned = {digest(f) for f in load_test_set(TEST_DIR)[0]} if TEST_DIR else set()
    paths, labels, classes = load_test_set(CALIB_DIR)  # same ImageFolder reader, here over the training images
    picked: list[list[Path]] = []
    for i in range(len(classes)):
        pool = [f for f, y in zip(paths, labels) if y == i and digest(f) not in banned]
        step = max(1, len(pool) // max(1, -(-CALIB_N // len(classes))))
        picked.append(pool[::step][: -(-CALIB_N // len(classes))])
    chosen = [f for group in zip(*picked) for f in group] + [f for group in picked for f in group[min(map(len, picked)):]]
    return [preprocess_rgb(load_rgb(f)) for f in chosen[:CALIB_N]], [str(f) for f in chosen[:CALIB_N]]


def calibrate() -> dict:
    """Write calib.cache for model.onnx from training images (see int8_calibration.py)."""
    import int8_calibration

    batches, files = calibration_batches()
    info = int8_calibration.make_cache(WORK / "model.onnx", batches, WORK / "calib.cache", fp16=True)
    return {**info, "n_images": len(files), "source": CALIB_DIR}


def build_trt(precision: str) -> Path:
    """Build a TensorRT plan with trtexec inside the Triton image (it ships trtexec)."""
    plan = WORK / f"model_{precision}.plan"
    plan.unlink(missing_ok=True)  # a failed build must not pass for a stale engine
    cmd = DOCKER_GPU + ["--entrypoint", "/usr/src/tensorrt/bin/trtexec", IMAGE, "--onnx=model.onnx",
                        f"--saveEngine={plan.name}", "--memPoolSize=workspace:1024"]
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        # INT8 with FP16 fallback: layers TensorRT leaves unquantised stay in FP16.  Pure INT8 collapses to chance on
        # this attention model (measured: 0.25 accuracy, calibrated or not); mixed precision keeps most of the accuracy.
        cmd += ["--int8", "--fp16", *os.environ.get("BENCH_INT8_ARGS", "").split()]
        if not (WORK / "calib.cache").exists() and CALIB_DIR:
            try:
                CALIB.update(calibrate())
            except Exception as exc:  # fall back to an uncalibrated engine, reported as invalid
                CALIB["error"] = str(exc)[:400]
        if (WORK / "calib.cache").exists():
            cmd.append("--calib=calib.cache")
    proc = sh(cmd)
    if proc.returncode != 0 or not plan.exists():
        raise RuntimeError(f"trtexec {precision} failed: {(proc.stdout + proc.stderr)[-600:]}")
    return plan


def stage(variant: str, out_dim: int | None) -> str:
    """Create the Triton model dir for a variant and return its name.

    ONNX and TensorRT configs leave inputs/outputs out; Triton auto-completes them.
    The libtorch backend cannot, so the pt variant needs the class count.
    """
    name = f"bench_{variant}"
    if variant == "pt":
        if out_dim is None:
            raise RuntimeError("class count unknown; set BENCH_NUM_CLASSES or run another variant first")
        d = write_config(name, f'name: "{name}"\nplatform: "pytorch_libtorch"\nmax_batch_size: 0\n'
                               + io_block("INPUT__0", "OUTPUT__0", out_dim))
        shutil.copy(WORK / "model.pt", d / "1" / "model.pt")
    elif variant == "onnx":
        d = write_config(name, f'name: "{name}"\nbackend: "onnxruntime"\nmax_batch_size: 0\n')
        shutil.copy(WORK / "model.onnx", d / "1" / "model.onnx")
        data = WORK / "model.onnx.data"
        if data.exists():
            shutil.copy(data, d / "1" / "model.onnx.data")
    else:  # trt_fp32 | trt_fp16 | trt_int8
        plan = build_trt(variant.split("_")[1])
        d = write_config(name, f'name: "{name}"\nplatform: "tensorrt_plan"\nmax_batch_size: 0\n')
        shutil.copy(plan, d / "1" / "model.plan")
    return name


# ----- Triton client (KServe v2 over HTTP, binary tensor extension) ----------

def http_call(method: str, path: str, body: bytes = b"", headers: dict | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", HTTP_PORT, timeout=120)
    conn.request(method, path, body, headers or {})
    r = conn.getresponse()
    return r.status, dict(r.getheaders()), r.read()


def model_io(name: str) -> tuple[str, str, int]:
    _, _, raw = http_call("GET", f"/v2/models/{name}")
    meta = json.loads(raw)
    return meta["inputs"][0]["name"], meta["outputs"][0]["name"], int(meta["outputs"][0]["shape"][-1])


def infer_once(name: str, in_name: str, payload: bytes, header: bytes):
    status, _, _ = http_call("POST", f"/v2/models/{name}/infer", header + payload,
                             {"Inference-Header-Content-Length": str(len(header))})
    if status != 200:
        raise RuntimeError(f"infer returned HTTP {status}")


def server_compute_ms(name: str) -> tuple[int, float]:
    _, _, raw = http_call("GET", f"/v2/models/{name}/stats")
    s = json.loads(raw)["model_stats"][0]["inference_stats"]["compute_infer"]
    return int(s["count"]), int(s["ns"]) / 1e6


def bench(name: str) -> tuple[dict, int]:
    in_name, out_name, num_classes = model_io(name)
    x = np.random.rand(1, 3, IMGSZ, IMGSZ).astype(np.float32).tobytes()
    header = json.dumps({
        "inputs": [{"name": in_name, "shape": [1, 3, IMGSZ, IMGSZ], "datatype": "FP32",
                    "parameters": {"binary_data_size": len(x)}}],
        "outputs": [{"name": out_name, "parameters": {"binary_data": True}}],
    }).encode()
    for _ in range(WARMUP):
        infer_once(name, in_name, x, header)
    n0, ms0 = server_compute_ms(name)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        infer_once(name, in_name, x, header)
        times.append((time.perf_counter() - t0) * 1000)
    n1, ms1 = server_compute_ms(name)
    t = np.asarray(times)
    return {
        "client_mean_ms": float(t.mean()), "client_std_ms": float(t.std()),
        "client_p50_ms": float(np.percentile(t, 50)), "client_p95_ms": float(np.percentile(t, 95)),
        "client_p99_ms": float(np.percentile(t, 99)), "fps": float(1000.0 / t.mean()),
        "server_compute_mean_ms": (ms1 - ms0) / max(n1 - n0, 1),
    }, num_classes


# ----- accuracy: test set, logits, metrics -----------------------------------

def load_test_set(root: str) -> tuple[list[Path], np.ndarray, list[str]]:
    """Image paths, integer labels and class names from an ImageFolder-style directory."""
    base = Path(root)
    classes = sorted(d.name for d in base.iterdir() if d.is_dir())
    paths, labels = [], []
    for i, c in enumerate(classes):
        for f in sorted((base / c).iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                paths.append(f)
                labels.append(i)
    if not paths:
        raise RuntimeError(f"no images under {root}")
    return paths, np.asarray(labels), classes


def preprocess_rgb(rgb: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 RGB -> float32 1x3xSxS, resized and normalised like the training eval transform.

    PREPROCESS=letterbox (set for photographs) keeps the aspect ratio and pads, through imgprep.py, the one path the
    trainer and the probe use too; the default is the straight resize the leukemia experiments use.
    """
    if os.environ.get("PREPROCESS", "squash") == "letterbox":
        import imgprep

        return imgprep.preprocess_rgb(rgb, IMGSZ, "letterbox")
    from PIL import Image

    img = Image.fromarray(rgb).resize((IMGSZ, IMGSZ), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def infer_logits(name: str, in_name: str, out_name: str, x: np.ndarray) -> np.ndarray:
    payload = x.astype(np.float32).tobytes()
    header = json.dumps({
        "inputs": [{"name": in_name, "shape": list(x.shape), "datatype": "FP32",
                    "parameters": {"binary_data_size": len(payload)}}],
        "outputs": [{"name": out_name, "parameters": {"binary_data": True}}],
    }).encode()
    status, hdrs, body = http_call("POST", f"/v2/models/{name}/infer", header + payload,
                                   {"Inference-Header-Content-Length": str(len(header))})
    if status != 200:
        raise RuntimeError(f"infer returned HTTP {status}: {body[:200]!r}")
    n = int({k.lower(): v for k, v in hdrs.items()}["inference-header-content-length"])
    return np.frombuffer(body[n:], dtype=np.float32).reshape(-1)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def confusion_matrix(y: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return cm


def macro_f1(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(axis=0), 1)
    rec = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return float(f1.mean())


def roc_ovr(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact one-vs-rest ROC (ties handled) -> (fpr, tpr) corner points from (0, 0) to (1, 1), and the AUC.

    Corner points on a straight run are dropped, so the stored curve is the same polyline in far fewer points.
    """
    order = np.argsort(-score, kind="stable")
    s, t = score[order], y_true[order].astype(float)
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]  # last index of each distinct score
    tps, fps = np.cumsum(t)[last], np.cumsum(1 - t)[last]
    pos, neg = t.sum(), (1 - t).sum()
    if pos == 0 or neg == 0:
        raise ValueError("ROC needs both positives and negatives")
    tpr, fpr = np.r_[0.0, tps / pos], np.r_[0.0, fps / neg]
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    dx, dy = np.diff(fpr), np.diff(tpr)
    turn = np.abs(dx[1:] * dy[:-1] - dx[:-1] * dy[1:]) > 1e-12  # direction changes at the interior points
    keep = np.r_[True, turn, True]
    return fpr[keep], tpr[keep], auc


def accuracy_report(logits: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    k = len(classes)
    pred = logits.argmax(axis=1)
    cm = confusion_matrix(y, pred, k)
    prob = softmax(logits)
    roc, tprs, aucs = {}, [], []
    grid = np.linspace(0, 1, ROC_POINTS)
    for i, c in enumerate(classes):
        fpr, tpr, auc = roc_ovr(y == i, prob[:, i])
        roc[c] = {"fpr": [round(float(v), 5) for v in fpr], "tpr": [round(float(v), 5) for v in tpr],
                  "auc": round(auc, 4)}
        tprs.append(np.interp(grid, fpr, tpr))
        aucs.append(auc)
    return {
        "n_images": int(len(y)), "classes": classes,
        "accuracy": round(float((pred == y).mean()), 4), "macro_f1": round(macro_f1(cm), 4),
        "macro_auc": round(float(np.mean(aucs)), 4),
        "confusion_matrix": cm.tolist(),
        "predictions": pred.tolist() if len(pred) <= 2000 else None,  # to compare with the model that was trained
        "roc": {"fpr": [round(float(v), 5) for v in grid], "macro_tpr": [round(float(v), 5) for v in np.mean(tprs, axis=0)],
                "classes": roc},
    }


def evaluate(name: str, paths: list[Path], y: np.ndarray, classes: list[str]) -> dict:
    in_name, out_name, _ = model_io(name)
    logits = np.stack([infer_logits(name, in_name, out_name, preprocess_rgb(load_rgb(p))) for p in paths])
    return accuracy_report(logits, y, classes)


# ----- container lifecycle ---------------------------------------------------

def start_triton() -> None:
    sh(["docker", "rm", "-f", CONTAINER])
    proc = sh(["docker", "run", "-d", "--name", CONTAINER, "--runtime", "nvidia",
               "-p", f"127.0.0.1:{HTTP_PORT}:8000", "-v", f"{REPO}:/models", IMAGE,
               "tritonserver", "--model-repository=/models", "--model-control-mode=explicit",
               "--log-verbose=0"])
    if proc.returncode != 0:
        raise RuntimeError(f"could not start Triton: {proc.stderr[-400:]}")
    for _ in range(60):
        try:
            if http_call("GET", "/v2/health/ready")[0] == 200:
                return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError("Triton did not become ready in 60 s: " + sh(["docker", "logs", "--tail", "20", CONTAINER]).stdout[-400:])


def device_label() -> str:
    """Board model and JetPack/L4T release, read from the Jetson itself."""
    try:
        model = Path("/proc/device-tree/model").read_text().strip("\x00\n ")
    except OSError:
        model = "NVIDIA Jetson"
    try:
        rel = Path("/etc/nv_tegra_release").read_text().split(",")[0].lstrip("# ").strip()
    except OSError:
        rel = "unknown L4T"
    return f"{model} ({rel})"


def unpack_data() -> None:
    """test_data.zip and calib_data.zip, carried over from the Colab step, become the test and calibration folders."""
    global TEST_DIR, CALIB_DIR
    import zipfile

    for name, folder in (("test_data.zip", "test_images"), ("calib_data.zip", "calib_images")):
        if (WORK / name).exists():
            shutil.rmtree(WORK / folder, ignore_errors=True)
            zipfile.ZipFile(WORK / name).extractall(WORK / folder)
    if not TEST_DIR and (WORK / "test_images").is_dir():
        TEST_DIR = str(WORK / "test_images")
    if not CALIB_DIR and (WORK / "calib_images").is_dir():
        CALIB_DIR = str(WORK / "calib_images")


def stage_models() -> None:
    """Copy model files from BENCH_MODEL_DIR (already on the Jetson) into the working directory."""
    src = os.environ.get("BENCH_MODEL_DIR")
    if not src:
        return
    for name in ("model.pt", "model.onnx", "model.onnx.data", "calib.cache"):
        f = Path(src) / name
        if f.exists() and not (WORK / name).exists():
            shutil.copy(f, WORK / name)


def main() -> None:
    stage_models()
    unpack_data()
    results: dict = {
        "device": device_label(), "imgsz": IMGSZ, "batch": 1, "runs": RUNS, "warmup": WARMUP,
        "image": IMAGE, "int8_calibrated": (WORK / "calib.cache").exists(), "variants": {},
    }
    out_dim = int(os.environ["BENCH_NUM_CLASSES"]) if os.environ.get("BENCH_NUM_CLASSES") else None
    test = load_test_set(TEST_DIR) if TEST_DIR else None
    if test:
        results["test_set"] = {"dir": TEST_DIR, "n_images": len(test[0]), "classes": test[2]}
    shutil.rmtree(REPO, ignore_errors=True)
    REPO.mkdir()

    try:
        start_triton()
        # pt goes last: its config needs the class count, read from an earlier variant
        for v in sorted(VARIANTS, key=lambda v: v == "pt"):
            name = f"bench_{v}"
            try:
                stage(v, out_dim)
                status, _, body = http_call("POST", f"/v2/repository/models/{name}/load")
                if status != 200:
                    raise RuntimeError(body.decode()[:600])
                res, out_dim = bench(name)
                if v == "trt_int8":
                    results["int8_calibrated"] = (WORK / "calib.cache").exists()
                res["status"] = "success"
                res["size_mb"] = sum(f.stat().st_size for f in (REPO / name / "1").iterdir()) / 1e6
                if test:
                    try:  # a failed accuracy pass must not lose this variant's latency numbers
                        res["accuracy"] = evaluate(name, *test)
                        if v == "trt_int8" and not (WORK / "calib.cache").exists():
                            res["accuracy"]["valid"] = False
                            res["accuracy"]["note"] = "INT8 engine built without a calibration cache"
                    except Exception as exc:
                        res["accuracy_error"] = str(exc)[:400]
                results["variants"][v] = res
            except Exception as exc:  # one failed variant should not sink the rest
                results["variants"][v] = {"status": "failed", "error": str(exc)[:600]}
            finally:
                http_call("POST", f"/v2/repository/models/{name}/unload")
    finally:
        sh(["docker", "rm", "-f", CONTAINER])

    if CALIB:
        results["int8_calibration"] = CALIB
    (WORK / "benchmark_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
