"""Benchmark one model through Triton on the Jetson: PyTorch, ONNX, TensorRT FP32/FP16/INT8.

Runs on the Jetson host (python3 + numpy + docker, nothing else).  It starts its own
Triton container from the image already on the box, on a separate port and model
repo, so the fl-platform Triton and its models are never touched.  Models are
loaded one at a time because the Orin Nano has 7.6 GB shared between CPU and GPU.

Inputs in the working directory: model.pt (TorchScript) and model.onnx, both from
prepare_models.py.  Optional INT8 calibration cache: calib.cache.

Settings come from env vars so the pipeline can pass them:
    BENCH_IMGSZ=224  BENCH_RUNS=300  BENCH_WARMUP=50  BENCH_VARIANTS=pt,onnx,trt_fp32,trt_fp16,trt_int8
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


def build_trt(precision: str) -> Path:
    """Build a TensorRT plan with trtexec inside the Triton image (it ships trtexec)."""
    plan = WORK / f"model_{precision}.plan"
    cmd = DOCKER_GPU + ["--entrypoint", "/usr/src/tensorrt/bin/trtexec", IMAGE, "--onnx=model.onnx",
                        f"--saveEngine={plan.name}", "--memPoolSize=workspace:1024"]
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")
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


def main() -> None:
    results: dict = {
        "device": device_label(), "imgsz": IMGSZ, "batch": 1, "runs": RUNS, "warmup": WARMUP,
        "image": IMAGE, "int8_calibrated": (WORK / "calib.cache").exists(), "variants": {},
    }
    out_dim = int(os.environ["BENCH_NUM_CLASSES"]) if os.environ.get("BENCH_NUM_CLASSES") else None
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
                res["status"] = "success"
                res["size_mb"] = sum(f.stat().st_size for f in (REPO / name / "1").iterdir()) / 1e6
                results["variants"][v] = res
            except Exception as exc:  # one failed variant should not sink the rest
                results["variants"][v] = {"status": "failed", "error": str(exc)[:600]}
            finally:
                http_call("POST", f"/v2/repository/models/{name}/unload")
    finally:
        sh(["docker", "rm", "-f", CONTAINER])

    (WORK / "benchmark_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
