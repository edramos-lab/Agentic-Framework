"""Write a TensorRT INT8 calibration cache for an ONNX model, on the Jetson host.

Needs the `tensorrt` Python module (JetPack) and the CUDA runtime library; no pycuda or cuda-python.  Device
memory is handled through ctypes calls to libcudart.  `make_cache` feeds the given float32 NCHW batches to a
calibrator while TensorRT builds (and discards) a calibrated engine; the calibrator leaves its scale table in
`cache_path`, which `trtexec --int8 --calib=<cache_path>` then reads to build the real engine.

The entropy calibrator is the one `trtexec` reads caches with.  (A min-max cache was tried: it gave the right
accuracy only when the engine was built with the host's TensorRT, not through the Triton image's trtexec.)
"""

from __future__ import annotations

import ctypes
import glob
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorrt as trt


def cudart() -> ctypes.CDLL:
    candidates = ["libcudart.so", *sorted(glob.glob("/usr/local/cuda*/lib64/libcudart.so*")),
                  *sorted(glob.glob("/usr/lib/aarch64-linux-gnu/libcudart.so*"))]
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError("libcudart not found; is the CUDA toolkit installed on this host?")


def check(code: int, what: str) -> None:
    if code != 0:
        raise RuntimeError(f"{what} failed with CUDA error {code}")


def _calibrator(base):
    class Calibrator(base):
        def __init__(self, batches: Iterable[np.ndarray], cache_path: Path, shape: tuple[int, ...]):
            super().__init__()
            self.cuda = cudart()
            self.batches = iter(batches)
            self.cache_path = cache_path
            self.shape = shape
            self.nbytes = int(np.prod(shape)) * 4
            self.device = ctypes.c_void_p()
            check(self.cuda.cudaMalloc(ctypes.byref(self.device), ctypes.c_size_t(self.nbytes)), "cudaMalloc")
            self.used = 0

        def get_batch_size(self) -> int:
            return self.shape[0]

        def get_batch(self, names):  # noqa: ARG002
            batch = next(self.batches, None)
            if batch is None:
                return None
            batch = np.ascontiguousarray(batch, dtype=np.float32)
            if batch.shape != self.shape:
                raise ValueError(f"calibration batch has shape {batch.shape}, the network input is {self.shape}")
            check(self.cuda.cudaMemcpy(self.device, batch.ctypes.data_as(ctypes.c_void_p),
                                       ctypes.c_size_t(self.nbytes), 1), "cudaMemcpy")  # 1 = host to device
            self.used += 1
            return [int(self.device.value)]

        def read_calibration_cache(self):
            return None  # always calibrate from the supplied images

        def write_calibration_cache(self, cache) -> None:
            self.cache_path.write_bytes(bytes(cache))

        def free(self) -> None:
            self.cuda.cudaFree(self.device)

    return Calibrator


def build_plan(onnx_path: Path, batches: Iterable[np.ndarray], cache_path: Path, fp16: bool = False,
               workspace_mb: int = 1024):
    """Calibrate and build: returns (serialized plan, summary).  fp16 lets layers that gain nothing stay in FP16."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    inp = network.get_input(0)
    shape = tuple(1 if d < 0 else d for d in inp.shape)  # a dynamic batch axis is calibrated at batch 1
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    config.set_flag(trt.BuilderFlag.INT8)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if any(d < 0 for d in inp.shape):
        profile = builder.create_optimization_profile()
        profile.set_shape(inp.name, shape, shape, shape)
        config.add_optimization_profile(profile)
    calibrator = _calibrator(trt.IInt8EntropyCalibrator2)(batches, cache_path, shape)
    config.int8_calibrator = calibrator
    try:
        plan = builder.build_serialized_network(network, config)
    finally:
        calibrator.free()
    if plan is None or not cache_path.exists():
        raise RuntimeError("TensorRT produced no calibration cache")
    return plan, {"calibration_batches": calibrator.used, "input_shape": list(shape), "algorithm": "entropy",
                  "fp16_fallback": fp16, "tensorrt": trt.__version__}


def make_cache(onnx_path: Path, batches: Iterable[np.ndarray], cache_path: Path, fp16: bool = False,
               workspace_mb: int = 1024) -> dict:
    """Run TensorRT's INT8 calibration of `onnx_path` over `batches`; returns a small summary of what was done."""
    return build_plan(onnx_path, batches, cache_path, fp16, workspace_mb)[1]
