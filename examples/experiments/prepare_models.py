"""Convert a trained .pt into the two files the Jetson benchmark needs.

Run this on any machine that has torch (torch.onnx needs the `onnx` package):
    python3 prepare_models.py --weights best.pt --imgsz 224 --out ./bench_models

Writes model.pt (TorchScript, for Triton's pytorch backend) and model.onnx (FP32,
fixed batch 1, input "input" / output "output").  Upload both to the Jetson with
experiment_requirements.jetson.upload_files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def build_from_state_dict(state: dict, arch: str | None) -> torch.nn.Module:
    """Rebuild a timm model from a bare state_dict; the class count comes from the head."""
    if not arch:
        raise SystemExit("weights are a state_dict; pass --arch (a timm model name) or set BENCH_ARCH")
    import timm

    head = [k for k, v in state.items() if k.endswith("weight") and v.ndim == 2][-1]
    model = timm.create_model(arch, pretrained=False, num_classes=state[head].shape[0])
    model.load_state_dict(state)
    print(f"built {arch} with {state[head].shape[0]} classes from {head}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    # Env vars let the pipeline run this as a Colab experiment (`colab exec` takes no arguments).
    ap.add_argument("--weights", default=os.environ.get("BENCH_WEIGHTS"))
    ap.add_argument("--imgsz", type=int, default=int(os.environ.get("BENCH_IMGSZ", "224")))
    ap.add_argument("--out", default=os.environ.get("BENCH_OUT", "bench_models"))
    # Needed only when the weights are a bare state_dict rather than a full module
    ap.add_argument("--arch", default=os.environ.get("BENCH_ARCH"), help="timm model name, e.g. efficientvit_b0")
    args = ap.parse_args()
    if not args.weights:
        raise SystemExit("pass --weights or set BENCH_WEIGHTS")

    model = torch.load(args.weights, map_location="cpu", weights_only=False)
    if isinstance(model, dict):  # checkpoint wrapper such as {"model": ...}
        model = model.get("model", model)
    if not isinstance(model, torch.nn.Module):
        model = build_from_state_dict(model, args.arch)
    model = model.float().eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    x = torch.rand(1, 3, args.imgsz, args.imgsz)

    torch.jit.trace(model, x).save(str(out / "model.pt"))
    export = dict(input_names=["input"], output_names=["output"], opset_version=17)
    try:
        torch.onnx.export(model, x, str(out / "model.onnx"), dynamo=False, **export)
    except TypeError:  # older torch has no dynamo flag
        torch.onnx.export(model, x, str(out / "model.onnx"), **export)
    print(f"wrote {out}/model.pt and {out}/model.onnx")


if __name__ == "__main__":
    main()
