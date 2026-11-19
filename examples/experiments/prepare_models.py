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


def main() -> None:
    ap = argparse.ArgumentParser()
    # Env vars let the pipeline run this as a Colab experiment (`colab exec` takes no arguments).
    ap.add_argument("--weights", default=os.environ.get("BENCH_WEIGHTS"))
    ap.add_argument("--imgsz", type=int, default=int(os.environ.get("BENCH_IMGSZ", "224")))
    ap.add_argument("--out", default=os.environ.get("BENCH_OUT", "bench_models"))
    args = ap.parse_args()
    if not args.weights:
        raise SystemExit("pass --weights or set BENCH_WEIGHTS")

    model = torch.load(args.weights, map_location="cpu", weights_only=False)
    if isinstance(model, dict):  # checkpoint wrapper such as {"model": ...}
        model = model.get("model", model)
    if not isinstance(model, torch.nn.Module):
        raise SystemExit("weights hold a state_dict, not a full module; build the model and load it first")
    model = model.float().eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    x = torch.rand(1, 3, args.imgsz, args.imgsz)

    torch.jit.trace(model, x).save(str(out / "model.pt"))
    torch.onnx.export(model, x, str(out / "model.onnx"), input_names=["input"],
                      output_names=["output"], opset_version=17)
    print(f"wrote {out}/model.pt and {out}/model.onnx")


if __name__ == "__main__":
    main()
