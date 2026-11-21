"""Fine-tune EfficientViT-B0 on the packed blood-cell images and export it for the Jetson (runs on Colab).

Reads the training pool of kaggle_data.py (class sub-folders of 256x256 JPEGs; or the zip named by TRAIN_ZIP), holds
out 20% for validation,
trains with the augmentations and class-weighted loss of the fl-platform client, and keeps the epoch with the best
validation accuracy.  Writes per-epoch curves to training_history.json and the best model as model.pt
(TorchScript) and model.onnx (opset 17, input "input", output "output", batch 1) for the Triton benchmark.
Class index = position of the folder name in sorted order, as torchvision's ImageFolder does.

Background augmentation (BG_AUG=1, see train_augmented.py): with probability AUG_P a training image is re-rendered
with its cells kept and its background (and, half of the time, the cells' own colour) replaced by one of the
TRAIN_PROTOCOLS of stain_render.py, with jittered parameters.  The HELD_OUT_PROTOCOLS are never used here, so the
probe can test the model on stains it has not seen.  Validation images are never altered.
train_baseline.py trains the same model without it.

Settings (env): TRAIN_ZIP="" (only to train from a zip of class folders)  EPOCHS=30  BATCH=32  LR=0.001  IMGSZ=224
                ARCH=efficientvit_b0.r224_in1k  SEED=42  VAL_SPLIT=0.2  TRAIN_LIMIT=0 (images, 0 = all)
                BG_AUG=0  AUG_P=0.7  OUT_TAG="" (file-name suffix: model<tag>.pt, training_history<tag>.json)
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    import timm
    import onnx  # noqa: F401  (needed by torch.onnx.export)
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm", "onnx"])
    import timm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

for _dir in (Path.cwd(), Path("/content")):  # a Colab kernel does not always put the script's folder on sys.path
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

ENV = os.environ.get
ZIP = ENV("TRAIN_ZIP", "")
EPOCHS = int(ENV("EPOCHS", "30"))
BATCH = int(ENV("BATCH", "32"))
LR = float(ENV("LR", "0.001"))
IMGSZ = int(ENV("IMGSZ", "224"))
ARCH = ENV("ARCH", "efficientvit_b0.r224_in1k")
SEED = int(ENV("SEED", "42"))
VAL_SPLIT = float(ENV("VAL_SPLIT", "0.2"))
LIMIT = int(ENV("TRAIN_LIMIT", "0"))
BG_AUG = ENV("BG_AUG", "0") == "1"
AUG_P = float(ENV("AUG_P", "0.7"))
TAG = ENV("OUT_TAG", "")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class BackgroundAugmented(datasets.ImageFolder):
    """ImageFolder whose images are, with probability AUG_P, re-rendered on the background of a training protocol."""

    def __init__(self, root, transform, masks):
        super().__init__(root, transform)
        self.masks, self.calls = masks, 0

    def __getitem__(self, index):
        from PIL import Image

        import stain_render as sr

        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        self.calls += 1
        rng = np.random.default_rng([SEED, index, self.calls])
        if rng.random() < AUG_P:
            rgb = sr.render(np.asarray(img), self.masks[index], sr.random_train_protocol(rng),
                            int(rng.integers(1 << 30)), stain=bool(rng.random() < 0.5))
            img = Image.fromarray(rgb)
        return self.transform(img), target


def cell_masks(root: Path) -> dict[int, np.ndarray]:
    """Segment every training image once; the masks are shared with the data-loader workers."""
    import stain_render as sr
    from PIL import Image

    base = datasets.ImageFolder(root)
    return {i: sr.segment(np.asarray(Image.open(p).convert("RGB"))) for i, (p, _) in enumerate(base.samples)}


def loaders():
    if ZIP:
        root = Path("train_images")
        if not root.exists():
            with zipfile.ZipFile(ZIP) as z:
                z.extractall(root)
    else:
        import kaggle_data

        root = Path(kaggle_data.prepare()["train_dir"])
    # ToTensor comes first so every augmentation runs on tensors (the PIL colour-jitter path breaks on some
    # torchvision/numpy combinations)
    train_tf = transforms.Compose([
        transforms.Resize((IMGSZ + 32, IMGSZ + 32)), transforms.ToTensor(), transforms.RandomCrop(IMGSZ),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.05), transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.Normalize(MEAN, STD), transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    val_tf = transforms.Compose([transforms.Resize((IMGSZ, IMGSZ)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    full_val = datasets.ImageFolder(root, val_tf)
    full_train = BackgroundAugmented(root, train_tf, cell_masks(root)) if BG_AUG else datasets.ImageFolder(root, train_tf)
    idx = list(range(len(full_train)))
    random.Random(SEED).shuffle(idx)  # the same seeded split the fl-platform client uses
    if LIMIT:
        idx = idx[:LIMIT]
    n_val = int(len(idx) * VAL_SPLIT)
    tr, va = idx[n_val:], idx[:n_val]
    counts = np.bincount([full_train.targets[i] for i in tr], minlength=len(full_train.classes)).astype(float)
    weights = 1.0 / np.maximum(counts, 1)
    weights = weights / weights.sum() * len(counts)
    kw = dict(batch_size=BATCH, num_workers=4, pin_memory=torch.cuda.is_available())
    return (DataLoader(Subset(full_train, tr), shuffle=True, **kw), DataLoader(Subset(full_val, va), shuffle=False, **kw),
            full_train.classes, torch.tensor(weights, dtype=torch.float32), len(tr), len(va))


def run_epoch(model, loader, loss_fn, device, opt=None, scaler=None):
    model.train(opt is not None)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            total += len(y)
            correct += (out.argmax(1) == y).sum().item()
            loss_sum += loss.item() * len(y)
    return loss_sum / total, correct / total


def export(model, device) -> None:
    model = model.float().eval().cpu()
    x = torch.rand(1, 3, IMGSZ, IMGSZ)
    torch.jit.trace(model, x).save(f"model{TAG}.pt")
    kw = dict(input_names=["input"], output_names=["output"], opset_version=17)
    try:
        torch.onnx.export(model, x, f"model{TAG}.onnx", dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, x, f"model{TAG}.onnx", **kw)


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, classes, weights, n_train, n_val = loaders()
    model = timm.create_model(ARCH, pretrained=True, num_classes=len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    hist = {"epochs": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, best_state = -1.0, None
    for ep in range(1, EPOCHS + 1):
        tl, ta = run_epoch(model, train, loss_fn, device, opt, scaler)
        vl, va = run_epoch(model, val, loss_fn, device)
        sched.step()
        for k, v in zip(hist, (ep, tl, vl, ta, va)):
            hist[k].append(round(v, 5))
        print(f"epoch {ep}/{EPOCHS} train {tl:.4f}/{ta:.4f} val {vl:.4f}/{va:.4f}", flush=True)
        if va > best:
            best, best_state = va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    export(model, device)
    hist.update({"best_val_acc": round(best, 5), "best_epoch": hist["val_acc"].index(max(hist["val_acc"])) + 1,
                 "classes": classes, "arch": ARCH, "n_train": n_train, "n_val": n_val, "device": str(device),
                 "val_split_seed": SEED, "background_augmentation": BG_AUG,
                 "augmentation_protocols": __import__("stain_render").TRAIN_PROTOCOLS if BG_AUG else [],
                 "augmentation_probability": AUG_P if BG_AUG else 0.0})
    Path(f"training_history{TAG}.json").write_text(json.dumps(hist, indent=2))
    print(json.dumps({k: v for k, v in hist.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()
