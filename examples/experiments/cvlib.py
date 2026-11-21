"""Training, prediction and metrics for small-data image classification (shared by train_cv.py and shortcut_probe.py).

Images are kept in memory as PIL images (the cache of data_source.py is small).  Every model sees the same pixels:
letterboxed (or squashed, PREPROCESS) by imgprep, then, for training only, light geometric and photometric
augmentation and, optionally, a replaced background (photo_conditions.augment_background).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

for _dir in (Path.cwd(), Path("/content")):  # a Colab kernel does not always put the script's folder on sys.path
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import numpy as np
from PIL import Image

import imgprep

try:
    import timm
    import onnx  # noqa: F401  (needed by torch.onnx.export)
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm", "onnx"])
    import timm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = imgprep.MEAN.tolist(), imgprep.STD.tolist()


class ImageList(Dataset):
    """PIL images with labels; `train` adds augmentation; `masks` and `pool` add the background replacement."""

    def __init__(self, images, labels, size=224, train=False, masks=None, pool=None, aug_p=0.7, seed=0):
        self.images, self.labels, self.size, self.train = images, np.asarray(labels), size, train
        self.masks, self.pool, self.aug_p, self.seed, self.calls = masks, pool, aug_p, seed, 0
        self.aug = T.Compose([
            T.RandomHorizontalFlip(), T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=MEAN),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03), T.GaussianBlur(3, sigma=(0.1, 1.5))])
        self.norm = T.Normalize(MEAN, STD)
        self.erase = T.RandomErasing(p=0.2, scale=(0.02, 0.1))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.train and self.masks is not None:
            self.calls += 1
            rng = np.random.default_rng([self.seed, i, self.calls])
            if rng.random() < self.aug_p:
                import photo_conditions as pc

                img = Image.fromarray(pc.augment_background(np.asarray(img), self.masks[i], rng, self.pool or []))
        t = T.functional.to_tensor(imgprep.to_square(img, self.size))
        if self.train:
            t = self.erase(self.norm(self.aug(t)))
        else:
            t = self.norm(t)
        return t, int(self.labels[i])


def make_model(arch: str, n_classes: int) -> nn.Module:
    return timm.create_model(arch, pretrained=True, num_classes=n_classes).to(DEVICE)


def _scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")


def run_epoch(model, loader, loss_fn, opt=None, scaler=None):
    model.train(opt is not None)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
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


def fit(model, train: ImageList, val: ImageList | None, epochs: int, lr: float = 5e-4, batch: int = 16, seed: int = 0) -> dict:
    """Fine-tune `model` in place; returns the per-epoch curves (train and, if `val` is given, validation)."""
    torch.manual_seed(seed)
    counts = np.bincount(train.labels, minlength=int(train.labels.max()) + 1).astype(float)
    weights = 1.0 / np.maximum(counts, 1)
    weights = torch.tensor(weights / weights.sum() * len(counts), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = _scaler()
    tl = DataLoader(train, batch_size=batch, shuffle=True, num_workers=0, drop_last=len(train) > batch)
    vl = DataLoader(val, batch_size=batch, shuffle=False, num_workers=0) if val is not None else None
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    for _ in range(epochs):
        a, b = run_epoch(model, tl, loss_fn, opt, scaler)
        hist["train_loss"].append(a), hist["train_acc"].append(b)
        if vl is not None:
            c, d = run_epoch(model, vl, loss_fn)
            hist["val_loss"].append(c), hist["val_acc"].append(d)
        sched.step()
    return hist


def predict(model, images, size: int = 224, batch: int = 32) -> np.ndarray:
    """Softmax probabilities, one row per image."""
    model.eval()
    ds = ImageList(images, np.zeros(len(images), int), size)
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=batch):
            out.append(torch.softmax(model(x.to(DEVICE)).float(), 1).cpu().numpy())
    return np.concatenate(out)


def embeddings(arch: str, images, size: int = 224) -> np.ndarray:
    """Frozen pretrained features (pooled, no classifier head)."""
    net = timm.create_model(arch, pretrained=True, num_classes=0).to(DEVICE).eval()
    ds, out = ImageList(images, np.zeros(len(images), int), size), []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=32):
            out.append(net(x.to(DEVICE)).float().cpu().numpy())
    return np.concatenate(out)


def linear_probe(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, n_classes: int, l2: float = 1e-2) -> np.ndarray:
    """Multinomial logistic regression (L-BFGS, L2) on standardised features: probabilities for Xte."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    a, b = torch.tensor((Xtr - mu) / sd, dtype=torch.float32), torch.tensor(ytr)
    w = torch.zeros(a.shape[1], n_classes, requires_grad=True)
    bias = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([w, bias], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(a @ w + bias, b) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return torch.softmax(torch.tensor((Xte - mu) / sd, dtype=torch.float32) @ w + bias, 1).numpy()


# ----- metrics ------------------------------------------------------------------

def confusion(y: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y, pred), 1)
    return cm


def auc_ovr(y: np.ndarray, probs: np.ndarray) -> float:
    """Macro one-vs-rest AUC by the rank (Mann-Whitney) formula, ties counted half; classes without both sides skipped."""
    aucs = []
    for c in range(probs.shape[1]):
        pos, neg = probs[y == c, c], probs[y != c, c]
        if len(pos) and len(neg):
            aucs.append(float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())))
    return float(np.mean(aucs)) if aucs else float("nan")


def ece(y: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of the top-class confidence."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def metrics(y: np.ndarray, probs: np.ndarray) -> dict:
    k = probs.shape[1]
    pred = probs.argmax(1)
    cm = confusion(y, pred, k)
    recall = [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else float("nan") for i in range(k)]
    prec = [float(cm[i, i] / cm[:, i].sum()) if cm[:, i].sum() else 0.0 for i in range(k)]
    f1 = [2 * p * r / (p + r) if p + r > 0 and not np.isnan(r) else 0.0 for p, r in zip(prec, recall)]
    present = [r for r in recall if not np.isnan(r)]
    return {"accuracy": float((pred == y).mean()), "balanced_accuracy": float(np.mean(present)), "macro_f1": float(np.mean(f1)),
            "macro_auc": auc_ovr(y, probs), "ece": ece(y, probs), "per_class_recall": recall, "confusion_matrix": cm.tolist()}


def export(model, size: int = 224, tag: str = "") -> None:
    """TorchScript and ONNX (opset 17, input "input", output "output", batch 1) for the Jetson benchmark."""
    model = model.float().eval().cpu()
    x = torch.rand(1, 3, size, size)
    torch.jit.trace(model, x).save(f"model{tag}.pt")
    kw = dict(input_names=["input"], output_names=["output"], opset_version=17)
    try:
        torch.onnx.export(model, x, f"model{tag}.onnx", dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, x, f"model{tag}.onnx", **kw)
    model.to(DEVICE)
