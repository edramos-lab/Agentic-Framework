"""Person / background masks for photographs, from a pretrained segmentation network (torchvision DeepLabV3).

person_masks(records, image_dir) -> {record path: bool array at the cached image size}, computed once and cached in
DATA_ROOT/person_masks.npz.  Run it on the GPU when there is one.  The network is trained on the 'person' class of
COCO-style data, so it marks the whole body (clothes, legs, feet) and leaves the room and the standing platform as
background; check a few masks by eye before relying on them for a new kind of photograph.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(os.environ.get("DATA_ROOT", "/content/data"))
_model = None


def _load():
    global _model
    if _model is None:
        import torch
        from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights as W
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

        weights = W.DEFAULT
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _model = (deeplabv3_mobilenet_v3_large(weights=weights).eval().to(dev), weights.transforms(),
                  weights.meta["categories"].index("person"), dev)
    return _model


def person_mask(img: Image.Image) -> np.ndarray:
    """Boolean mask (same height and width as `img`) of the pixels the network calls 'person'."""
    import torch

    model, tf, person, dev = _load()
    with torch.no_grad():
        out = model(tf(img.convert("RGB")).unsqueeze(0).to(dev))["out"][0]
    m = (out.argmax(0) == person).cpu().numpy().astype(np.uint8) * 255
    m = np.asarray(Image.fromarray(m).resize(img.size, Image.NEAREST)) > 127
    try:  # close pin-holes and drop specks
        import cv2

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k).astype(bool)
    except ImportError:
        pass
    return m


def person_masks(paths: list[str], image_dir: str | Path, cache: Path | None = None) -> dict[str, np.ndarray]:
    """Masks for the images at image_dir/<path>, from the cache when it already holds them all."""
    cache = cache or ROOT / "person_masks.npz"
    stored: dict[str, tuple] = {}
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)["data"].item()
        if set(paths) <= set(stored):
            return {p: np.unpackbits(stored[p][1])[: stored[p][0][0] * stored[p][0][1]].reshape(stored[p][0]).astype(bool) for p in paths}
    out = {p: person_mask(Image.open(Path(image_dir) / p)) for p in paths}
    packed = {p: (m.shape, np.packbits(m)) for p, m in out.items()}  # bits, not bytes: the masks stay small
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, data=np.array(packed, dtype=object))
    return out
