"""One image preprocessing path, for training, probing and the Jetson benchmark (no torch needed).

PREPROCESS=squash (default; what the leukemia experiments use): resize straight to size x size.
PREPROCESS=letterbox: resize with the aspect ratio kept and pad to a square with the ImageNet mean colour, so a tall
portrait photograph keeps the angles the labels are about.
The trainer, the probe, triton_benchmark.preprocess_rgb and the INT8 calibration all go through here, so every
number in a paper is computed from the same pixels.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet statistics
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FILL = tuple(int(round(c * 255)) for c in MEAN)


def mode() -> str:
    return os.environ.get("PREPROCESS", "squash")


def to_square(img: Image.Image, size: int, how: str | None = None) -> Image.Image:
    """The image as a size x size RGB image, squashed or letterboxed."""
    how = how or mode()
    img = img.convert("RGB")
    if how != "letterbox":
        return img.resize((size, size), Image.BILINEAR)
    scale = size / max(img.size)
    w, h = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    canvas = Image.new("RGB", (size, size), FILL)
    canvas.paste(img.resize((w, h), Image.BILINEAR), ((size - w) // 2, (size - h) // 2))
    return canvas


def preprocess_rgb(rgb: np.ndarray, size: int, how: str | None = None) -> np.ndarray:
    """uint8 HxWx3 RGB -> float32 1x3xSxS, resized (or letterboxed) and normalised."""
    x = (np.asarray(to_square(Image.fromarray(rgb), size, how), dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])
