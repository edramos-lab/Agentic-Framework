"""Perturbations of a photograph that keep or remove the person, shared by the probe and the augmentation.

All functions take uint8 RGB arrays and a boolean person mask of the same size.  The probe uses them to ask what a
model reads; the augmented training uses a SUBSET of the backgrounds (flat colours, gradients, backgrounds of other
training images), so the probe's other backgrounds (noise textures, backgrounds of images the model never saw) are held out.
"""

from __future__ import annotations

import cv2
import numpy as np

NEUTRAL = (200, 200, 200)


def feather(mask: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)[..., None]


def compose(rgb: np.ndarray, mask: np.ndarray, background: np.ndarray) -> np.ndarray:
    """The person of `rgb` over `background` (same size)."""
    a = feather(mask)
    return np.clip(rgb * a + background * (1 - a), 0, 255).astype(np.uint8)


def flat(shape: tuple, colour: tuple) -> np.ndarray:
    out = np.empty(shape, np.float32)
    out[...] = np.array(colour, np.float32)
    return out


def gradient(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """A smooth two-colour gradient at a random angle."""
    h, w = shape[:2]
    c0, c1 = rng.integers(60, 240, 3), rng.integers(60, 240, 3)
    t = np.cos(a := rng.uniform(0, np.pi)) * np.linspace(0, 1, w)[None, :] + np.sin(a) * np.linspace(0, 1, h)[:, None]
    t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
    return (c0[None, None, :] * (1 - t[..., None]) + c1[None, None, :] * t[..., None]).astype(np.float32)


def texture(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """Coloured band-limited noise: nothing like a room, never used in training."""
    h, w = shape[:2]
    low = rng.normal(0, 1, (max(2, h // 24), max(2, w // 24), 3)).astype(np.float32)
    field = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(150 + 38 * field / max(float(field.std()), 1e-6), 0, 255)


def canvas_of(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The scene of `rgb` with the person painted out (inpainted at low resolution): a background of its own."""
    s = min(1.0, 256 / max(rgb.shape[:2]))
    small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = cv2.resize(cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    painted = cv2.inpaint(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), m, 5, cv2.INPAINT_TELEA)
    return cv2.resize(cv2.cvtColor(painted, cv2.COLOR_BGR2RGB), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)


def fit_to(canvas: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(canvas, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def random_training_background(rgb: np.ndarray, rng: np.random.Generator, pool: list[np.ndarray]) -> np.ndarray:
    """A background of the kinds used in training: a flat colour, a gradient, or the scene of another training image."""
    kind = rng.integers(3) if pool else rng.integers(2)
    if kind == 0:
        return flat(rgb.shape, tuple(int(c) for c in rng.integers(90, 240, 3)))
    if kind == 1:
        return gradient(rgb.shape, rng)
    return fit_to(pool[rng.integers(len(pool))], rgb.shape)


def augment_background(rgb: np.ndarray, mask: np.ndarray, rng: np.random.Generator, pool: list[np.ndarray]) -> np.ndarray:
    return compose(rgb.astype(np.float32), mask, random_training_background(rgb, rng, pool))


def conditions(rgb: np.ndarray, mask: np.ndarray, other_canvas: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Every probe condition of one image.  `other_canvas` is the scene of a different image the model never trained on."""
    rng = np.random.default_rng(seed)
    f = rgb.astype(np.float32)
    out = {"original": rgb}
    out["person_masked"] = compose(f, ~mask, flat(rgb.shape, (124, 116, 104)))  # only the surroundings remain
    out["person_only"] = compose(f, mask, flat(rgb.shape, NEUTRAL))  # a flat colour: a kind of background used in training
    out["bg_other"] = compose(f, mask, fit_to(other_canvas, rgb.shape))
    out["bg_texture"] = compose(f, mask, texture(rgb.shape, rng))
    ys, xs = np.where(mask)
    if len(ys) >= 100:  # framing removed: crop to the person's bounding box (small margin), whatever the camera distance was
        my, mx = int(0.04 * rgb.shape[0]), int(0.04 * rgb.shape[1])
        out["person_box_crop"] = rgb[max(0, ys.min() - my): ys.max() + my + 1, max(0, xs.min() - mx): xs.max() + mx + 1]
    else:
        out["person_box_crop"] = rgb
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    out["grayscale"] = np.stack([gray] * 3, axis=-1)
    out["blur"] = cv2.GaussianBlur(rgb, (0, 0), 4)
    return out
