"""Cell segmentation and synthetic Romanowsky-stain rendering, shared by the dataset analysis, the trainers and the probe.

The renderers keep the cells and replace what surrounds them: the background is re-drawn in the typical tint of a
staining protocol, and (stain mode) the cells' own colour is shifted towards that protocol.  The protocols are
SYNTHETIC approximations of the characteristic appearance of each Romanowsky stain (background tint and the
blue/red balance of the cells), not photographs of those stains.

The seven protocols are split once, here, so that training and testing never see the same renderings:
  TRAIN_PROTOCOLS     what the background-augmented model is trained on
  HELD_OUT_PROTOCOLS  never seen in training; the probe's verdict about robustness is drawn from these only
"""

from __future__ import annotations

import cv2
import numpy as np

# Appearance of each protocol: background RGB and the shift (da, db) applied to the cells in CIELAB.
# a* > 0 is redder/pinker, b* < 0 is bluer.
PROTOCOLS: dict[str, dict] = {
    "wright":          {"bg": (238, 232, 236), "da": 0.0, "db": -4.0},
    "giemsa":          {"bg": (232, 226, 238), "da": 3.0, "db": -8.0},
    "wright_giemsa":   {"bg": (240, 230, 232), "da": 2.0, "db": -5.0},
    "may_grunwald":    {"bg": (226, 224, 236), "da": 4.0, "db": -10.0},
    "leishman":        {"bg": (244, 226, 224), "da": 6.0, "db": 2.0},
    "jenner_giemsa":   {"bg": (230, 230, 242), "da": 1.0, "db": -9.0},
    "diff_quik":       {"bg": (222, 236, 240), "da": -3.0, "db": -6.0},
}
# Held out: one warm (leishman), one cyan (diff_quik) and one close to the training family (jenner_giemsa).
TRAIN_PROTOCOLS = ["wright", "giemsa", "wright_giemsa", "may_grunwald"]
HELD_OUT_PROTOCOLS = ["leishman", "diff_quik", "jenner_giemsa"]
SCALE_BAR = (0.80, 0.88)  # x, y of the top-left corner of the scale-bar region, as fractions of width/height
NEUTRAL_GREY = (225, 225, 225)
REFERENCE_SIZE = 1024  # the image size the segmentation parameters below were tuned at


def _odd(n: float) -> int:
    return max(1, int(round(n)) | 1)


# ----- segmentation ----------------------------------------------------------

def background_colour(lab: np.ndarray) -> np.ndarray:
    """Median Lab colour of the brightest 25% of pixels: the plasma between the cells."""
    l = lab[..., 0]
    return np.median(lab[l >= np.percentile(l, 75)].reshape(-1, 3), axis=0)


def segment(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of every cell (red and white) in an RGB uint8 smear image of any size.

    Kernel sizes and the minimum cell area are scaled with the image, so a 256 px training image and a 1024 px test
    image are segmented the same way.
    """
    s = max(rgb.shape[:2]) / REFERENCE_SIZE
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(bgr, (0, 0), max(0.5, 2 * s))
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    dist = np.linalg.norm(lab - background_colour(lab), axis=2)
    d8 = np.clip(dist * 4, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cells = (d8 > t).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(5 * s),) * 2)
    cells = cv2.morphologyEx(cells, cv2.MORPH_OPEN, k)
    cells = cv2.morphologyEx(cells, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(9 * s),) * 2))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cells, connectivity=8)
    keep = np.zeros_like(cells)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 150 * s * s:
            keep[lbl == i] = 1
    # fill holes: flood the outside from a padded border; whatever stays unflooded lies inside a cell
    inv = cv2.copyMakeBorder((1 - keep).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    h, w = inv.shape
    cv2.floodFill(inv, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
    keep[inv[1:-1, 1:-1] == 1] = 1
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(9 * s),) * 2)
    return cv2.dilate(keep, grow).astype(bool)


# ----- renderers -------------------------------------------------------------

def feather(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    s = max(mask.shape[:2]) / REFERENCE_SIZE
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(0.5, sigma * s))[..., None]


def blend(fg: np.ndarray, bg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a = feather(mask)
    return np.clip(fg * a + bg * (1 - a), 0, 255).astype(np.uint8)


def erase(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint the masked pixels from the surrounding background."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    small = cv2.resize(bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = cv2.inpaint(small, m, 5, cv2.INPAINT_TELEA)
    out = cv2.resize(out, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return blend(rgb.astype(np.float32), cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32), ~mask)


def flat_background(shape: tuple[int, int, int], colour: tuple[int, int, int], seed: int) -> np.ndarray:
    """A plain background with the gentle low-frequency unevenness and fine grain of a real slide."""
    rng = np.random.default_rng(seed)
    h, w = shape[:2]
    low = cv2.resize(rng.normal(0, 1, (6, 8)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    grain = rng.normal(0, 1, (h, w)).astype(np.float32)
    bg = np.empty(shape, np.float32)
    bg[...] = np.array(colour, np.float32)
    bg += (3.0 * low + 1.0 * grain)[..., None]
    return np.clip(bg, 0, 255)


def shift_stain(rgb: np.ndarray, mask: np.ndarray, da: float, db: float) -> np.ndarray:
    """Shift the cells' CIELAB a*/b* (OpenCV stores them with +128 offset)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    shifted = lab.copy()
    shifted[..., 1] = np.clip(lab[..., 1] + da, 0, 255)
    shifted[..., 2] = np.clip(lab[..., 2] + db, 0, 255)
    out = cv2.cvtColor(shifted.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    return blend(out, rgb.astype(np.float32), mask)


def render(rgb: np.ndarray, cells: np.ndarray, protocol: dict, seed: int, stain: bool) -> np.ndarray:
    """The cells on the protocol's background; with `stain`, also in the protocol's cell colour."""
    bg = flat_background(rgb.shape, protocol["bg"], seed)
    fg = shift_stain(rgb, cells, protocol["da"], protocol["db"]) if stain else rgb
    return blend(fg.astype(np.float32), bg, cells)


def random_train_protocol(rng: np.random.Generator) -> dict:
    """A training protocol with its parameters jittered a little, so the model sees a neighbourhood, not 4 points."""
    p = PROTOCOLS[TRAIN_PROTOCOLS[rng.integers(len(TRAIN_PROTOCOLS))]]
    return {"bg": tuple(int(np.clip(c + rng.integers(-6, 7), 0, 255)) for c in p["bg"]),
            "da": p["da"] + rng.uniform(-1.5, 1.5), "db": p["db"] + rng.uniform(-1.5, 1.5)}
