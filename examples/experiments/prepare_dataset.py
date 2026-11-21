"""Download the blood-smear dataset from Kaggle in Colab, split it, and pack what the Jetson needs.

The split itself is made by kaggle_data.py, which every Colab experiment calls, so all of them see identical
train and test images.  This step exists to hand the Jetson its two inputs, as files the pipeline carries over:
  test_data.zip    the held-out test images, byte for byte at original resolution (the Jetson benchmark classifies them)
  calib_data.zip   CALIB_N training images (256x256 JPEG, spread evenly over the classes) for the INT8 calibration
  split_info.json  dataset, seed, images per class and split, exact duplicates dropped

Settings (env): CALIB_N=256, plus those of kaggle_data.py (KAGGLE_DATASET, SPLIT_SEED, TEST_FRACTION).
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

for _dir in (Path.cwd(), Path("/content")):  # a Colab kernel does not always put the script's folder on sys.path
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import kaggle_data  # noqa: E402

CALIB_N = int(os.environ.get("CALIB_N", "256"))


def main() -> None:
    data = kaggle_data.prepare()
    info, test_dir, train_dir = data["info"], Path(data["test_dir"]), Path(data["train_dir"])
    with zipfile.ZipFile("test_data.zip", "w", zipfile.ZIP_STORED) as z:
        for cls in sorted(d for d in test_dir.iterdir() if d.is_dir()):
            for f in sorted(cls.iterdir()):
                z.write(f, f"{cls.name}/{f.name}")  # writes the linked file's bytes, not the link
    classes = info["classes"]
    per_class = -(-CALIB_N // len(classes))
    n_calib = 0
    with zipfile.ZipFile("calib_data.zip", "w", zipfile.ZIP_STORED) as z:
        for cls in classes:
            files = sorted((train_dir / cls).iterdir())
            step = max(1, len(files) // per_class)
            for f in files[::step][:per_class]:
                z.write(f, f"{cls}/{f.name}")
                n_calib += 1
    info = {**info, "calibration_images": n_calib,
            "test_zip_mb": round(Path("test_data.zip").stat().st_size / 1e6, 1),
            "calib_zip_mb": round(Path("calib_data.zip").stat().st_size / 1e6, 1)}
    Path("split_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
