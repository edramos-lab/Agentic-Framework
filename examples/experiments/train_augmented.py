"""Train EfficientViT-B0 WITH background/stain augmentation: the model that is deployed to the Jetson (Colab).

Training images are re-rendered, with probability AUG_P, on the background and in the cell colour of four Romanowsky
protocols (stain_render.TRAIN_PROTOCOLS); three protocols are held out for the probe.  Same recipe as
train_baseline.py otherwise.  Writes model.pt, model.onnx and training_history.json.
"""

import os
import sys
from pathlib import Path

for _dir in (Path.cwd(), Path("/content")):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))
os.environ["BG_AUG"], os.environ["OUT_TAG"] = "1", ""

import train_efficientvit  # noqa: E402  (reads its settings from the environment at import)

if __name__ == "__main__":
    train_efficientvit.main()
