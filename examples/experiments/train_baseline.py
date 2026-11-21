"""Train EfficientViT-B0 WITHOUT background augmentation: the baseline the augmented model is compared with (Colab).

Same recipe as train_augmented.py (train_efficientvit.py) with the augmentation off.  Writes model_baseline.pt,
model_baseline.onnx and training_history_baseline.json.
"""

import os
import sys
from pathlib import Path

for _dir in (Path.cwd(), Path("/content")):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))
os.environ["BG_AUG"], os.environ["OUT_TAG"] = "0", "_baseline"

import train_efficientvit  # noqa: E402  (reads its settings from the environment at import)

if __name__ == "__main__":
    train_efficientvit.main()
