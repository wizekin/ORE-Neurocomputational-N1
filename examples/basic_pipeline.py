"""Minimal example for an image supplied by the caller."""
from pathlib import Path

import numpy as np
from PIL import Image  # Optional for this example only; install Pillow separately.

from ore_n1 import N1Processor

image_path = Path("input.png")
image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
result = N1Processor().process(image)
np.save("ore_n1_features.npy", result.feature_vector)
print(f"Extracted {result.feature_vector.size} features")
