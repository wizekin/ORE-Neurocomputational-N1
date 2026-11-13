"""Synthetic stimuli for repeatable development and sanity-check experiments."""
from __future__ import annotations

import numpy as np


def oriented_grating(
    shape: tuple[int, int] = (96, 96),
    theta: float = 0.0,
    wavelength: float = 8.0,
    phase: float = 0.0,
    contrast: float = 0.45,
    mean_luminance: float = 0.5,
) -> np.ndarray:
    """Create a sinusoidal grating whose carrier normal is at angle theta."""
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    coordinate = xx * np.cos(theta) + yy * np.sin(theta)
    image = mean_luminance + contrast * np.sin(2 * np.pi * coordinate / wavelength + phase)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def checkerboard(shape: tuple[int, int] = (96, 96), cell_size: int = 8) -> np.ndarray:
    yy, xx = np.indices(shape)
    pattern = ((yy // cell_size + xx // cell_size) % 2).astype(np.float32)
    return 0.15 + 0.7 * pattern
