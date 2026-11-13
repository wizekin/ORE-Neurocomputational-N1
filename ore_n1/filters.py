"""Small, normalized oriented Gabor filter bank used by the V1 approximation."""
from __future__ import annotations

import numpy as np


def make_gabor_pair(
    theta: float, sigma: float, wavelength: float,
    aspect_ratio: float, kernel_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-mean, unit-L2 even/odd quadrature Gabor kernels."""
    radius = kernel_size // 2
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    ct, st = np.cos(theta), np.sin(theta)
    x_theta = xx * ct + yy * st
    y_theta = -xx * st + yy * ct
    envelope = np.exp(-0.5 * ((x_theta / sigma) ** 2 + (aspect_ratio * y_theta / sigma) ** 2))
    phase = 2.0 * np.pi * x_theta / wavelength

    def normalize(kernel: np.ndarray) -> np.ndarray:
        kernel = kernel - kernel.mean()
        norm = float(np.sqrt(np.sum(kernel * kernel)))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError("Gabor kernel degenerated; adjust sigma/wavelength/kernel size")
        return (kernel / norm).astype(np.float32)

    return normalize(envelope * np.cos(phase)), normalize(envelope * np.sin(phase))


def build_gabor_bank(config) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    return tuple(
        make_gabor_pair(
            theta, config.gabor_sigma, config.gabor_wavelength,
            config.gabor_aspect_ratio, config.gabor_kernel_size,
        )
        for theta in config.orientations
    )
