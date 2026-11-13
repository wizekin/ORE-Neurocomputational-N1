"""Configuration for the biologically motivated N1 approximation."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi


@dataclass(frozen=True, slots=True)
class N1Config:
    """Numerical parameters for the 12-operation N1 pipeline.

    Spatial scales are expressed in pixels. Angles are radians. Defaults are
    engineering starting points, not fitted physiological constants.
    """

    photoreceptor_sigma: float = 6.0
    photoreceptor_floor: float = 1e-3
    bipolar_sigma: float = 1.0
    center_sigma: float = 1.0
    surround_sigma: float = 3.0
    lgn_sigma: float = 2.0
    lgn_gain_floor: float = 0.10
    lgn_gain_exponent: float = 0.5
    orientations: tuple[float, ...] = (0.0, pi / 4, pi / 2, 3 * pi / 4)
    gabor_sigma: float = 2.5
    gabor_wavelength: float = 4.0
    gabor_aspect_ratio: float = 0.5
    gabor_kernel_size: int = 15
    normalization_sigma: float = 3.0
    normalization_exponent: float = 2.0
    normalization_epsilon: float = 0.10
    pooling_grid: tuple[int, int] = (4, 4)
    temporal_enabled: bool = False
    temporal_tau: float = 0.15
    strict_numerics: bool = True

    def __post_init__(self) -> None:
        positive = (
            "photoreceptor_sigma", "photoreceptor_floor", "bipolar_sigma",
            "center_sigma", "surround_sigma", "lgn_sigma", "lgn_gain_floor",
            "lgn_gain_exponent", "gabor_sigma", "gabor_wavelength",
            "gabor_aspect_ratio", "normalization_sigma",
            "normalization_exponent", "normalization_epsilon", "temporal_tau",
        )
        for name in positive:
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.surround_sigma <= self.center_sigma:
            raise ValueError("surround_sigma must be greater than center_sigma")
        if self.gabor_kernel_size < 5 or self.gabor_kernel_size % 2 == 0:
            raise ValueError("gabor_kernel_size must be an odd integer >= 5")
        if not self.orientations or any(not isfinite(a) for a in self.orientations):
            raise ValueError("orientations must contain finite angles")
        if len(set(round(a % pi, 8) for a in self.orientations)) != len(self.orientations):
            raise ValueError("orientations must be unique modulo pi")
        if len(self.pooling_grid) != 2 or any(int(v) != v or v < 1 for v in self.pooling_grid):
            raise ValueError("pooling_grid must contain two positive integers")

    @property
    def orientation_count(self) -> int:
        return len(self.orientations)
