"""Validated parameters for the N2 biological visual-processing approximation."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True, slots=True)
class N2Config:
    scales: tuple[float, ...] = (1.0, 2.0, 4.0)
    contour_sigma: float = 1.5
    contour_distance: int = 3
    contour_alignment_power: float = 2.0
    completion_iterations: int = 3
    completion_decay: float = 0.65
    ownership_temperature: float = 0.15
    figure_ground_smoothness: float = 0.8
    curvature_sigma: float = 1.0
    fragment_threshold: float = 0.20
    graph_link_distance: float = 6.0
    crowding_eccentricity_slope: float = 0.15
    crowding_base: float = 2.0
    crowding_strength: float = 0.35
    invariance_temperature: float = 0.20
    recurrent_iterations: int = 5
    recurrent_tolerance: float = 1e-4
    recurrent_feedback: float = 0.20
    recurrent_lateral: float = 0.15
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if not self.scales or any(not isfinite(s) or s <= 0 for s in self.scales):
            raise ValueError("scales must be finite positive values")
        positive = ("contour_sigma", "contour_alignment_power", "ownership_temperature",
                    "curvature_sigma", "fragment_threshold", "graph_link_distance",
                    "invariance_temperature", "recurrent_tolerance", "epsilon")
        for name in positive:
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.contour_distance < 1 or self.completion_iterations < 0 or self.recurrent_iterations < 1:
            raise ValueError("iteration counts and contour distance are invalid")
        if not 0 <= self.completion_decay <= 1:
            raise ValueError("completion_decay must lie in [0, 1]")
        if not 0 <= self.recurrent_feedback <= 1 or not 0 <= self.recurrent_lateral <= 1:
            raise ValueError("recurrent weights must lie in [0, 1]")
