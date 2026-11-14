"""Validated configuration for N3 recurrent orthographic prediction."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True, slots=True)
class N3Config:
    """Numerical controls; defaults are hypotheses, not fitted biological values."""
    transition_alpha: float = 0.5
    visual_temperature: float = 0.18
    unknown_similarity_threshold: float = 0.25
    unknown_temperature: float = 0.10
    sensory_precision_base: float = 1.0
    attention_gain: float = 1.0
    prediction_gain: float = 0.8
    damping: float = 0.45
    max_iterations: int = 20
    convergence_tolerance: float = 1e-5
    unigram_weight: float = 0.35
    neighbor_weight: float = 0.325
    top_k: int = 5
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        positive = ("transition_alpha", "visual_temperature", "unknown_temperature",
                    "sensory_precision_base", "convergence_tolerance", "epsilon")
        for name in positive:
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        nonnegative = ("attention_gain", "prediction_gain", "unigram_weight", "neighbor_weight")
        for name in nonnegative:
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not -1.0 <= self.unknown_similarity_threshold <= 1.0:
            raise ValueError("unknown_similarity_threshold must lie in [-1, 1]")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("damping must lie in (0, 1]")
        if self.max_iterations < 1 or self.top_k < 1:
            raise ValueError("max_iterations and top_k must be positive integers")
        if self.unigram_weight + 2.0 * self.neighbor_weight <= 0:
            raise ValueError("at least one contextual-prior weight must be positive")
