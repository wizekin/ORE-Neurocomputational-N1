"""Validated controls for the ORE N4 attention/precision-gating stage."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class N4Config:
    """Numerical controls are experimental hypotheses, not measured neural constants."""

    epsilon: float = 1e-8
    attention_temperature: float = 0.7
    entropy_salience_weight: float = 0.35
    error_salience_weight: float = 0.35
    surprise_salience_weight: float = 0.20
    external_salience_weight: float = 0.10
    task_relevance_weight: float = 0.65
    goal_class_weight: float = 0.35
    seed_attention_weight: float = 0.25
    uniform_attention_fraction: float = 0.10
    spatial_smoothing: float = 0.15
    attention_gain: float = 1.0
    prediction_gain: float = 0.7
    task_prior_gain: float = 0.5
    sensory_precision_base: float = 1.0
    sensory_precision_min: float = 0.05
    sensory_precision_max: float = 5.0
    prior_precision_min: float = 0.05
    prior_precision_max: float = 5.0
    damping: float = 0.55
    max_iterations: int = 20
    convergence_tolerance: float = 1e-5
    top_k: int = 5
    abstain_threshold: float = 0.35
    ambiguity_margin: float = 0.08
    strict_upstream_validity: bool = False
    enable_spatial_attention: bool = True
    enable_top_down_attention: bool = True
    enable_class_attention: bool = True
    enable_precision_gating: bool = True

    def __post_init__(self) -> None:
        positive = (
            "epsilon", "attention_temperature", "sensory_precision_base",
            "convergence_tolerance",
        )
        for name in positive:
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        nonnegative = (
            "entropy_salience_weight", "error_salience_weight",
            "surprise_salience_weight", "external_salience_weight",
            "task_relevance_weight", "goal_class_weight", "seed_attention_weight",
            "attention_gain", "prediction_gain", "task_prior_gain",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

        if not 0.0 <= self.uniform_attention_fraction <= 1.0:
            raise ValueError("uniform_attention_fraction must lie in [0, 1]")
        if not 0.0 <= self.spatial_smoothing <= 1.0:
            raise ValueError("spatial_smoothing must lie in [0, 1]")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("damping must lie in (0, 1]")
        if self.max_iterations < 1 or self.top_k < 1:
            raise ValueError("max_iterations and top_k must be positive integers")
        for low_name, high_name in (
            ("sensory_precision_min", "sensory_precision_max"),
            ("prior_precision_min", "prior_precision_max"),
        ):
            low, high = getattr(self, low_name), getattr(self, high_name)
            if not isfinite(low) or not isfinite(high) or low <= 0 or high < low:
                raise ValueError(f"{low_name}/{high_name} must be finite, positive and ordered")
        if not 0.0 <= self.abstain_threshold <= 1.0:
            raise ValueError("abstain_threshold must lie in [0, 1]")
        if not 0.0 <= self.ambiguity_margin <= 1.0:
            raise ValueError("ambiguity_margin must lie in [0, 1]")
        if not (
            self.entropy_salience_weight + self.error_salience_weight
            + self.surprise_salience_weight > 0
        ):
            raise ValueError("at least one internal bottom-up salience weight must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
