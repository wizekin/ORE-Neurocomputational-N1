"""Validated controls for ORE N5 experience-dependent plasticity."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class N5Config:
    """Numerical controls for a bounded, inspectable learning approximation.

    Defaults are engineering hypotheses, not measured synaptic constants.
    """

    epsilon: float = 1e-8
    fast_learning_rate: float = 0.035
    slow_learning_rate: float = 0.0035
    fast_trace_decay: float = 0.70
    slow_trace_decay: float = 0.995
    max_step_norm: float = 0.08
    max_prototype_drift: float = 0.35
    prior_pseudocount_mass: float = 32.0
    transition_learning_rate: float = 0.25
    slow_consolidation_rate: float = 0.02
    replay_capacity: int = 128
    replay_max_samples: int = 24
    replay_decay: float = 0.92
    novelty_floor: float = 0.15
    novelty_gain: float = 0.65
    homeostatic_target: float = 0.50
    homeostatic_gain: float = 0.025
    class_gain_min: float = 0.50
    class_gain_max: float = 2.00
    consolidation_weight: float = 0.60
    max_experiences: int = 512
    use_n4_attention: bool = True
    use_replay: bool = True
    use_homeostasis: bool = True
    use_slow_consolidation: bool = True

    def __post_init__(self) -> None:
        positive = (
            "epsilon", "fast_learning_rate", "slow_learning_rate",
            "max_step_norm", "max_prototype_drift",
            "prior_pseudocount_mass", "transition_learning_rate",
        )
        for name in positive:
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_prototype_drift > 2.0:
            raise ValueError("max_prototype_drift cannot exceed the maximum unit-vector distance")

        nonnegative = ("slow_consolidation_rate", "homeostatic_gain")
        for name in nonnegative:
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

        unit_interval = (
            "fast_trace_decay", "slow_trace_decay", "replay_decay",
            "novelty_floor", "novelty_gain", "homeostatic_target",
            "consolidation_weight",
        )
        for name in unit_interval:
            value = getattr(self, name)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")

        if not (
            isfinite(self.class_gain_min)
            and isfinite(self.class_gain_max)
            and 0.0 < self.class_gain_min <= 1.0 <= self.class_gain_max
        ):
            raise ValueError("class gain bounds must be finite and contain 1.0")

        for name in ("replay_capacity", "replay_max_samples", "max_experiences"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.replay_max_samples > self.replay_capacity:
            raise ValueError("replay_max_samples cannot exceed replay_capacity")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
