"""Validated numerical controls for N7."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from math import isfinite


@dataclass(frozen=True, slots=True)
class N7Config:
    epsilon: float = 1e-8
    global_precision: float = 1.0
    min_precision: float = 0.05
    max_precision: float = 20.0
    max_error: float = 10.0
    context_prior_strength: float = 0.85
    generative_blend: float = 0.85
    kl_weight: float = 0.25
    belief_temperature: float = 1.0
    learning_rate: float = 0.15
    max_iterations: int = 40
    convergence_tolerance: float = 1e-7
    ambiguity_margin: float = 0.06
    memory_decay: float = 0.90
    lexical_context_scale: int = 8
    discourse_context_scale: int = 8
    task_context_scale: int = 4
    intent_labels: tuple[str, ...] = ("recognize", "classify", "compare", "learn", "unknown")
    domain_labels: tuple[str, ...] = ("orthography", "language", "visual", "general")
    style_labels: tuple[str, ...] = ("concise", "detailed", "procedural", "exploratory")
    cue_prototypes: dict[str, tuple[float, ...]] = field(default_factory=lambda: {
        "visual_strength": (0.85,0.55,0.90,0.25,0.10,0.35,0.20,0.25,0.15,0.10,0.20,0.20,0.15,0.10,0.10,0.10,0.15,0.10,0.05,0.10),
        "orthographic_strength": (0.90,0.65,0.30,0.20,0.10,0.35,0.20,0.25,0.10,0.15,0.15,0.10,0.10,0.15,0.10,0.15,0.10,0.15,0.05,0.10),
        "context_strength": (0.55,0.85,0.25,0.20,0.10,0.25,0.15,0.35,0.15,0.15,0.20,0.15,0.10,0.20,0.15,0.20,0.10,0.20,0.10,0.15),
        "task_strength": (0.60,0.45,0.20,0.75,0.20,0.15,0.10,0.15,0.10,0.20,0.10,0.15,0.10,0.10,0.10,0.15,0.15,0.10,0.10,0.10),
    })
    observation_variance: dict[str, float] = field(default_factory=lambda: {
        "visual_strength": 0.20, "orthographic_strength": 0.25,
        "context_strength": 0.40, "task_strength": 0.35,
    })
    layer_by_cue: dict[str, str] = field(default_factory=lambda: {
        "visual_strength":"visual", "orthographic_strength":"orthographic",
        "context_strength":"context", "task_strength":"task",
    })

    def __post_init__(self) -> None:
        for name in ("epsilon","global_precision","belief_temperature","convergence_tolerance"):
            value = getattr(self,name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("min_precision","max_precision","max_error","learning_rate"):
            value = getattr(self,name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("context_prior_strength","generative_blend","memory_decay","ambiguity_margin"):
            value = getattr(self,name)
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0,1]")
        if self.min_precision > self.max_precision:
            raise ValueError("min_precision cannot exceed max_precision")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not self.intent_labels or not self.domain_labels or not self.style_labels:
            raise ValueError("intent, domain, and style vocabularies must be non-empty")
        keys = set(self.cue_prototypes)
        if keys != set(self.observation_variance) or keys != set(self.layer_by_cue):
            raise ValueError("cue prototype, variance, and layer keys must match")
        width = len(self.intent_labels) * len(self.domain_labels)
        for key, values in self.cue_prototypes.items():
            if len(values) != width or any(not isfinite(float(v)) or not 0 <= float(v) <= 1 for v in values):
                raise ValueError(f"cue prototype {key} must have {width} probabilities in [0,1]")
            variance = self.observation_variance[key]
            if not isfinite(variance) or variance <= 0:
                raise ValueError(f"observation variance for {key} must be positive")
    def to_dict(self):
        return asdict(self)
