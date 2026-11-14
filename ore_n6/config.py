"""Validated controls for ORE N6 hierarchical orthographic decoding."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class N6Config:
    """Numerical controls are engineering hypotheses, not measured neural constants."""

    epsilon: float = 1e-8
    transition_smoothing: float = 0.5
    unigram_smoothing: float = 0.5
    class_gain_strength: float = 0.30
    prototype_evidence_weight: float = 0.20
    prototype_temperature: float = 0.18
    max_lexicon_first_graphemes: int = 16
    max_lexicon_candidates_per_position: int = 2000
    forward_context_weight: float = 0.35
    reverse_context_weight: float = 0.35
    context_temperature: float = 1.0
    lexical_frequency_weight: float = 0.12
    lexical_transition_weight: float = 0.25
    boundary_weight: float = 0.30
    default_boundary_probability: float = 0.50
    lexical_confidence_temperature: float = 1.0
    unknown_segment_penalty: float = 2.0
    abstain_threshold: float = 0.35
    ambiguity_margin: float = 0.06
    top_k_position_candidates: int = 5
    top_k_segmentations: int = 5
    max_sequence_length: int = 2048
    max_lexicon_entries: int = 100_000
    max_lexicon_graphemes: int = 64
    strict_upstream_validity: bool = False
    use_class_gains: bool = True
    use_bidirectional_context: bool = True
    use_boundary_evidence: bool = True

    def __post_init__(self) -> None:
        positive = (
            "epsilon", "transition_smoothing", "unigram_smoothing",
            "context_temperature", "lexical_confidence_temperature", "prototype_temperature",
        )
        for name in positive:
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        nonnegative = (
            "class_gain_strength", "forward_context_weight", "reverse_context_weight",
            "lexical_frequency_weight", "lexical_transition_weight", "boundary_weight",
            "unknown_segment_penalty",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

        if not isfinite(self.prototype_evidence_weight) or not 0.0 <= self.prototype_evidence_weight < 1.0:
            raise ValueError("prototype_evidence_weight must lie in [0, 1)")

        for name in ("default_boundary_probability", "abstain_threshold", "ambiguity_margin"):
            value = getattr(self, name)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")

        for name in (
            "top_k_position_candidates", "top_k_segmentations", "max_sequence_length",
            "max_lexicon_entries", "max_lexicon_graphemes",
            "max_lexicon_first_graphemes", "max_lexicon_candidates_per_position",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
