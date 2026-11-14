"""Validated numerical controls for ORE N8 lexical-semantic access."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class N8Config:
    """Experimental controls; these are not measured neural constants."""

    epsilon: float = 1e-8
    posterior_temperature: float = 0.85
    frequency_weight: float = 0.15
    domain_weight: float = 0.90
    intent_weight: float = 0.55
    explicit_sense_weight: float = 1.10
    feature_weight: float = 0.65
    association_weight: float = 0.25
    unknown_log_prior: float = -2.4
    max_lexicon_entries: int = 100_000
    max_tokens: int = 2_048
    max_relations: int = 8_192
    max_candidates_per_token: int = 64
    abstain_threshold: float = 0.42
    ambiguity_margin: float = 0.09
    minimum_relation_confidence: float = 0.20
    require_provenance: bool = True
    casefold_orthography: bool = True

    def __post_init__(self) -> None:
        positive = ("epsilon", "posterior_temperature")
        for name in positive:
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        nonnegative = (
            "frequency_weight", "domain_weight", "intent_weight",
            "explicit_sense_weight", "feature_weight", "association_weight",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

        if not isfinite(self.unknown_log_prior):
            raise ValueError("unknown_log_prior must be finite")
        for name in ("max_lexicon_entries", "max_tokens", "max_relations",
                     "max_candidates_per_token"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 <= self.abstain_threshold <= 1.0:
            raise ValueError("abstain_threshold must lie in [0, 1]")
        if not 0.0 <= self.ambiguity_margin <= 1.0:
            raise ValueError("ambiguity_margin must lie in [0, 1]")
        if not 0.0 <= self.minimum_relation_confidence <= 1.0:
            raise ValueError("minimum_relation_confidence must lie in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
