"""Public data contracts for ORE N6 hierarchical orthographic decoding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass(slots=True)
class N6LexiconEntry:
    """One externally supplied, provenance-checked orthographic word form."""

    entry_id: str
    graphemes: tuple[str, ...]
    frequency: float
    verified_by: str
    source_reference: str
    language: str = "und"


@dataclass(slots=True)
class N6Input:
    """An N4 sequence, its immutable N5 model snapshot, and an optional lexicon."""

    n4_result: Any
    n5_model: Any
    lexicon_entries: Iterable[N6LexiconEntry] = field(default_factory=tuple)
    boundary_probabilities: np.ndarray | None = None
    n2_features: np.ndarray | None = None


@dataclass(slots=True)
class N6Result:
    """Decoded graphemes, lexical segmentation, alternatives, and full provenance."""

    sequence_id: str
    class_names: tuple[str, ...]
    input_probabilities: np.ndarray
    contextual_probabilities: np.ndarray
    decoded_probabilities: np.ndarray
    decoded_graphemes: tuple[str, ...]
    boundary_probabilities: np.ndarray
    tokens: list[dict[str, Any]] = field(default_factory=list)
    position_candidates: list[dict[str, Any]] = field(default_factory=list)
    segmentation_alternatives: list[dict[str, Any]] = field(default_factory=list)
    operation_trace: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible output while preserving intermediate signals."""
        return {
            "schema": "ore.n6.hierarchical_orthographic_access.v1",
            "sequence_id": self.sequence_id,
            "class_names": list(self.class_names),
            "input_probabilities": self.input_probabilities.tolist(),
            "contextual_probabilities": self.contextual_probabilities.tolist(),
            "decoded_probabilities": self.decoded_probabilities.tolist(),
            "decoded_graphemes": list(self.decoded_graphemes),
            "boundary_probabilities": self.boundary_probabilities.tolist(),
            "tokens": self.tokens,
            "position_candidates": self.position_candidates,
            "segmentation_alternatives": self.segmentation_alternatives,
            "operation_trace": self.operation_trace,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        }
