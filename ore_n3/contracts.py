"""Typed contracts for N3 inputs and outputs."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

UNKNOWN_LABEL = "<UNK>"

@dataclass(slots=True)
class LabeledN2Patch:
    """A labelled, already-segmented patch represented by an N2 result or vector."""
    grapheme: str
    n2_result: Any | None = None
    features: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.grapheme, str) or not self.grapheme.strip():
            raise ValueError("grapheme labels must be non-empty strings")
        if self.grapheme == UNKNOWN_LABEL:
            raise ValueError(f"{UNKNOWN_LABEL} is reserved for uncertainty handling")
        if self.n2_result is None and self.features is None:
            raise ValueError("provide either n2_result or features")

@dataclass(slots=True)
class N3Patch:
    """One position in an already-segmented sequence."""
    patch_id: str
    n2_result: Any | None = None
    features: np.ndarray | None = None
    segmentation_confidence: float = 1.0
    attention: float = 0.0

    def __post_init__(self) -> None:
        if not self.patch_id:
            raise ValueError("patch_id must be non-empty")
        if self.n2_result is None and self.features is None:
            raise ValueError("provide either n2_result or features")

@dataclass(slots=True)
class N3Sequence:
    """Ordered patch observations; input order is logical reading order."""
    sequence_id: str
    patches: tuple[N3Patch, ...] | list[N3Patch]

    def __post_init__(self) -> None:
        self.patches = tuple(self.patches)
        if not self.sequence_id:
            raise ValueError("sequence_id must be non-empty")
        if not self.patches:
            raise ValueError("N3Sequence must contain at least one patch")
        if len({patch.patch_id for patch in self.patches}) != len(self.patches):
            raise ValueError("patch_id values must be unique within a sequence")

@dataclass(slots=True)
class N3Result:
    """Full N3 output with intermediate beliefs and error signals retained."""
    sequence_id: str
    class_names: tuple[str, ...]
    visual_probabilities: np.ndarray
    initial_beliefs: np.ndarray
    contextual_priors: np.ndarray
    posterior_beliefs: np.ndarray
    classwise_prediction_error: np.ndarray
    feature_residuals: np.ndarray
    trace: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation without discarding uncertainty."""
        return {
            "schema": "ore.n3.recurrent_orthographic_prediction.v1",
            "sequence_id": self.sequence_id,
            "class_names": list(self.class_names),
            "visual_probabilities": self.visual_probabilities.tolist(),
            "initial_beliefs": self.initial_beliefs.tolist(),
            "contextual_priors": self.contextual_priors.tolist(),
            "posterior_beliefs": self.posterior_beliefs.tolist(),
            "classwise_prediction_error": self.classwise_prediction_error.tolist(),
            "feature_residuals": self.feature_residuals.tolist(),
            "trace": self.trace,
            "candidates": self.candidates,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        }
