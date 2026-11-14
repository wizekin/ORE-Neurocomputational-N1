"""Typed public contracts for the integrated ORE N9 controller.

N9 is an additive coordination layer. Upstream inputs are treated as immutable,
and absent N5-N8 implementations are represented explicitly rather than guessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(slots=True)
class N9Patch:
    """One already-localized text patch in logical reading order.

    Exactly one evidence source is required: a grayscale image, an N2 result,
    a feature vector, or a caller-supplied grapheme probability mapping.
    N9 does not silently segment a full line into characters.
    """

    patch_id: str
    image: np.ndarray | None = None
    n2_result: Any | None = None
    features: np.ndarray | None = None
    grapheme_probabilities: Mapping[str, float] | None = None
    segmentation_confidence: float = 1.0
    x: float | None = None
    y: float | None = None
    observed: bool = True
    valid_mask: np.ndarray | None = None
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.patch_id, str) or not self.patch_id.strip():
            raise ValueError("patch_id must be a non-empty string")
        supplied = sum(
            value is not None
            for value in (self.image, self.n2_result, self.features, self.grapheme_probabilities)
        )
        if supplied != 1:
            raise ValueError(
                "provide exactly one of image, n2_result, features, or grapheme_probabilities"
            )
        if not np.isfinite(self.segmentation_confidence) or not (
            0.0 <= self.segmentation_confidence <= 1.0
        ):
            raise ValueError("segmentation_confidence must lie in [0, 1]")
        for name, value in (("x", self.x), ("y", self.y)):
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} coordinate must be finite")
        if self.image is not None:
            image = np.asarray(self.image)
            if image.ndim != 2 or min(image.shape) < 4:
                raise ValueError("patch image must be a 2-D grayscale array at least 4x4")
            if not np.isfinite(image).all():
                raise ValueError("patch image must contain only finite values")
            if np.issubdtype(image.dtype, np.floating) and (
                np.any(image < 0.0) or np.any(image > 1.0)
            ):
                raise ValueError("floating grayscale images must lie in [0, 1]")
            if self.valid_mask is not None and np.asarray(self.valid_mask).shape != image.shape:
                raise ValueError("valid_mask must have the same shape as image")
        if self.features is not None:
            features = np.asarray(self.features)
            if features.ndim != 1 or features.size == 0 or not np.isfinite(features).all():
                raise ValueError("features must be a non-empty finite one-dimensional vector")
        if self.grapheme_probabilities is not None:
            if not self.grapheme_probabilities:
                raise ValueError("grapheme_probabilities cannot be empty")
            if any(not isinstance(k, str) or not k for k in self.grapheme_probabilities):
                raise ValueError("grapheme probability keys must be non-empty strings")
            values = np.asarray(list(self.grapheme_probabilities.values()), dtype=np.float64)
            if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
                raise ValueError("grapheme probabilities must be finite, non-negative and nonzero")


@dataclass(slots=True)
class N9Request:
    """A single integrated recognition/comprehension request."""

    sequence_id: str
    patches: tuple[N9Patch, ...] | list[N9Patch]
    task: str = "recognize"
    script_profile: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    word_spans: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None
    source_id: str | None = None
    learning_authorized: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id must be a non-empty string")
        self.patches = tuple(self.patches)
        if not self.patches:
            raise ValueError("N9 requires at least one localized patch")
        if any(not isinstance(patch, N9Patch) for patch in self.patches):
            raise TypeError("patches must contain N9Patch objects")
        if len({patch.patch_id for patch in self.patches}) != len(self.patches):
            raise ValueError("patch_id values must be unique")
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must be a non-empty string")
        if self.word_spans is not None:
            self.word_spans = tuple((int(start), int(end)) for start, end in self.word_spans)
            previous_end = 0
            for start, end in self.word_spans:
                if start < previous_end or start < 0 or end <= start or end > len(self.patches):
                    raise ValueError("word_spans must be ordered non-overlapping [start, end) ranges")
                previous_end = end


@dataclass(slots=True)
class N9Adapters:
    """Optional upstream processors and extension adapters.

    N1-N4 use their repository APIs. N5-N8 and the syntax adapter are injected
    through small callable/object contracts, allowing later phases to be added
    without fabricating implementations or changing the N9 controller.
    """

    n1: Any | None = None
    n2: Any | None = None
    n3: Any | None = None
    n4: Any | None = None
    n5: Any | None = None
    n6: Any | None = None
    n7: Any | None = None
    n8: Any | None = None
    syntax: Any | None = None


@dataclass(frozen=True, slots=True)
class N9Config:
    """Numerical and decision controls for bounded N9 inference."""

    max_feedback_iterations: int = 8
    damping: float = 0.25
    feedback_gain: float = 0.20
    context_gain: float = 0.15
    confidence_threshold: float = 0.70
    ambiguity_margin: float = 0.12
    top_k: int = 5
    beam_width: int = 32
    epsilon: float = 1e-8
    strict_integrity: bool = True

    def __post_init__(self) -> None:
        positive = ("epsilon",)
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("damping",):
            value = getattr(self, name)
            if not np.isfinite(value) or not (0.0 < value <= 1.0):
                raise ValueError(f"{name} must lie in (0, 1]")
        for name in ("feedback_gain", "context_gain", "confidence_threshold", "ambiguity_margin"):
            value = getattr(self, name)
            if not np.isfinite(value) or not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.max_feedback_iterations < 1 or self.top_k < 1 or self.beam_width < 1:
            raise ValueError("iteration and candidate limits must be positive")


@dataclass(slots=True)
class N9Result:
    """Inspectable output from all 120 N9 operations."""

    sequence_id: str
    class_names: tuple[str, ...]
    visual_probabilities: np.ndarray
    posterior_beliefs: np.ndarray
    attention_allocation: np.ndarray
    lexical_hypotheses: list[dict[str, Any]]
    syntax: dict[str, Any]
    semantic_graph: dict[str, Any]
    propositions: list[dict[str, Any]]
    context_model: dict[str, Any]
    decision: dict[str, Any]
    operation_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    provenance: dict[str, Any]
    learning: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible output while retaining evidence-channel separation."""
        return {
            "schema": "ore.n9.integrated_neurocomputational_ore.v1",
            "sequence_id": self.sequence_id,
            "class_names": list(self.class_names),
            "visual_probabilities": self.visual_probabilities.tolist(),
            "posterior_beliefs": self.posterior_beliefs.tolist(),
            "attention_allocation": self.attention_allocation.tolist(),
            "lexical_hypotheses": self.lexical_hypotheses,
            "syntax": self.syntax,
            "semantic_graph": self.semantic_graph,
            "propositions": self.propositions,
            "context_model": self.context_model,
            "decision": self.decision,
            "operation_trace": self.operation_trace,
            "diagnostics": self.diagnostics,
            "provenance": self.provenance,
            "learning": self.learning,
            "metadata": self.metadata,
        }


N9Input = N9Request
