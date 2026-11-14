"""Public data contracts for ORE N4."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class N4Input:
    """N3 output plus optional measured or task-derived attentional cues.

    Cue arrays are one value per patch and must lie in [0, 1].
    ``goal_class_weights`` contains one non-negative value per N3 class.
    Missing cues are not fabricated: salience defaults to internally-derived
    uncertainty/error, task relevance defaults to zero, and segmentation/seed
    attention are read from N3 candidate metadata when present.
    """

    n3_result: Any
    bottom_up_salience: np.ndarray | None = None
    task_relevance: np.ndarray | None = None
    goal_class_weights: np.ndarray | None = None
    seed_attention: np.ndarray | None = None
    segmentation_confidence: np.ndarray | None = None


@dataclass(slots=True)
class N4Result:
    """Attention-gated hypotheses, precision values and inspectable traces."""

    sequence_id: str
    class_names: tuple[str, ...]
    visual_probabilities: np.ndarray
    contextual_priors: np.ndarray
    n3_posterior_beliefs: np.ndarray
    posterior_beliefs: np.ndarray
    bottom_up_salience: np.ndarray
    top_down_relevance: np.ndarray
    attention_allocation: np.ndarray
    class_attention: np.ndarray
    class_attention_gate: np.ndarray
    sensory_precision: np.ndarray
    prior_precision: np.ndarray
    classwise_prediction_error: np.ndarray
    iteration_trace: list[dict[str, Any]] = field(default_factory=list)
    operation_trace: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible values without dropping intermediate signals."""
        return {
            "schema": "ore.n4.attention_precision_gating.v1",
            "sequence_id": self.sequence_id,
            "class_names": list(self.class_names),
            "visual_probabilities": self.visual_probabilities.tolist(),
            "contextual_priors": self.contextual_priors.tolist(),
            "n3_posterior_beliefs": self.n3_posterior_beliefs.tolist(),
            "posterior_beliefs": self.posterior_beliefs.tolist(),
            "bottom_up_salience": self.bottom_up_salience.tolist(),
            "top_down_relevance": self.top_down_relevance.tolist(),
            "attention_allocation": self.attention_allocation.tolist(),
            "class_attention": self.class_attention.tolist(),
            "class_attention_gate": self.class_attention_gate.tolist(),
            "sensory_precision": self.sensory_precision.tolist(),
            "prior_precision": self.prior_precision.tolist(),
            "classwise_prediction_error": self.classwise_prediction_error.tolist(),
            "iteration_trace": self.iteration_trace,
            "operation_trace": self.operation_trace,
            "candidates": self.candidates,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        }
