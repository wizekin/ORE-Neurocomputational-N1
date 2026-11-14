"""Public N7 predictive/contextual processing contracts."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class N7Input:
    """N6 evidence plus separately supplied contextual/provider signals."""
    n6_result: Any
    context_state: Mapping[str, Any] = field(default_factory=dict)
    script_profile: str | None = None
    lexical_context: Mapping[str, Any] = field(default_factory=dict)
    syntax_context: Mapping[str, Any] = field(default_factory=dict)
    semantic_context: Mapping[str, Any] = field(default_factory=dict)
    discourse_context: Mapping[str, Any] = field(default_factory=dict)
    task_context: Mapping[str, Any] = field(default_factory=dict)
    provider_evidence: Mapping[str, float] = field(default_factory=dict)
    processing_mode: str = "inference"


@dataclass(slots=True)
class N7Result:
    """Preserved lower-level evidence and higher-level predictive beliefs."""
    sequence_id: str
    evidence: dict[str, Any]
    predictions: list[float]
    precision: list[float]
    prediction_errors: list[float]
    layer_errors: dict[str, float]
    posterior: list[float]
    style_posterior: dict[str, float]
    ranked_hypotheses: list[dict[str, Any]]
    memory_trace: dict[str, Any]
    association_ranking: list[tuple[str, float]]
    operation_trace: list[dict[str, Any]]
    convergence: list[dict[str, float]]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)
