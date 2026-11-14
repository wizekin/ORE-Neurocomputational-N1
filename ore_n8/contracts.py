"""Public input, semantic-resource, relation and output contracts for ORE N8."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(slots=True)
class N8SemanticEntry:
    """One externally verified orthography-to-concept/sense mapping.

    ``domain_weights`` and ``intent_weights`` express caller-supplied compatibility
    in [0, 1]. ``semantic_features`` is a sparse, non-negative feature vector.
    Provenance fields are mandatory when the default strict policy is enabled.
    """

    entry_id: str
    orthography: str
    concept_id: str
    sense_id: str
    gloss: str
    verified_by: str
    source_reference: str
    frequency: float = 1.0
    language: str = "und"
    concept_type: str = "unspecified"
    domain_weights: Mapping[str, float] = field(default_factory=dict)
    intent_weights: Mapping[str, float] = field(default_factory=dict)
    semantic_features: Mapping[str, float] = field(default_factory=dict)
    association_tags: tuple[str, ...] = ()


@dataclass(slots=True)
class N8RelationEvidence:
    """Explicit externally supplied relation between two N6 token positions.

    N8 does not infer syntactic/dependency relations from word order. An edge is
    included only when the caller supplies this evidence with provenance.
    """

    source_token_index: int
    target_token_index: int
    relation: str
    confidence: float
    verified_by: str
    source_reference: str


@dataclass(slots=True)
class N8Input:
    """N6 token sequence + N7 context posterior + verified semantic resources."""

    n7_result: Any
    n6_result: Any
    semantic_lexicon: Iterable[N8SemanticEntry] = field(default_factory=tuple)
    relation_evidence: Iterable[N8RelationEvidence] = field(default_factory=tuple)
    semantic_context: Mapping[str, Any] = field(default_factory=dict)
    processing_mode: str = "inference"


@dataclass(slots=True)
class N8Result:
    """Inspectable lexical-semantic candidates, graph, frames and diagnostics."""

    sequence_id: str
    token_semantics: list[dict[str, Any]]
    intent_posterior: dict[str, float]
    domain_posterior: dict[str, float]
    concept_activations: dict[str, float]
    concept_ranking: list[dict[str, Any]]
    semantic_graph: dict[str, Any]
    compositional_frames: list[dict[str, Any]]
    preserved_evidence: dict[str, Any]
    operation_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation without dropping alternatives."""
        return asdict(self)
