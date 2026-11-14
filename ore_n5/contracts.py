"""Data contracts for ORE N5 verified experience-dependent plasticity."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

UNKNOWN_LABEL = "<UNK>"


@dataclass(slots=True)
class N5Experience:
    """One independently verified labelled visual-patch experience."""

    experience_id: str
    patch_id: str
    sequence_id: str
    grapheme: str
    features: np.ndarray
    verified_by: str
    source_reference: str
    verified_at: str
    confidence: float = 1.0
    sequence_position: int | None = None
    feature_sha256: str | None = None


@dataclass(slots=True)
class N5LearningRequest:
    """An explicitly authorized batch of verified learning experiences."""

    parent_model: "N5ModelState"
    experiences: Iterable[N5Experience]
    independently_verified: bool = False
    user_authorized: bool = False
    n4_result: Any | None = None
    verified_sequences: Iterable[Iterable[str]] = ()


@dataclass(slots=True)
class N5ModelState:
    """Independent, versioned child-model state owned by N5."""

    class_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    fast_prototypes: np.ndarray
    slow_prototypes: np.ndarray
    fast_eligibility: np.ndarray
    slow_eligibility: np.ndarray
    exposure_counts: np.ndarray
    class_gains: np.ndarray
    fast_unigram_counts: np.ndarray
    slow_unigram_counts: np.ndarray
    fast_forward_counts: np.ndarray
    slow_forward_counts: np.ndarray
    fast_reverse_counts: np.ndarray
    slow_reverse_counts: np.ndarray
    replay_buffer: list[dict[str, Any]] = field(default_factory=list)
    version: int = 1
    parent_fingerprint: str | None = None
    fingerprint: str = ""
    visual_temperature: float = 0.18
    unknown_similarity_threshold: float = 0.18
    unknown_temperature: float = 0.08
    transition_alpha: float = 0.5
    schema_version: str = "1.0"

    @property
    def feature_dimension(self) -> int:
        return int(np.asarray(self.feature_mean).size)

    @property
    def output_class_names(self) -> tuple[str, ...]:
        return (*self.class_names, UNKNOWN_LABEL)

    @classmethod
    def from_n3_processor(
        cls, processor: Any, *, prior_pseudocount_mass: float = 32.0
    ) -> "N5ModelState":
        """Create a detached N5 parent snapshot from a fitted N3Processor."""
        if not bool(getattr(processor, "_fitted", False)):
            raise ValueError("N3Processor must be fitted before N5 initialization")
        population = getattr(processor, "visual_population", None)
        transitions = getattr(processor, "transition_model", None)
        required = ("class_names", "mean", "scale", "prototypes")
        if population is None or transitions is None or any(
            getattr(population, key, None) is None for key in required
        ):
            raise ValueError("fitted N3Processor does not expose a complete visual population")

        names = tuple(population.class_names)
        if not names or UNKNOWN_LABEL in names or len(set(names)) != len(names):
            raise ValueError("N3 visual vocabulary is invalid")
        if tuple(transitions.class_names) != (*names, UNKNOWN_LABEL):
            raise ValueError("N3 transition vocabulary is incompatible with its visual vocabulary")

        mass = float(prior_pseudocount_mass)
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("prior_pseudocount_mass must be finite and positive")
        unigram = np.asarray(transitions.unigram, dtype=np.float64) * mass
        forward = np.asarray(transitions.forward, dtype=np.float64) * mass
        reverse = np.asarray(transitions.reverse, dtype=np.float64) * mass

        model = cls(
            class_names=names,
            feature_mean=np.asarray(population.mean, dtype=np.float32).copy(),
            feature_scale=np.asarray(population.scale, dtype=np.float32).copy(),
            fast_prototypes=np.asarray(population.prototypes, dtype=np.float32).copy(),
            slow_prototypes=np.asarray(population.prototypes, dtype=np.float32).copy(),
            fast_eligibility=np.zeros_like(population.prototypes, dtype=np.float32),
            slow_eligibility=np.zeros_like(population.prototypes, dtype=np.float32),
            exposure_counts=np.zeros(len(names), dtype=np.float64),
            class_gains=np.ones(len(names), dtype=np.float64),
            fast_unigram_counts=unigram.copy(),
            slow_unigram_counts=unigram.copy(),
            fast_forward_counts=forward.copy(),
            slow_forward_counts=forward.copy(),
            fast_reverse_counts=reverse.copy(),
            slow_reverse_counts=reverse.copy(),
            version=1,
            visual_temperature=float(population.temperature),
            unknown_similarity_threshold=float(population.unknown_threshold),
            unknown_temperature=float(population.unknown_temperature),
            transition_alpha=float(processor.config.transition_alpha),
        )
        from .stages import fingerprint_model
        model.fingerprint = fingerprint_model(model)
        return model

    def clone(self) -> "N5ModelState":
        """Return a deep copy with detached arrays and replay records."""
        return N5ModelState.from_dict(self.to_dict())

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": "ore.n5.plasticity_model.v1",
            "schema_version": self.schema_version,
            "class_names": list(self.class_names),
            "feature_mean": np.asarray(self.feature_mean).tolist(),
            "feature_scale": np.asarray(self.feature_scale).tolist(),
            "fast_prototypes": np.asarray(self.fast_prototypes).tolist(),
            "slow_prototypes": np.asarray(self.slow_prototypes).tolist(),
            "fast_eligibility": np.asarray(self.fast_eligibility).tolist(),
            "slow_eligibility": np.asarray(self.slow_eligibility).tolist(),
            "exposure_counts": np.asarray(self.exposure_counts).tolist(),
            "class_gains": np.asarray(self.class_gains).tolist(),
            "fast_unigram_counts": np.asarray(self.fast_unigram_counts).tolist(),
            "slow_unigram_counts": np.asarray(self.slow_unigram_counts).tolist(),
            "fast_forward_counts": np.asarray(self.fast_forward_counts).tolist(),
            "slow_forward_counts": np.asarray(self.slow_forward_counts).tolist(),
            "fast_reverse_counts": np.asarray(self.fast_reverse_counts).tolist(),
            "slow_reverse_counts": np.asarray(self.slow_reverse_counts).tolist(),
            "replay_buffer": deepcopy(self.replay_buffer),
            "version": int(self.version),
            "parent_fingerprint": self.parent_fingerprint,
            "visual_temperature": float(self.visual_temperature),
            "unknown_similarity_threshold": float(self.unknown_similarity_threshold),
            "unknown_temperature": float(self.unknown_temperature),
            "transition_alpha": float(self.transition_alpha),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "N5ModelState":
        if not isinstance(payload, dict) or payload.get("schema") != "ore.n5.plasticity_model.v1":
            raise ValueError("unsupported N5 model schema")
        float64_fields = {
            "exposure_counts", "class_gains", "fast_unigram_counts",
            "slow_unigram_counts", "fast_forward_counts", "slow_forward_counts",
            "fast_reverse_counts", "slow_reverse_counts",
        }
        array_fields = (
            "feature_mean", "feature_scale", "fast_prototypes", "slow_prototypes",
            "fast_eligibility", "slow_eligibility", "exposure_counts", "class_gains",
            "fast_unigram_counts", "slow_unigram_counts", "fast_forward_counts",
            "slow_forward_counts", "fast_reverse_counts", "slow_reverse_counts",
        )
        missing = [key for key in array_fields if key not in payload]
        if missing:
            raise ValueError(f"N5 model is missing arrays: {missing}")

        model = cls(
            class_names=tuple(payload.get("class_names", ())),
            **{
                key: np.asarray(
                    payload[key], dtype=np.float64 if key in float64_fields else np.float32
                )
                for key in array_fields
            },
            replay_buffer=deepcopy(list(payload.get("replay_buffer", []))),
            version=int(payload.get("version", 1)),
            parent_fingerprint=payload.get("parent_fingerprint"),
            fingerprint=str(payload.get("fingerprint", "")),
            visual_temperature=float(payload.get("visual_temperature", 0.18)),
            unknown_similarity_threshold=float(payload.get("unknown_similarity_threshold", 0.18)),
            unknown_temperature=float(payload.get("unknown_temperature", 0.08)),
            transition_alpha=float(payload.get("transition_alpha", 0.5)),
            schema_version=str(payload.get("schema_version", "1.0")),
        )
        from .stages import fingerprint_model
        if not model.fingerprint or fingerprint_model(model) != model.fingerprint:
            raise ValueError("N5 model fingerprint mismatch")
        return model

    def to_n3_processor(self, config: Any | None = None) -> Any:
        """Export a detached effective child model for subsequent N3 inference."""
        from ore_n3.config import N3Config
        from ore_n3.contracts import UNKNOWN_LABEL as N3_UNKNOWN
        from ore_n3.pipeline import N3Processor
        from ore_n3.transitions import OrthographicTransitionModel
        from .stages import normalize_rows

        if N3_UNKNOWN != UNKNOWN_LABEL:
            raise RuntimeError("N3/N5 unknown-label contract mismatch")
        from .stages import fingerprint_model
        if fingerprint_model(self) != self.fingerprint:
            raise ValueError("cannot export an N5 model with an invalid fingerprint")

        processor = N3Processor(config or N3Config())
        population = processor.visual_population
        population.class_names = tuple(self.class_names)
        population.mean = np.asarray(self.feature_mean, dtype=np.float32).copy()
        population.scale = np.asarray(self.feature_scale, dtype=np.float32).copy()
        blended = normalize_rows(
            0.75 * np.asarray(self.fast_prototypes) + 0.25 * np.asarray(self.slow_prototypes),
            1e-8,
        )
        population.prototypes = (
            blended * np.asarray(self.class_gains, dtype=np.float32)[:, None]
        ).astype(np.float32)
        population.temperature = float(self.visual_temperature)
        population.unknown_threshold = float(self.unknown_similarity_threshold)
        population.unknown_temperature = float(self.unknown_temperature)

        import hashlib
        import json
        population_payload = {
            "classes": list(population.class_names),
            "mean": population.mean.tolist(),
            "scale": population.scale.tolist(),
            "prototypes": population.prototypes.tolist(),
        }
        population.fingerprint = hashlib.sha256(json.dumps(
            population_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()

        unigram_counts = (
            np.asarray(self.fast_unigram_counts) + np.asarray(self.slow_unigram_counts)
        )
        unigram = unigram_counts / max(float(unigram_counts.sum()), 1e-8)
        forward_counts = (
            np.asarray(self.fast_forward_counts) + np.asarray(self.slow_forward_counts)
            + self.transition_alpha
        )
        reverse_counts = (
            np.asarray(self.fast_reverse_counts) + np.asarray(self.slow_reverse_counts)
            + self.transition_alpha
        )
        forward = forward_counts / np.maximum(forward_counts.sum(axis=1, keepdims=True), 1e-8)
        reverse = reverse_counts / np.maximum(reverse_counts.sum(axis=0, keepdims=True), 1e-8)
        transition_payload = {
            "classes": list(self.output_class_names),
            "unigram": unigram.tolist(),
            "forward": forward.tolist(),
            "reverse": reverse.tolist(),
            "alpha": self.transition_alpha,
        }
        transition_fingerprint = hashlib.sha256(json.dumps(
            transition_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()
        processor.transition_model = OrthographicTransitionModel(
            tuple(self.output_class_names), unigram.astype(np.float32),
            forward.astype(np.float32), reverse.astype(np.float32),
            transition_fingerprint,
        )
        processor._fitted = True
        processor.training_trace = [{
            "step": "N5_EXPORT",
            "source_n5_fingerprint": self.fingerprint,
            "source_n5_version": self.version,
            "read_only_export": True,
        }]
        return processor


@dataclass(slots=True)
class N5Result:
    """New child model plus its complete operation-level learning trace."""

    sequence_ids: tuple[str, ...]
    parent_fingerprint: str
    child_model: N5ModelState
    operation_trace: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ore.n5.experience_dependent_plasticity.v1",
            "sequence_ids": list(self.sequence_ids),
            "parent_fingerprint": self.parent_fingerprint,
            "child_model": self.child_model.to_dict(),
            "operation_trace": self.operation_trace,
            "diagnostics": self.diagnostics,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }
