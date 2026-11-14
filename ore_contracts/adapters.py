"""Explicit adapters from legacy ORE result objects into canonical contracts.

Legacy package names are preserved as provenance. These adapters do not
relabel old N1/N5/N6/N9 work as the new canonical B-N phase ownership.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import ContractEnvelope, EvidenceKind, sha256_json
from .advanced import (
    FeatureTensor, VisualFeatureBundle, GraphemeEvidence,
    PositionCodeBundle, PositionCodeChannel, OrthographicMemorySnapshot,
    PredictionState, SequenceCandidate, SequenceHypothesisSet,
    ORERecognitionResult,
)


def make_envelope(
    contract_id: str,
    run_id: str,
    producer_id: str,
    *,
    config: Mapping[str, Any] | None = None,
    evidence_kind: EvidenceKind = EvidenceKind.DERIVED,
    sample_id: str | None = None,
    data_hash: str | None = None,
    script_profile_id: str | None = None,
    parent_artifact_ids: Sequence[str] = (),
    model_version: str | None = None,
) -> ContractEnvelope:
    """Create a provenance-complete envelope using a canonical config digest."""
    return ContractEnvelope(
        contract_id=contract_id,
        schema_version="1.0.0",
        run_id=run_id,
        producer_id=producer_id,
        config_hash=sha256_json(dict(config or {})),
        created_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        evidence_kind=evidence_kind,
        parent_artifact_ids=tuple(parent_artifact_ids),
        sample_id=sample_id,
        model_version=model_version,
        script_profile_id=script_profile_id,
        data_hash=data_hash,
    )


class LegacyN1Adapter:
    """Preserve legacy N1 feature maps as computational visual features."""

    @staticmethod
    def to_visual_feature_bundle(result: Any, envelope: ContractEnvelope) -> VisualFeatureBundle:
        if envelope.contract_id != "IF-05":
            raise ValueError("expected IF-05 envelope")
        outputs = getattr(result, "stage_outputs", None)
        if not isinstance(outputs, Mapping):
            raise TypeError("legacy N1 result must expose stage_outputs")
        tensors: list[FeatureTensor] = []
        for key in ("normalized_energy", "temporal_activity", "spatial_pooled", "feature_vector"):
            value = outputs.get(key)
            if value is None:
                continue
            array = np.asarray(value)
            layout = {
                "normalized_energy": "ORIENTATION_Y_X",
                "temporal_activity": "ORIENTATION_Y_X",
                "spatial_pooled": "ORIENTATION_GRID_Y_GRID_X",
                "feature_vector": "FLAT",
            }[key]
            tensors.append(FeatureTensor(
                name=key,
                values=array,
                layout=layout,
                dtype=str(array.dtype),
                units="computational_activation",
                spatial_resolution=tuple(int(v) for v in array.shape[-2:]) if array.ndim >= 2 else (),
                representation_kind="legacy_ore_n1_visual_approximation",
            ))
        if not tensors:
            raise ValueError("legacy N1 result contains no supported feature tensors")
        return VisualFeatureBundle(
            envelope=envelope,
            feature_tensors=tuple(tensors),
            source_region_ids=(envelope.sample_id or "unspecified-region",),
            producer_kind="legacy_ore_n1_adapter",
            receptive_field_metadata=dict(getattr(result, "metadata", {}) or {}),
        )


class LegacyN2Adapter:
    """Adapt legacy N2 representations without assigning canonical biological ownership."""

    @staticmethod
    def to_visual_feature_bundle(result: Any, envelope: ContractEnvelope) -> VisualFeatureBundle:
        if envelope.contract_id != "IF-05":
            raise ValueError("expected IF-05 envelope")
        representations = getattr(result, "representations", None)
        if not isinstance(representations, Mapping):
            raise TypeError("legacy N2 result must expose representations")
        tensors = []
        for name, value in representations.items():
            if not isinstance(value, np.ndarray):
                continue
            tensors.append(FeatureTensor(
                name=str(name), values=value, layout="OPAQUE_N2_LAYOUT",
                dtype=str(value.dtype), units="computational_activation",
                spatial_resolution=tuple(int(v) for v in value.shape[-2:]) if value.ndim >= 2 else (),
                representation_kind="legacy_ore_n2_visual_structure",
            ))
        if not tensors:
            raise ValueError("legacy N2 result contains no array representations")
        return VisualFeatureBundle(
            envelope, tuple(tensors), (envelope.sample_id or "unspecified-region",),
            "legacy_ore_n2_adapter",
            receptive_field_metadata=dict(getattr(result, "metadata", {}) or {}),
        )


class LegacyN3Adapter:
    """Adapt N3 bottom-up probabilities without claiming they are calibrated."""

    @staticmethod
    def to_grapheme_evidence(
        result: Any,
        envelopes: Sequence[ContractEnvelope],
        *,
        inventory_version: str = "legacy-n3-v1",
    ) -> tuple[GraphemeEvidence, ...]:
        probabilities = np.asarray(getattr(result, "visual_probabilities", None), dtype=float)
        class_names = tuple(getattr(result, "class_names", ()))
        if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
            raise ValueError("N3 visual probability matrix and class vocabulary are incompatible")
        if len(envelopes) != probabilities.shape[0]:
            raise ValueError("one IF-06 envelope is required per patch")
        candidates = list(getattr(result, "candidates", []) or [])
        if candidates and len(candidates) != probabilities.shape[0]:
            raise ValueError("N3 candidates must align with probability rows")
        output = []
        for index, row in enumerate(probabilities):
            if not np.isfinite(row).all() or np.any(row < 0) or row.sum() <= 0:
                raise ValueError("N3 visual probabilities must be finite and non-negative")
            row = row / row.sum()
            candidate = candidates[index] if candidates else {}
            patch_id = str(candidate.get("patch_id", f"patch-{index}"))
            values = {name: float(row[j]) for j, name in enumerate(class_names)}
            unknown = float(values.get("<UNK>", 0.0))
            entropy = float(-np.sum(row * np.log(np.maximum(row, 1e-12))))
            output.append(GraphemeEvidence(
                envelope=envelopes[index], patch_id=patch_id,
                segmentation_hypothesis_id=candidate.get("segmentation_hypothesis_id"),
                grapheme_inventory_version=inventory_version,
                probabilities=values, unknown_probability=unknown,
                supporting_feature_ids=tuple(candidate.get("feature_ids", ())),
                entropy=entropy,
                reliability=float(candidate.get("segmentation_confidence", 1.0)),
                calibration_status="UNCALIBRATED",
                producer_backend="legacy.ore_n3",
            ))
        return tuple(output)


class LegacyN4Adapter:
    """Adapt predictive/attention signals; never invent C/A/B/O position codes."""

    @staticmethod
    def to_prediction_state(
        result: Any, envelope: ContractEnvelope, *, observed_evidence_ref: str
    ) -> PredictionState:
        if envelope.contract_id != "IF-10":
            raise ValueError("expected IF-10 envelope")
        return PredictionState(
            envelope=envelope,
            target="grapheme posterior",
            level="patch_sequence",
            predicted_values=np.asarray(result.contextual_priors),
            observed_evidence_ref=observed_evidence_ref,
            residual=np.asarray(result.classwise_prediction_error),
            residual_definition="N4 classwise posterior prediction residual",
            residual_units="probability_difference",
            sensory_precision=np.asarray(result.sensory_precision),
            prior_precision=np.asarray(result.prior_precision),
            pre_update_beliefs=np.asarray(result.n3_posterior_beliefs),
            post_update_beliefs=np.asarray(result.posterior_beliefs),
            iteration=max(0, len(getattr(result, "iteration_trace", ()))),
            stop_reason=str(getattr(result, "diagnostics", {}).get("stop_reason", "unspecified")),
            conflict_diagnostics={"source": "legacy.ore_n4", "calibration": "not_asserted"},
        )

    @staticmethod
    def position_code_bundle(
        envelope: ContractEnvelope,
        *,
        channels: Mapping[str, PositionCodeChannel] | None = None,
        patch_count: int = 0,
    ) -> PositionCodeBundle:
        """Build IF-07 only from explicitly supplied code implementations.

        Legacy N4 has no C/A/B/O channels. Without caller-supplied encodings the
        channels are explicitly unavailable rather than fabricated from attention.
        """
        if envelope.contract_id != "IF-07":
            raise ValueError("expected IF-07 envelope")
        defined = dict(channels or {})
        for key in ("C", "A", "B", "O"):
            defined.setdefault(key, PositionCodeChannel(
                channel=key, values=None, representation_kind="unavailable",
                units="not_available", backend_id="legacy.ore_n4:no-position-code",
                available=False,
            ))
        return PositionCodeBundle(
            envelope=envelope, channels=defined,
            logical_positions=tuple(range(patch_count)),
        )


class LegacyN5Adapter:
    """Adapt legacy N5 memory into IF-08 while retaining a legacy namespace."""

    @staticmethod
    def to_memory_snapshot(model: Any, envelope: ContractEnvelope) -> OrthographicMemorySnapshot:
        if envelope.contract_id != "IF-08":
            raise ValueError("expected IF-08 envelope")
        required = ("class_names", "fast_forward_counts", "slow_forward_counts",
                    "exposure_counts", "to_dict", "fingerprint")
        if any(not hasattr(model, item) for item in required):
            raise TypeError("legacy N5 model does not expose required memory fields")
        counts = np.asarray(model.fast_forward_counts, dtype=float) + np.asarray(model.slow_forward_counts, dtype=float)
        if counts.ndim != 2 or not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError("N5 transition counts are invalid")
        probabilities = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)
        exposure = {
            str(label): int(value)
            for label, value in zip(model.class_names, np.asarray(model.exposure_counts).tolist())
        }
        return OrthographicMemorySnapshot(
            envelope=envelope,
            memory_namespace="legacy.ore_n5.functional_memory_compatibility",
            transition_counts=counts,
            transition_probabilities=probabilities,
            chunks=(),
            exposure_counts=exposure,
            promotion_evidence=(),
            biological_memory_ref=None,
            memory_version=f"legacy-n5-{getattr(model, 'version', 'unknown')}",
        )


class LegacyN6Adapter:
    """Convert the highest-ranked N6 sequence while retaining alternatives and abstention."""

    @staticmethod
    def to_hypotheses(
        result: Any, envelope: ContractEnvelope, *, top_k: int = 5
    ) -> SequenceHypothesisSet:
        if envelope.contract_id != "IF-12":
            raise ValueError("expected IF-12 envelope")
        names = tuple(getattr(result, "class_names", ()))
        matrix = np.asarray(getattr(result, "decoded_probabilities", None), dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(names):
            raise ValueError("N6 decoded probabilities and class names are incompatible")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        decoded = tuple(getattr(result, "decoded_graphemes", ()))
        if len(decoded) != matrix.shape[0]:
            raise ValueError("N6 decoded grapheme count must match posterior rows")
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError("N6 decoded probabilities are invalid")
        row_sums = matrix.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("N6 decoded probability rows must have positive mass")
        matrix = matrix / row_sums
        rows = np.max(matrix, axis=1)
        confidence = float(np.mean(rows))
        path = tuple(f"position:{i}" for i in range(matrix.shape[0]))
        candidate = SequenceCandidate(
            graphemes=decoded, segmentation_path=path,
            component_scores={"S_ortho": confidence},
            score_definitions={"S_ortho": "mean maximum N6 decoded posterior; not calibrated"},
            confidence=confidence,
            decision="CANDIDATE",
            unknown_positions=tuple(i for i, label in enumerate(decoded) if label == "<UNK>"),
        )
        diagnostics = dict(getattr(result, "diagnostics", {}) or {})
        abstained = bool(diagnostics.get("abstained", False) or diagnostics.get("decision") == "ABSTAIN")
        return SequenceHypothesisSet(
            envelope=envelope, sequence_id=str(result.sequence_id),
            candidates=(candidate,), abstained=abstained,
            out_of_distribution=bool(diagnostics.get("out_of_distribution", False)),
        )


class LegacyN9Adapter:
    """Wrap the historical N9 result; it is not the canonical B-N9 controller."""

    @staticmethod
    def to_recognition_result(result: Any, envelope: ContractEnvelope) -> ORERecognitionResult:
        if envelope.contract_id != "IF-13":
            raise ValueError("expected IF-13 envelope")
        status_value = str((getattr(result, "decision", {}) or {}).get("status", "AMBIGUOUS")).upper()
        mapping = {
            "RESOLVED": "RESOLVED", "ACCEPT": "RESOLVED", "AMBIGUOUS": "AMBIGUOUS",
            "PARTIAL": "PARTIAL", "ABSTAIN": "ABSTAINED", "ABSTAINED": "ABSTAINED",
            "FAILED": "FAILED",
        }
        status = mapping.get(status_value, "AMBIGUOUS")
        probabilities = np.asarray(getattr(result, "posterior_beliefs", []), dtype=float)
        confidence = None
        if probabilities.ndim == 2 and probabilities.size:
            confidence = float(np.mean(np.max(probabilities, axis=1)))
        visual = np.asarray(getattr(result, "visual_probabilities", []), dtype=float)
        representation = {
            "visual_probabilities": visual.tolist(),
            "posterior_beliefs": probabilities.tolist(),
            "evidence_channels_separate": True,
            "legacy_source": "ore_n9",
        }
        hypotheses = SequenceHypothesisSet(
            envelope=make_envelope(
                "IF-12", envelope.run_id, "legacy.ore_n9.adapter",
                config={"source_envelope": envelope.fingerprint()},
                evidence_kind=EvidenceKind.DERIVED,
                sample_id=envelope.sample_id,
                data_hash=envelope.data_hash,
                script_profile_id=envelope.script_profile_id,
            ),
            sequence_id=str(result.sequence_id),
            candidates=(),
            abstained=True,
        )
        return ORERecognitionResult(
            envelope=envelope, status=status, hypotheses=hypotheses,
            orthographic_representation=representation,
            confidence=confidence, calibration_status="UNCALIBRATED",
            limitations=("Legacy N9 output is not evidence of canonical B-N9 integration.",),
            unsupported_channels=("canonical B-N1..B-N8 biological execution",),
        )


__all__ = [
    "make_envelope", "LegacyN1Adapter", "LegacyN2Adapter", "LegacyN3Adapter",
    "LegacyN4Adapter", "LegacyN5Adapter", "LegacyN6Adapter", "LegacyN9Adapter",
]
