"""Framework-neutral ORE registry and P1 contracts (IF-00 through IF-04)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import re
from typing import Any, Mapping


class ArtifactStatus(str, Enum):
    SPECIFIED = "SPECIFIED"
    IMPLEMENTED = "IMPLEMENTED"
    SOFTWARE_TESTED = "SOFTWARE_TESTED"
    TASK_VALIDATED = "TASK_VALIDATED"
    BIOLOGICALLY_SUPPORTED = "BIOLOGICALLY_SUPPORTED"
    PENDING = "PENDING"
    NOT_TESTED = "NOT_TESTED"
    FAILED = "FAILED"


class OperationMode(str, Enum):
    FUNCTIONAL = "FUNCTIONAL"
    BIOLOGICAL = "BIOLOGICAL"
    HYBRID = "HYBRID"
    TRAIN = "TRAIN"
    INFER = "INFER"
    VALIDATE = "VALIDATE"


class EvidenceKind(str, Enum):
    OBSERVED = "OBSERVED"
    DERIVED = "DERIVED"
    PREDICTED = "PREDICTED"
    LEARNED = "LEARNED"
    CALIBRATED = "CALIBRATED"
    EXTERNAL_REFERENCE = "EXTERNAL_REFERENCE"


class ReadingDirection(str, Enum):
    LTR = "LTR"
    RTL = "RTL"
    TTB = "TTB"
    BTT = "BTT"
    MIXED = "MIXED"


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    phase_id: str
    name: str
    responsibility: str
    track: str
    status: ArtifactStatus = ArtifactStatus.SPECIFIED


FUNCTIONAL_PHASES = (
    PhaseSpec("F-A", "Initialize and validate", "Configuration, script profile, run state and compatibility checks.", "FUNCTIONAL"),
    PhaseSpec("F-B", "Prepare learning data", "Source registration, hashes, labels, split policy and safe transformations.", "FUNCTIONAL"),
    PhaseSpec("F-C", "Learn visual and character representations", "Functional visual encoder, identity evidence, view invariance and unknown mass.", "FUNCTIONAL"),
    PhaseSpec("F-D", "Learn position and order representations", "Independent absolute, relative/open-bigram and overlap position backends.", "FUNCTIONAL"),
    PhaseSpec("F-E", "Learn orthographic regularities", "Soft transitions, candidate chunks, promotion and exposure memory.", "FUNCTIONAL"),
    PhaseSpec("F-F", "Train, evaluate and freeze", "Functional optimization, validation, calibration and immutable checkpoints.", "FUNCTIONAL"),
    PhaseSpec("F-G", "Recognize a visual string", "Candidate generation, evidence scoring, ranking and abstention.", "FUNCTIONAL"),
    PhaseSpec("F-H", "Return result and trace", "Structured output, trace serialization and provenance.", "FUNCTIONAL"),
    PhaseSpec("F-I", "Optional incremental learning", "Authorized child-model update, regression and rollback.", "FUNCTIONAL"),
    PhaseSpec("F-J", "Acceptance and release", "Software gates, task experiments, reproducibility and release decision.", "FUNCTIONAL"),
)

BIOLOGICAL_PHASES = (
    PhaseSpec("B-N1", "Neural simulation foundations", "Neuron state, membrane dynamics, firing, synaptic transmission, populations and simulation time.", "BIOLOGICAL"),
    PhaseSpec("B-N2", "Biological visual processing", "Retinal/LGN-inspired preprocessing and V1-like visual responses.", "BIOLOGICAL"),
    PhaseSpec("B-N3", "Hierarchical visual architecture", "Intermediate/higher visual populations, feature composition and tested invariance.", "BIOLOGICAL"),
    PhaseSpec("B-N4", "Neural orthographic representations", "Distributed grapheme identity, identity-position conjunctions and neural sequence codes.", "BIOLOGICAL"),
    PhaseSpec("B-N5", "Neural connectivity and recurrence", "Excitatory/inhibitory, local/long-range, recurrent and feedback interactions.", "BIOLOGICAL"),
    PhaseSpec("B-N6", "Biological learning and memory", "Experience-dependent plasticity, homeostasis, retention and forgetting.", "BIOLOGICAL"),
    PhaseSpec("B-N7", "Predictive and contextual processing", "Top-down predictions, bottom-up residuals, uncertainty and precision modulation.", "BIOLOGICAL"),
    PhaseSpec("B-N8", "Brain-data validation", "Independent behavioral, representational, temporal and connectivity validation.", "BIOLOGICAL"),
    PhaseSpec("B-N9", "Integrated neurocomputational ORE", "Orchestration, dependency scheduling, evidence ledger, lineage and release governance.", "BIOLOGICAL"),
)
ALL_PHASES = FUNCTIONAL_PHASES + BIOLOGICAL_PHASES
PHASE_BY_ID = {phase.phase_id: phase for phase in ALL_PHASES}
FUNCTIONAL_TO_BIOLOGICAL = {
    "F-A": tuple(f"B-N{i}" for i in range(1, 10)), "F-B": ("B-N2", "B-N3", "B-N8"),
    "F-C": ("B-N2", "B-N3"), "F-D": ("B-N4",),
    "F-E": ("B-N6", "B-N7"), "F-F": ("B-N1", "B-N6"),
    "F-G": ("B-N2", "B-N3", "B-N4", "B-N5", "B-N6", "B-N7"),
    "F-H": tuple(f"B-N{i}" for i in range(1, 10)),
    "F-I": ("B-N6",), "F-J": ("B-N8", "B-N9"),
}


def validate_registry() -> None:
    ids = [phase.phase_id for phase in ALL_PHASES]
    if len(ids) != len(set(ids)):
        raise ValueError("Canonical phase IDs must be unique.")
    if set(FUNCTIONAL_TO_BIOLOGICAL) != {p.phase_id for p in FUNCTIONAL_PHASES}:
        raise ValueError("Crosswalk must define every functional phase.")
    biological_ids = {p.phase_id for p in BIOLOGICAL_PHASES}
    if any(set(v) - biological_ids for v in FUNCTIONAL_TO_BIOLOGICAL.values()):
        raise ValueError("Crosswalk contains an unknown biological phase.")


validate_registry()

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_sha256(value: str | None, field: str, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 digest.")


@dataclass(frozen=True, slots=True)
class ContractEnvelope:
    contract_id: str
    schema_version: str
    run_id: str
    producer_id: str
    config_hash: str
    created_at: str
    evidence_kind: EvidenceKind
    parent_artifact_ids: tuple[str, ...] = ()
    sample_id: str | None = None
    model_version: str | None = None
    script_profile_id: str | None = None
    data_hash: str | None = None

    def __post_init__(self) -> None:
        for field in ("contract_id", "schema_version", "run_id", "producer_id"):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must be non-empty.")
        _require_sha256(self.config_hash, "config_hash")
        _require_sha256(self.data_hash, "data_hash", optional=True)
        if not isinstance(self.evidence_kind, EvidenceKind):
            object.__setattr__(self, "evidence_kind", EvidenceKind(self.evidence_kind))
        if len(set(self.parent_artifact_ids)) != len(self.parent_artifact_ids):
            raise ValueError("parent_artifact_ids must not contain duplicates.")
        try:
            parsed = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError("created_at must be ISO-8601.") from exc
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError("created_at must be UTC.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_kind"] = self.evidence_kind.value
        data["parent_artifact_ids"] = list(self.parent_artifact_ids)
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContractEnvelope":
        data = dict(value)
        data["evidence_kind"] = EvidenceKind(data["evidence_kind"])
        data["parent_artifact_ids"] = tuple(data.get("parent_artifact_ids", ()))
        return cls(**data)

    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class ScriptProfile:
    profile_id: str
    profile_version: str
    script_id: str
    reading_direction: ReadingDirection
    grapheme_segmentation_policy: str
    unicode_normalization: str
    case_policy: str = "PRESERVE"
    diacritic_policy: str = "PRESERVE"
    joining_policy: str = "SCRIPT_DEFINED"
    ligature_policy: str = "PRESERVE"
    punctuation_policy: str = "PRESERVE"
    logical_reading_axis: tuple[float, float] = (1.0, 0.0)
    mixed_layout_policy: str | None = None
    geometry_constraints: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for field in ("profile_id", "profile_version", "script_id",
                      "grapheme_segmentation_policy", "unicode_normalization"):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must be non-empty.")
        if not isinstance(self.reading_direction, ReadingDirection):
            object.__setattr__(self, "reading_direction", ReadingDirection(self.reading_direction))
        if len(self.logical_reading_axis) != 2 or self.logical_reading_axis == (0, 0):
            raise ValueError("logical_reading_axis must be a non-zero 2D vector.")
        if self.reading_direction is ReadingDirection.MIXED and not self.mixed_layout_policy:
            raise ValueError("MIXED direction requires mixed_layout_policy.")
        if len(dict(self.geometry_constraints)) != len(self.geometry_constraints):
            raise ValueError("geometry_constraints keys must be unique.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reading_direction"] = self.reading_direction.value
        data["logical_reading_axis"] = list(self.logical_reading_axis)
        data["geometry_constraints"] = dict(self.geometry_constraints)
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScriptProfile":
        data = dict(value)
        data["reading_direction"] = ReadingDirection(data["reading_direction"])
        data["logical_reading_axis"] = tuple(data.get("logical_reading_axis", (1.0, 0.0)))
        constraints = data.get("geometry_constraints", {})
        data["geometry_constraints"] = tuple(sorted(dict(constraints).items())) if isinstance(constraints, Mapping) else tuple(constraints)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ModelBundleRef:
    family: str
    version: str
    fingerprint: str
    schema_version: str
    script_profile_compatibility: tuple[str, ...] = ()
    parent_fingerprint: str | None = None
    checkpoint_ref: str | None = None

    def __post_init__(self) -> None:
        for field in ("family", "version", "fingerprint", "schema_version"):
            if not getattr(self, field).strip():
                raise ValueError(f"ModelBundleRef.{field} must be non-empty.")
        _require_sha256(self.fingerprint, "model fingerprint")
        _require_sha256(self.parent_fingerprint, "parent_fingerprint", optional=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelBundleRef":
        data = dict(value)
        data["script_profile_compatibility"] = tuple(data.get("script_profile_compatibility", ()))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ComputeBudget:
    wall_clock_seconds: float | None = None
    max_recurrent_iterations: int = 0
    max_memory_mb: float | None = None

    def __post_init__(self) -> None:
        if self.wall_clock_seconds is not None and self.wall_clock_seconds <= 0:
            raise ValueError("wall_clock_seconds must be positive.")
        if self.max_recurrent_iterations < 0:
            raise ValueError("max_recurrent_iterations cannot be negative.")
        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive.")


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    mode: OperationMode
    deterministic_seed: int
    compute_budget: ComputeBudget
    module_backend_selection: tuple[tuple[str, str], ...] = ()
    timeout_limits_seconds: tuple[tuple[str, float], ...] = ()
    model_refs: tuple[ModelBundleRef, ...] = ()
    settings_json: str = "{}"

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty.")
        if not isinstance(self.mode, OperationMode):
            object.__setattr__(self, "mode", OperationMode(self.mode))
        if self.deterministic_seed < 0:
            raise ValueError("deterministic_seed cannot be negative.")
        if len(dict(self.module_backend_selection)) != len(self.module_backend_selection):
            raise ValueError("module backend keys must be unique.")
        if len(dict(self.timeout_limits_seconds)) != len(self.timeout_limits_seconds):
            raise ValueError("timeout keys must be unique.")
        if any(seconds <= 0 for _, seconds in self.timeout_limits_seconds):
            raise ValueError("timeout limits must be positive.")
        if len({ref.family for ref in self.model_refs}) != len(self.model_refs):
            raise ValueError("Only one model reference per family is allowed.")
        parsed = json.loads(self.settings_json)
        if not isinstance(parsed, dict):
            raise ValueError("settings_json must encode a JSON object.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        data["module_backend_selection"] = dict(self.module_backend_selection)
        data["timeout_limits_seconds"] = dict(self.timeout_limits_seconds)
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunContext":
        data = dict(value)
        data["mode"] = OperationMode(data["mode"])
        data["compute_budget"] = ComputeBudget(**data["compute_budget"])
        data["module_backend_selection"] = tuple(sorted(dict(data.get("module_backend_selection", {})).items()))
        data["timeout_limits_seconds"] = tuple(sorted(dict(data.get("timeout_limits_seconds", {})).items()))
        data["model_refs"] = tuple(
            item if isinstance(item, ModelBundleRef) else ModelBundleRef.from_dict(item)
            for item in data.get("model_refs", ())
        )
        return cls(**data)


@dataclass(frozen=True, slots=True)
class SegmentationHypothesis:
    patch_id: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    logical_order: int
    alternative_group_id: str | None = None

    def __post_init__(self) -> None:
        if not self.patch_id.strip():
            raise ValueError("patch_id must be non-empty.")
        left, top, right, bottom = self.bbox_xyxy
        if right <= left or bottom <= top:
            raise ValueError("bbox must have positive width and height.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1].")
        if self.logical_order < 0:
            raise ValueError("logical_order cannot be negative.")


@dataclass(frozen=True, slots=True)
class VisualObservation:
    observation_id: str
    raw_source_ref: str
    source_hash: str
    script_profile_id: str
    image_width: int
    image_height: int
    pixel_numeric_range: tuple[float, float]
    pixel_payload_ref: str | None = None
    patch_payload_ref: str | None = None
    channel_order: str = "GRAY"
    layout: str = "SINGLE_IMAGE"
    coordinate_frame: str = "IMAGE_PIXELS_X_RIGHT_Y_DOWN"
    missing_mask_ref: str | None = None
    segmentation_hypotheses: tuple[SegmentationHypothesis, ...] = ()
    source_license: str | None = None
    source_provenance: str | None = None
    parent_observation_id: str | None = None
    transformation_parameters_json: str = "{}"

    def __post_init__(self) -> None:
        for field in ("observation_id", "raw_source_ref", "script_profile_id",
                      "channel_order", "layout", "coordinate_frame"):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must be non-empty.")
        _require_sha256(self.source_hash, "source_hash")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive.")
        low, high = self.pixel_numeric_range
        if high <= low:
            raise ValueError("pixel_numeric_range must be increasing.")
        if not self.pixel_payload_ref and not self.patch_payload_ref:
            raise ValueError("pixel_payload_ref or patch_payload_ref is required.")
        if len({p.patch_id for p in self.segmentation_hypotheses}) != len(self.segmentation_hypotheses):
            raise ValueError("Segmentation patch IDs must be unique.")
        if not isinstance(json.loads(self.transformation_parameters_json), dict):
            raise ValueError("transformation_parameters_json must encode an object.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VisualObservation":
        data = dict(value)
        data["pixel_numeric_range"] = tuple(data["pixel_numeric_range"])
        data["segmentation_hypotheses"] = tuple(
            item if isinstance(item, SegmentationHypothesis) else SegmentationHypothesis(**item)
            for item in data.get("segmentation_hypotheses", ())
        )
        return cls(**data)


@dataclass(frozen=True, slots=True)
class DatasetSample:
    sample_id: str
    sample_hash: str
    split: str
    label: str | None = None
    label_provenance: str | None = None
    writer_id: str | None = None
    font_id: str | None = None
    source_id: str | None = None
    string_family_id: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty.")
        _require_sha256(self.sample_hash, "sample_hash")
        if self.split not in {"TRAIN", "VALIDATION", "TEST"}:
            raise ValueError("split must be TRAIN, VALIDATION or TEST.")
        if self.label is not None and not self.label_provenance:
            raise ValueError("A supplied label requires label_provenance.")


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    dataset_version: str
    license_id: str
    provenance: str
    samples: tuple[DatasetSample, ...]
    transformation_manifest_json: str = "{}"
    seed: int = 0
    preregistered: bool = False
    test_split_locked: bool = True

    def __post_init__(self) -> None:
        for field in ("dataset_id", "dataset_version", "license_id", "provenance"):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must be non-empty.")
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")
        if len({s.sample_id for s in self.samples}) != len(self.samples):
            raise ValueError("sample IDs must be unique.")
        if len({s.sample_hash for s in self.samples}) != len(self.samples):
            raise ValueError("Duplicate sample hashes are not allowed.")
        if not isinstance(json.loads(self.transformation_manifest_json), dict):
            raise ValueError("transformation_manifest_json must encode an object.")
        self.check_group_leakage()

    def check_group_leakage(self) -> None:
        for field in ("writer_id", "font_id", "source_id", "string_family_id"):
            assignments: dict[str, set[str]] = {}
            for sample in self.samples:
                group = getattr(sample, field)
                if group is not None:
                    assignments.setdefault(group, set()).add(sample.split)
            leaked = [key for key, splits in assignments.items() if len(splits) > 1]
            if leaked:
                raise ValueError(f"Leakage detected across splits for {field}: {sorted(leaked)}")
        if self.preregistered and not self.test_split_locked:
            raise ValueError("A preregistered manifest must keep its test split locked.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id, "dataset_version": self.dataset_version,
            "license_id": self.license_id, "provenance": self.provenance,
            "samples": [asdict(s) for s in self.samples],
            "transformation_manifest_json": self.transformation_manifest_json,
            "seed": self.seed, "preregistered": self.preregistered,
            "test_split_locked": self.test_split_locked,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetManifest":
        data = dict(value)
        data["samples"] = tuple(
            item if isinstance(item, DatasetSample) else DatasetSample(**item)
            for item in data["samples"]
        )
        return cls(**data)

    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


__all__ = [
    "ArtifactStatus", "OperationMode", "EvidenceKind", "ReadingDirection",
    "PhaseSpec", "FUNCTIONAL_PHASES", "BIOLOGICAL_PHASES", "ALL_PHASES",
    "PHASE_BY_ID", "FUNCTIONAL_TO_BIOLOGICAL", "validate_registry",
    "canonical_json", "sha256_json", "utc_now_iso", "ContractEnvelope",
    "ScriptProfile", "ModelBundleRef", "ComputeBudget", "RunContext",
    "SegmentationHypothesis", "VisualObservation", "DatasetSample",
    "DatasetManifest",
]


# Extended canonical interfaces (P2-P8).
from .advanced import (
    AcceptanceStatus, BiologicalValidator, BiologicalVisualBackend,
    EvaluationReport, ExperimentSpec, FeatureTensor, FunctionalVisualBackend,
    GraphemeEvidence, LearningRequest, LearningResult, MetricRecord,
    NeuralInputEvent, NeuralSimulationConfig, NeuralSimulator, NeuralStateSnapshot,
    ORERecognitionResult, OrthographicBackend, OrthographicMemorySnapshot,
    PlasticityBackend, PositionCodeBundle, PositionCodeChannel, PredictiveBackend,
    PredictionState, RecurrentBackend, SequenceCandidate, SequenceDecoder,
    SequenceHypothesisSet, TraceEvent, VisualFeatureBundle, evaluate_experiment,
    SeededReplicate, ExperimentExecutionReport, run_seeded_replicates,
)
from .adapters import (
    LegacyN1Adapter, LegacyN2Adapter, LegacyN3Adapter, LegacyN4Adapter,
    LegacyN5Adapter, LegacyN6Adapter, LegacyN9Adapter, BiologicalStateAdapter, make_envelope,
)
from .governance import LearningTransactionManager, LearningTransactionOutcome, model_fingerprint
from .integration import (
    CANONICAL_BIOLOGICAL_PHASES, INFERENCE_BIOLOGICAL_PHASES,
    BiologicalPhaseBackend, CanonicalBackendRegistry, CanonicalOREController,
    CanonicalRunResult, FrozenFusionPolicy, FunctionalEngine,
    LegacyN9FunctionalBackend, PhaseInvocation,
)
from .simulation import RateBasedB1PhaseBackend, RateBasedNeuralSimulator

__all__ += [
    "AcceptanceStatus", "EvaluationReport", "ExperimentSpec", "FeatureTensor",
    "GraphemeEvidence", "PositionCodeChannel", "PositionCodeBundle",
    "OrthographicMemorySnapshot", "LearningRequest", "LearningResult",
    "PredictionState", "SequenceCandidate", "SequenceHypothesisSet",
    "ORERecognitionResult", "TraceEvent", "NeuralInputEvent",
    "NeuralSimulationConfig", "NeuralStateSnapshot", "NeuralSimulator",
    "FunctionalVisualBackend", "BiologicalVisualBackend", "OrthographicBackend",
    "RecurrentBackend", "PlasticityBackend", "PredictiveBackend", "SequenceDecoder",
    "BiologicalValidator", "VisualFeatureBundle", "MetricRecord",
    "evaluate_experiment", "SeededReplicate", "ExperimentExecutionReport",
    "run_seeded_replicates", "RateBasedNeuralSimulator", "RateBasedB1PhaseBackend",
    "LearningTransactionManager", "LearningTransactionOutcome", "model_fingerprint",
    "make_envelope", "BiologicalStateAdapter", "LegacyN1Adapter", "LegacyN2Adapter", "LegacyN3Adapter",
    "LegacyN4Adapter", "LegacyN5Adapter", "LegacyN6Adapter", "LegacyN9Adapter",
    "CANONICAL_BIOLOGICAL_PHASES", "INFERENCE_BIOLOGICAL_PHASES",
    "PhaseInvocation", "FunctionalEngine", "BiologicalPhaseBackend",
    "FrozenFusionPolicy", "CanonicalBackendRegistry", "CanonicalRunResult",
    "CanonicalOREController", "LegacyN9FunctionalBackend",
]
