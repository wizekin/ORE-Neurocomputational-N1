"""Canonical ORE contracts IF-05 through IF-18 and backend protocols.

These are data/interface contracts. They do not claim that a backend is
biologically validated or that similarly named legacy N stages are equivalent.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from . import ContractEnvelope, EvidenceKind, ModelBundleRef, OperationMode, sha256_json


def _frozen_array(value: Any | None, name: str, ndim: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value).copy()
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class FeatureTensor:
    name: str
    values: np.ndarray
    layout: str
    dtype: str
    units: str
    spatial_resolution: tuple[int, ...] = ()
    representation_kind: str = "unspecified"

    def __post_init__(self) -> None:
        for field_name in ("name", "layout", "units", "representation_kind"):
            _nonempty(getattr(self, field_name), field_name)
        array = _frozen_array(self.values, self.name)
        if array.size == 0:
            raise ValueError("feature tensor cannot be empty")
        if self.spatial_resolution and any(int(v) <= 0 for v in self.spatial_resolution):
            raise ValueError("spatial_resolution values must be positive")
        object.__setattr__(self, "values", array)
        object.__setattr__(self, "dtype", str(array.dtype))


@dataclass(frozen=True, slots=True)
class VisualFeatureBundle:
    """IF-05 shared visual representation; units and representation kind are mandatory."""

    envelope: ContractEnvelope
    feature_tensors: tuple[FeatureTensor, ...]
    source_region_ids: tuple[str, ...]
    producer_kind: str
    receptive_field_metadata: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: np.ndarray | None = None
    reliability: np.ndarray | None = None
    transform_parent_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.envelope.contract_id != "IF-05":
            raise ValueError("VisualFeatureBundle requires an IF-05 envelope")
        if not self.feature_tensors:
            raise ValueError("VisualFeatureBundle requires at least one tensor")
        if len({tensor.name for tensor in self.feature_tensors}) != len(self.feature_tensors):
            raise ValueError("feature tensor names must be unique")
        _nonempty(self.producer_kind, "producer_kind")
        object.__setattr__(self, "uncertainty", _frozen_array(self.uncertainty, "uncertainty"))
        object.__setattr__(self, "reliability", _frozen_array(self.reliability, "reliability"))


@dataclass(frozen=True, slots=True)
class GraphemeEvidence:
    """IF-06 per-region identity evidence; activations are not presumed probabilities."""

    envelope: ContractEnvelope
    patch_id: str
    segmentation_hypothesis_id: str | None
    grapheme_inventory_version: str
    probabilities: Mapping[str, float]
    unknown_probability: float
    raw_activations: Mapping[str, float] = field(default_factory=dict)
    supporting_feature_ids: tuple[str, ...] = ()
    entropy: float | None = None
    reliability: float = 1.0
    calibration_status: str = "UNCALIBRATED"
    producer_backend: str = "unspecified"

    def __post_init__(self) -> None:
        if self.envelope.contract_id != "IF-06":
            raise ValueError("GraphemeEvidence requires an IF-06 envelope")
        for field_name in ("patch_id", "grapheme_inventory_version", "calibration_status", "producer_backend"):
            _nonempty(getattr(self, field_name), field_name)
        if not self.probabilities or any(not k for k in self.probabilities):
            raise ValueError("probabilities must contain non-empty grapheme keys")
        values = np.asarray(list(self.probabilities.values()), dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("grapheme probabilities must be finite and non-negative")
        if not np.isfinite(self.unknown_probability) or not 0 <= self.unknown_probability <= 1:
            raise ValueError("unknown_probability must be in [0, 1]")
        if abs(float(values.sum()) - 1.0) > 1e-4:
            raise ValueError("probabilities, including any explicit UNK entry, must sum to one")
        if not 0 <= self.reliability <= 1:
            raise ValueError("reliability must be in [0, 1]")
        if self.entropy is not None and (not np.isfinite(self.entropy) or self.entropy < 0):
            raise ValueError("entropy must be finite and non-negative")
        if any(not np.isfinite(v) for v in self.raw_activations.values()):
            raise ValueError("raw_activations must be finite")


@dataclass(frozen=True, slots=True)
class PositionCodeChannel:
    channel: str
    values: np.ndarray | None
    representation_kind: str
    units: str
    backend_id: str
    available: bool
    provenance_id: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.channel not in {"C", "A", "B", "O"}:
            raise ValueError("position channel must be C, A, B or O")
        for field_name in ("representation_kind", "units", "backend_id"):
            _nonempty(getattr(self, field_name), field_name)
        array = _frozen_array(self.values, f"{self.channel}.values")
        if self.available and array is None:
            raise ValueError(f"available channel {self.channel} requires values")
        if not self.available and array is not None:
            raise ValueError(f"unavailable channel {self.channel} cannot contain fabricated values")
        object.__setattr__(self, "values", array)


@dataclass(frozen=True, slots=True)
class PositionCodeBundle:
    """IF-07 separates identity C and positional A/B/O encodings."""

    envelope: ContractEnvelope
    channels: Mapping[str, PositionCodeChannel]
    physical_coordinates: np.ndarray | None = None
    logical_positions: tuple[int, ...] = ()
    uncertainty: np.ndarray | None = None
    positional_reliability: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.envelope.contract_id != "IF-07":
            raise ValueError("PositionCodeBundle requires an IF-07 envelope")
        if set(self.channels) != {"C", "A", "B", "O"}:
            raise ValueError("PositionCodeBundle must explicitly define C, A, B and O")
        for key, channel in self.channels.items():
            if channel.channel != key:
                raise ValueError("channel mapping key must match channel identity")
        if self.logical_positions and any(i < 0 for i in self.logical_positions):
            raise ValueError("logical positions cannot be negative")
        object.__setattr__(self, "physical_coordinates", _frozen_array(self.physical_coordinates, "physical_coordinates"))
        object.__setattr__(self, "uncertainty", _frozen_array(self.uncertainty, "uncertainty"))
        object.__setattr__(self, "positional_reliability", _frozen_array(self.positional_reliability, "positional_reliability"))


@dataclass(frozen=True, slots=True)
class OrthographicMemorySnapshot:
    """IF-08 functional orthographic memory; biological synapses remain separate."""

    envelope: ContractEnvelope
    memory_namespace: str
    transition_counts: np.ndarray | None = None
    transition_probabilities: np.ndarray | None = None
    chunks: tuple[Mapping[str, Any], ...] = ()
    exposure_counts: Mapping[str, int] = field(default_factory=dict)
    promotion_evidence: tuple[Mapping[str, Any], ...] = ()
    biological_memory_ref: str | None = None
    memory_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.envelope.contract_id != "IF-08":
            raise ValueError("OrthographicMemorySnapshot requires an IF-08 envelope")
        _nonempty(self.memory_namespace, "memory_namespace")
        for name in ("transition_counts", "transition_probabilities"):
            object.__setattr__(self, name, _frozen_array(getattr(self, name), name))
        if self.transition_counts is not None and self.transition_probabilities is not None:
            if self.transition_counts.shape != self.transition_probabilities.shape:
                raise ValueError("transition count and probability shapes must agree")
        if any(int(v) < 0 for v in self.exposure_counts.values()):
            raise ValueError("exposure counts cannot be negative")


@dataclass(frozen=True, slots=True)
class LearningRequest:
    """IF-09 explicit, provenance-bound permission to create a detached model child."""

    parent_model_fingerprint: str
    training_sample_ids: tuple[str, ...]
    split_authorization: str
    verified_targets: Mapping[str, str]
    verification_provenance: Mapping[str, str]
    update_channels: tuple[str, ...]
    explicitly_authorized: bool
    policy_reference: str
    regression_suite_id: str
    unsupervised_regime: str | None = None

    def __post_init__(self) -> None:
        for name in ("parent_model_fingerprint", "split_authorization", "policy_reference", "regression_suite_id"):
            _nonempty(getattr(self, name), name)
        if not self.training_sample_ids or len(set(self.training_sample_ids)) != len(self.training_sample_ids):
            raise ValueError("training_sample_ids must be non-empty and unique")
        if not self.update_channels:
            raise ValueError("at least one update channel must be declared")
        if not self.verified_targets and not self.unsupervised_regime:
            raise ValueError("verified targets or an explicitly named unsupervised regime is required")
        if self.verified_targets and any(k not in self.verification_provenance for k in self.verified_targets):
            raise ValueError("every verified target requires reviewer/source provenance")


@dataclass(frozen=True, slots=True)
class LearningResult:
    """IF-15 decision and lineage record for a learning transaction."""

    decision: str
    parent_fingerprint: str
    child_fingerprint: str | None
    changed_state_refs: tuple[str, ...]
    tests_executed: tuple[str, ...]
    metrics_before: Mapping[str, float] = field(default_factory=dict)
    metrics_after: Mapping[str, float] = field(default_factory=dict)
    regression_failures: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision not in {"ACCEPTED", "REJECTED", "ROLLED_BACK"}:
            raise ValueError("decision must be ACCEPTED, REJECTED or ROLLED_BACK")
        _nonempty(self.parent_fingerprint, "parent_fingerprint")
        if self.decision == "ACCEPTED" and not self.child_fingerprint:
            raise ValueError("accepted learning result requires child fingerprint")
        if self.decision != "ACCEPTED" and self.child_fingerprint is not None:
            raise ValueError("rejected or rolled-back results cannot expose a committed child")


@dataclass(frozen=True, slots=True)
class PredictionState:
    """IF-10 prediction, observation reference, residual and precision remain distinct."""

    envelope: ContractEnvelope
    target: str
    level: str
    predicted_values: np.ndarray
    observed_evidence_ref: str
    residual: np.ndarray
    residual_definition: str
    residual_units: str
    sensory_precision: np.ndarray
    prior_precision: np.ndarray
    pre_update_beliefs: np.ndarray | None = None
    post_update_beliefs: np.ndarray | None = None
    iteration: int = 0
    damping: float = 1.0
    stop_reason: str = "not_iterated"
    conflict_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.envelope.contract_id != "IF-10":
            raise ValueError("PredictionState requires an IF-10 envelope")
        for name in ("target", "level", "observed_evidence_ref", "residual_definition", "residual_units", "stop_reason"):
            _nonempty(getattr(self, name), name)
        for name in ("predicted_values", "residual", "sensory_precision", "prior_precision",
                     "pre_update_beliefs", "post_update_beliefs"):
            object.__setattr__(self, name, _frozen_array(getattr(self, name), name))
        if self.iteration < 0 or not 0 < self.damping <= 1:
            raise ValueError("iteration must be non-negative and damping in (0, 1]")


@dataclass(frozen=True, slots=True)
class SequenceCandidate:
    graphemes: tuple[str, ...]
    segmentation_path: tuple[str, ...]
    component_scores: Mapping[str, float]
    score_definitions: Mapping[str, str]
    confidence: float
    decision: str = "CANDIDATE"
    unknown_positions: tuple[int, ...] = ()
    neural_trace_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.graphemes:
            raise ValueError("candidate must contain a grapheme sequence")
        if not 0 <= self.confidence <= 1:
            raise ValueError("candidate confidence must be in [0, 1]")
        if set(self.component_scores) - {"S_vis", "S_geometry", "S_position", "S_ortho", "S_recurrent", "S_predictive", "S_neural"}:
            raise ValueError("unsupported candidate score channel")
        if any(not np.isfinite(v) for v in self.component_scores.values()):
            raise ValueError("candidate component scores must be finite")


@dataclass(frozen=True, slots=True)
class SequenceHypothesisSet:
    """IF-12 alternatives and channel scores; fusion is never implicit."""

    envelope: ContractEnvelope
    sequence_id: str
    candidates: tuple[SequenceCandidate, ...]
    abstained: bool = False
    out_of_distribution: bool = False
    fusion_policy_ref: str | None = None

    def __post_init__(self) -> None:
        if self.envelope.contract_id != "IF-12":
            raise ValueError("SequenceHypothesisSet requires an IF-12 envelope")
        _nonempty(self.sequence_id, "sequence_id")
        if not self.candidates and not self.abstained:
            raise ValueError("empty candidates require explicit abstention")


@dataclass(frozen=True, slots=True)
class ORERecognitionResult:
    """IF-13 canonical orthographic output, with uncertainty and provenance preserved."""

    envelope: ContractEnvelope
    status: str
    hypotheses: SequenceHypothesisSet
    orthographic_representation: Mapping[str, Any]
    neural_state_ref: str | None = None
    prediction_state_refs: tuple[str, ...] = ()
    confidence: float | None = None
    calibration_status: str = "UNCALIBRATED"
    run_manifest_ref: str | None = None
    limitations: tuple[str, ...] = ()
    unsupported_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.envelope.contract_id != "IF-13":
            raise ValueError("ORERecognitionResult requires an IF-13 envelope")
        if self.status not in {"RESOLVED", "AMBIGUOUS", "PARTIAL", "ABSTAINED", "FAILED"}:
            raise ValueError("unsupported recognition status")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """IF-14 append-only trace record with an ordered event index."""

    event_index: int
    wall_clock_utc: str
    module_id: str
    operation: str
    input_artifact_ids: tuple[str, ...]
    output_artifact_ids: tuple[str, ...]
    status: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    simulation_time_ms: float | None = None
    numerical_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_index < 0:
            raise ValueError("event_index cannot be negative")
        for name in ("wall_clock_utc", "module_id", "operation", "status"):
            _nonempty(getattr(self, name), name)
        if self.simulation_time_ms is not None and self.simulation_time_ms < 0:
            raise ValueError("simulation_time_ms cannot be negative")


class AcceptanceStatus(str, Enum):
    PASS = "PASS"
    PENDING = "PENDING"
    NOT_TESTED = "NOT_TESTED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class NeuralInputEvent:
    unit_index: int
    value: float
    event_time_ms: float
    signal_kind: str = "current"

    def __post_init__(self) -> None:
        if self.unit_index < 0 or not np.isfinite(self.value) or not np.isfinite(self.event_time_ms):
            raise ValueError("neural input event fields are invalid")
        _nonempty(self.signal_kind, "signal_kind")


@dataclass(frozen=True, slots=True)
class NeuralSimulationConfig:
    population_sizes: tuple[tuple[str, int], ...]
    recurrent_weights: np.ndarray
    dt_ms: float = 1.0
    membrane_tau_ms: float = 20.0
    resting_potential_mv: float = -70.0
    threshold_mv: float = -55.0
    sigmoid_slope_per_mv: float = 0.25
    max_rate_hz: float = 100.0
    model_class: str = "rate_based"

    def __post_init__(self) -> None:
        if not self.population_sizes or len({p for p, _ in self.population_sizes}) != len(self.population_sizes):
            raise ValueError("population IDs must be non-empty and unique")
        if any(not p or int(n) <= 0 for p, n in self.population_sizes):
            raise ValueError("population sizes must be positive")
        n = sum(size for _, size in self.population_sizes)
        weights = _frozen_array(self.recurrent_weights, "recurrent_weights", ndim=2)
        if weights.shape != (n, n):
            raise ValueError(f"recurrent_weights must have shape ({n}, {n})")
        object.__setattr__(self, "recurrent_weights", weights)
        if self.dt_ms <= 0 or self.membrane_tau_ms <= 0 or self.dt_ms > self.membrane_tau_ms:
            raise ValueError("dt_ms and tau must be positive, with dt_ms <= membrane_tau_ms")
        if self.sigmoid_slope_per_mv <= 0 or self.max_rate_hz <= 0:
            raise ValueError("rate parameters must be positive")
        if self.model_class != "rate_based":
            raise ValueError("this reference simulator implements only rate_based dynamics")

    @property
    def unit_count(self) -> int:
        return sum(size for _, size in self.population_sizes)


@dataclass(frozen=True, slots=True)
class NeuralStateSnapshot:
    """IF-18 simulator-neutral state; all neural quantities retain their physical units."""

    simulation_time_ms: float
    step_index: int
    dt_ms: float
    population_unit_indices: Mapping[str, tuple[int, ...]]
    membrane_potential_mv: np.ndarray | None
    firing_rate_hz: np.ndarray | None
    spike_events: tuple[Mapping[str, Any], ...]
    synaptic_current: np.ndarray | None
    synaptic_current_units: str
    connectivity_fingerprint: str
    simulator_version: str
    model_class: str
    numerical_status: str
    trace_ref: str | None = None

    def __post_init__(self) -> None:
        if self.simulation_time_ms < 0 or self.step_index < 0 or self.dt_ms <= 0:
            raise ValueError("neural snapshot time fields are invalid")
        for name in ("membrane_potential_mv", "firing_rate_hz", "synaptic_current"):
            object.__setattr__(self, name, _frozen_array(getattr(self, name), name))
        if self.membrane_potential_mv is not None and self.membrane_potential_mv.ndim != 1:
            raise ValueError("membrane potential must be one-dimensional")
        if self.firing_rate_hz is not None and self.firing_rate_hz.ndim != 1:
            raise ValueError("firing rate must be one-dimensional")
        if self.synaptic_current is not None and self.synaptic_current.ndim != 1:
            raise ValueError("synaptic current must be one-dimensional")
        _nonempty(self.synaptic_current_units, "synaptic_current_units")
        _nonempty(self.connectivity_fingerprint, "connectivity_fingerprint")
        _nonempty(self.simulator_version, "simulator_version")
        _nonempty(self.model_class, "model_class")
        _nonempty(self.numerical_status, "numerical_status")


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """IF-16 preregistration and split governance."""

    experiment_id: str
    hypothesis: str
    task: str
    stimulus_design: Mapping[str, Any]
    controlled_conditions: tuple[str, ...]
    baseline_models: tuple[str, ...]
    comparison_models: tuple[str, ...]
    preregistered_at_utc: str
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    seeds: tuple[int, ...]
    dependent_variables: tuple[str, ...]
    exclusions: tuple[str, ...]
    thresholds: Mapping[str, float]
    analysis_code_version: str
    intended_claim_level: str

    def __post_init__(self) -> None:
        for name in ("experiment_id", "hypothesis", "task", "preregistered_at_utc",
                     "analysis_code_version", "intended_claim_level"):
            _nonempty(getattr(self, name), name)
        if not self.seeds or any(seed < 0 for seed in self.seeds) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("experiment seeds must be non-empty, unique, non-negative integers")
        if set(self.train_groups) & set(self.validation_groups) or set(self.train_groups) & set(self.test_groups) or set(self.validation_groups) & set(self.test_groups):
            raise ValueError("train, validation and test groups must be disjoint")
        if not self.dependent_variables:
            raise ValueError("at least one dependent variable is required")
        if any(not np.isfinite(value) for value in self.thresholds.values()):
            raise ValueError("thresholds must be finite")


@dataclass(frozen=True, slots=True)
class MetricRecord:
    name: str
    value: float
    dataset_id: str
    split: str
    sample_size: int
    seeds: tuple[int, ...]
    confidence_interval: tuple[float, float] | None
    baseline_value: float | None
    code_fingerprint: str
    model_fingerprint: str
    analysis_version: str
    higher_is_better: bool = True
    independent_reference: bool = False

    def __post_init__(self) -> None:
        for name in ("name", "dataset_id", "split", "code_fingerprint", "model_fingerprint", "analysis_version"):
            _nonempty(getattr(self, name), name)
        if not np.isfinite(self.value) or self.sample_size < 1:
            raise ValueError("metric value must be finite and sample_size positive")
        if self.split not in {"TRAIN", "VALIDATION", "TEST", "EXTERNAL_REFERENCE"}:
            raise ValueError("invalid metric split")
        if self.confidence_interval is not None:
            lo, hi = self.confidence_interval
            if not np.isfinite([lo, hi]).all() or lo > hi:
                raise ValueError("confidence interval bounds are invalid")


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """IF-17 separate software, task and biological acceptance conclusions."""

    experiment_id: str
    software_status: AcceptanceStatus
    task_status: AcceptanceStatus
    robustness_status: AcceptanceStatus
    cost_status: AcceptanceStatus
    behavioral_correspondence_status: AcceptanceStatus
    neural_correspondence_status: AcceptanceStatus
    metrics: tuple[MetricRecord, ...]
    failures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.experiment_id, "experiment_id")
        for name in ("software_status", "task_status", "robustness_status", "cost_status",
                     "behavioral_correspondence_status", "neural_correspondence_status"):
            if not isinstance(getattr(self, name), AcceptanceStatus):
                object.__setattr__(self, name, AcceptanceStatus(getattr(self, name)))


def evaluate_experiment(
    spec: ExperimentSpec,
    *,
    software_checks: Mapping[str, bool] | None = None,
    task_metrics: Sequence[MetricRecord] = (),
    robustness_metrics: Sequence[MetricRecord] = (),
    cost_metrics: Sequence[MetricRecord] = (),
    behavioral_metrics: Sequence[MetricRecord] = (),
    neural_metrics: Sequence[MetricRecord] = (),
) -> EvaluationReport:
    """Evaluate independent gates; missing evidence stays PENDING/NOT_TESTED."""
    checks = dict(software_checks or {})
    software = (
        AcceptanceStatus.NOT_TESTED if not checks
        else AcceptanceStatus.PASS if all(checks.values())
        else AcceptanceStatus.FAILED
    )
    failures = [f"software check failed: {key}" for key, passed in checks.items() if not passed]

    def gate(records: Sequence[MetricRecord], require_independent: bool = False) -> AcceptanceStatus:
        if not records:
            return AcceptanceStatus.PENDING
        for metric in records:
            threshold = spec.thresholds.get(metric.name)
            if threshold is not None:
                passed = metric.value >= threshold if metric.higher_is_better else metric.value <= threshold
                if not passed:
                    failures.append(f"metric below threshold: {metric.name}")
                    return AcceptanceStatus.FAILED
            if require_independent and not metric.independent_reference:
                failures.append(f"non-independent biological reference: {metric.name}")
                return AcceptanceStatus.PENDING
        return AcceptanceStatus.PASS

    task = gate(task_metrics)
    robustness = gate(robustness_metrics)
    cost = gate(cost_metrics)
    behavioral = gate(behavioral_metrics, require_independent=True)
    neural = gate(neural_metrics, require_independent=True)
    all_metrics = tuple(task_metrics) + tuple(robustness_metrics) + tuple(cost_metrics) + tuple(behavioral_metrics) + tuple(neural_metrics)
    limitations = []
    if behavioral is not AcceptanceStatus.PASS or neural is not AcceptanceStatus.PASS:
        limitations.append("Biological correspondence remains unestablished without independent reference measurements.")
    if task is not AcceptanceStatus.PASS:
        limitations.append("Orthographic task validity remains pending or failed.")
    return EvaluationReport(
        spec.experiment_id, software, task, robustness, cost, behavioral, neural,
        all_metrics, tuple(failures), tuple(limitations),
    )


@dataclass(frozen=True, slots=True)
class NeuralSimulationConfigProtocol:
    """Marker alias to avoid binding downstream users to a simulator implementation."""


@runtime_checkable
class NeuralSimulator(Protocol):
    def initialize(self, config: NeuralSimulationConfig) -> NeuralStateSnapshot: ...
    def step(self, state: NeuralStateSnapshot, inputs: Sequence[NeuralInputEvent], dt_ms: float) -> NeuralStateSnapshot: ...
    def reset(self, state: NeuralStateSnapshot) -> NeuralStateSnapshot: ...


@runtime_checkable
class FunctionalVisualBackend(Protocol):
    def encode(self, observation: Any) -> VisualFeatureBundle: ...


@runtime_checkable
class BiologicalVisualBackend(Protocol):
    def process(self, observation: Any, simulator: NeuralSimulator) -> VisualFeatureBundle: ...


@runtime_checkable
class OrthographicBackend(Protocol):
    def represent(self, features: VisualFeatureBundle, evidence: Sequence[GraphemeEvidence], profile: Any) -> PositionCodeBundle: ...


@runtime_checkable
class RecurrentBackend(Protocol):
    def refine(self, state: Any, max_iterations: int) -> Any: ...


@runtime_checkable
class PlasticityBackend(Protocol):
    def create_child(self, parent: ModelBundleRef, request: LearningRequest) -> LearningResult: ...


@runtime_checkable
class PredictiveBackend(Protocol):
    def infer(self, evidence: Any, context: Any | None) -> PredictionState: ...


@runtime_checkable
class SequenceDecoder(Protocol):
    def decode(self, evidence: Any, representations: PositionCodeBundle) -> SequenceHypothesisSet: ...


@runtime_checkable
class BiologicalValidator(Protocol):
    def evaluate(self, spec: ExperimentSpec) -> EvaluationReport: ...


__all__ = [
    "FeatureTensor", "VisualFeatureBundle", "GraphemeEvidence",
    "PositionCodeChannel", "PositionCodeBundle", "OrthographicMemorySnapshot",
    "LearningRequest", "LearningResult", "PredictionState", "SequenceCandidate",
    "SequenceHypothesisSet", "ORERecognitionResult", "TraceEvent",
    "AcceptanceStatus", "NeuralInputEvent", "NeuralSimulationConfig",
    "NeuralStateSnapshot", "ExperimentSpec", "MetricRecord", "EvaluationReport",
    "evaluate_experiment", "NeuralSimulator", "FunctionalVisualBackend",
    "BiologicalVisualBackend", "OrthographicBackend", "RecurrentBackend",
    "PlasticityBackend", "PredictiveBackend", "SequenceDecoder",
    "BiologicalValidator",
]
