from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from ore_contracts import (
    AcceptanceStatus, BiologicalStateAdapter, CanonicalBackendRegistry, CanonicalOREController,
    ComputeBudget, ContractEnvelope, DatasetManifest, DatasetSample,
    EvidenceKind, ExperimentSpec, FeatureTensor, FrozenFusionPolicy,
    GraphemeEvidence, LearningRequest, LearningTransactionManager,
    LegacyN1Adapter, LegacyN3Adapter, LegacyN4Adapter, LegacyN5Adapter,
    LegacyN6Adapter, LegacyN9Adapter, MetricRecord, NeuralInputEvent, NeuralSimulationConfig,
    OperationMode, OrthographicMemorySnapshot, PositionCodeBundle, RateBasedB1PhaseBackend,
    PositionCodeChannel, RateBasedNeuralSimulator, ReadingDirection,
    RunContext, ScriptProfile, SequenceCandidate, SequenceHypothesisSet,
    TraceEvent, VisualFeatureBundle, evaluate_experiment, make_envelope,
    sha256_json, run_seeded_replicates,
)

HASH = "a" * 64


def envelope(contract_id, producer="test", **kwargs):
    return make_envelope(contract_id, "run-test", producer, config={"test": True}, **kwargs)


def test_if05_visual_feature_bundle_keeps_kind_units_and_source():
    item = FeatureTensor("v1", np.ones((4, 8, 8)), "ORIENTATION_Y_X", "float64",
                         "computational_activation", (8, 8), "legacy_visual")
    bundle = VisualFeatureBundle(envelope("IF-05"), (item,), ("region-1",), "legacy-test")
    assert bundle.feature_tensors[0].values.shape == (4, 8, 8)
    assert bundle.feature_tensors[0].representation_kind == "legacy_visual"
    assert not bundle.feature_tensors[0].values.flags.writeable


def test_if06_grapheme_evidence_validates_unknown_mass_and_distribution():
    item = GraphemeEvidence(
        envelope("IF-06"), "patch-1", None, "g1",
        {"a": 0.7, "<UNK>": 0.3}, 0.3, reliability=0.8,
    )
    assert item.probabilities["a"] == 0.7
    with pytest.raises(ValueError, match="sum to one"):
        GraphemeEvidence(envelope("IF-06"), "p", None, "g1", {"a": 0.8}, 0.0)


def test_if07_unavailable_position_channels_are_explicit_not_fabricated():
    channels = {
        key: PositionCodeChannel(key, None, "unavailable", "not_available",
                                  "test:no-position-encoding", False)
        for key in ("C", "A", "B", "O")
    }
    bundle = PositionCodeBundle(envelope("IF-07"), channels, logical_positions=(0, 1))
    assert all(not channel.available for channel in bundle.channels.values())
    assert bundle.logical_positions == (0, 1)


@dataclass
class FakeModel:
    version: int = 1
    fingerprint: str = "parent-fingerprint"
    parent_fingerprint: str | None = None

    def to_dict(self):
        return {"version": self.version, "parent_fingerprint": self.parent_fingerprint}


def make_learning_request(parent, authorized=True):
    return LearningRequest(
        parent_model_fingerprint=parent.fingerprint,
        training_sample_ids=("sample-1",),
        split_authorization="TRAIN_ONLY",
        verified_targets={"sample-1": "a"},
        verification_provenance={"sample-1": "review:1"},
        update_channels=("functional_parameters",),
        explicitly_authorized=authorized,
        policy_reference="policy://verified-labels",
        regression_suite_id="regression-v1",
    )


def test_if09_if15_learning_transaction_returns_detached_child_only_after_gates():
    parent = FakeModel()
    request = make_learning_request(parent)

    def propose(detached, _request):
        child = deepcopy(detached)
        child.version = 2
        child.parent_fingerprint = detached.fingerprint
        child.fingerprint = "child-fingerprint"
        return child

    outcome = LearningTransactionManager().execute(
        parent, request, propose,
        [("regression", lambda p, c: p.version == 1 and c.version == 2)],
    )
    assert outcome.result.decision == "ACCEPTED"
    assert outcome.child_model.version == 2
    assert parent.version == 1
    assert outcome.result.provenance["promotion"] == "NOT_PERFORMED"


def test_learning_requires_explicit_authorization():
    parent = FakeModel()
    request = make_learning_request(parent, authorized=False)
    outcome = LearningTransactionManager().execute(
        parent, request, lambda p, r: p, []
    )
    assert outcome.result.decision == "REJECTED"
    assert outcome.child_model is None


def test_learning_rolls_back_when_regression_gate_fails():
    parent = FakeModel()
    outcome = LearningTransactionManager().execute(
        parent, make_learning_request(parent),
        lambda p, r: FakeModel(2, "child-fingerprint", p.fingerprint),
        [("accuracy", lambda _p, _c: False)],
    )
    assert outcome.result.decision == "ROLLED_BACK"
    assert outcome.child_model is None
    assert parent.version == 1


def test_if10_prediction_state_keeps_prediction_and_observation_reference():
    from ore_contracts import PredictionState
    state = PredictionState(
        envelope("IF-10"), "graphemes", "patches",
        np.array([[0.8, 0.2]]), "observation://pixels",
        np.array([[0.1, -0.1]]), "posterior-minus-observation", "probability",
        np.array([1.0]), np.array([0.5]),
    )
    assert state.observed_evidence_ref == "observation://pixels"
    assert state.predicted_values.shape == (1, 2)


def test_if12_hypothesis_set_requires_candidates_or_abstention():
    candidate = SequenceCandidate(("a", "b"), ("p0", "p1"), {"S_vis": 0.8},
                                  {"S_vis": "visual channel only"}, 0.8)
    result = SequenceHypothesisSet(envelope("IF-12"), "s1", (candidate,))
    assert result.candidates[0].component_scores["S_vis"] == 0.8
    with pytest.raises(ValueError, match="explicit abstention"):
        SequenceHypothesisSet(envelope("IF-12"), "s2", ())


def test_if14_trace_events_are_indexed_and_timestamped():
    event = TraceEvent(0, "2026-11-14T10:00:00Z", "F-C", "encode",
                       ("obs-1",), ("features-1",), "EXECUTED")
    assert event.event_index == 0
    with pytest.raises(ValueError, match="event_index"):
        TraceEvent(-1, "2026-11-14T10:00:00Z", "F-C", "encode", (), (), "FAILED")


def test_if18_rate_simulator_has_deterministic_state_and_separate_units():
    config = NeuralSimulationConfig(
        population_sizes=(("v1", 2),),
        recurrent_weights=np.zeros((2, 2), dtype=np.float32),
        dt_ms=1.0, membrane_tau_ms=10.0,
    )
    simulator = RateBasedNeuralSimulator(config)
    state = simulator.initialize()
    event = NeuralInputEvent(0, 20.0, 0.5)
    next_state = simulator.step(state, [event], dt_ms=1.0)
    assert next_state.step_index == 1
    assert next_state.simulation_time_ms == 1.0
    assert next_state.membrane_potential_mv.shape == (2,)
    assert next_state.firing_rate_hz.shape == (2,)
    assert next_state.synaptic_current_units == "mV"
    assert next_state.spike_events == ()
    reset = simulator.reset(next_state)
    assert reset.step_index == 0
    assert reset.simulation_time_ms == 0.0


def test_if18_simulator_rejects_out_of_window_events():
    config = NeuralSimulationConfig((("p", 1),), np.zeros((1, 1)))
    simulator = RateBasedNeuralSimulator(config)
    with pytest.raises(ValueError, match="next simulation interval"):
        simulator.step(simulator.initialize(), [NeuralInputEvent(0, 1.0, 3.0)])


def test_if16_experiment_spec_rejects_group_leakage():
    with pytest.raises(ValueError, match="disjoint"):
        ExperimentSpec(
            "exp", "hypothesis", "recognition", {}, (), (), (),
            "2026-11-14T00:00:00Z", ("writer-1",), ("writer-2",), ("writer-1",),
            (1, 2), ("accuracy",), (), {"accuracy": 0.9}, "code-v1", "task",
        )


def test_if17_evaluation_keeps_software_task_and_biological_gates_separate():
    spec = ExperimentSpec(
        "exp-1", "visual prediction improves held-out accuracy", "OCR",
        {"stimuli": "synthetic"}, ("same-stimuli",), ("F-C/F-D/F-G",),
        ("B-N2/B-N3/B-N4",), "2026-11-14T00:00:00Z",
        ("writer-a",), ("writer-b",), ("writer-c",), (1, 2, 3),
        ("accuracy", "latency"), (), {"accuracy": 0.9, "latency_ms": 100.0},
        "analysis-v1", "TASK",
    )
    metrics = [
        MetricRecord("accuracy", 0.95, "ds", "TEST", 100, (1, 2, 3),
                     (0.92, 0.97), 0.90, "code", "model", "analysis"),
        MetricRecord("latency_ms", 50.0, "ds", "TEST", 100, (1, 2, 3),
                     (45.0, 60.0), 70.0, "code", "model", "analysis",
                     higher_is_better=False),
    ]
    report = evaluate_experiment(spec, software_checks={"contracts": True, "determinism": True},
                                 task_metrics=metrics, cost_metrics=(metrics[1],))
    assert report.software_status is AcceptanceStatus.PASS
    assert report.task_status is AcceptanceStatus.PASS
    assert report.cost_status is AcceptanceStatus.PASS
    assert report.behavioral_correspondence_status is AcceptanceStatus.PENDING
    assert report.neural_correspondence_status is AcceptanceStatus.PENDING


def test_legacy_n1_adapter_labels_its_origin_and_units():
    result = SimpleNamespace(
        stage_outputs={
            "normalized_energy": np.ones((4, 8, 8), dtype=np.float32),
            "feature_vector": np.ones(16, dtype=np.float32),
        },
        metadata={"pipeline": "legacy N1"},
    )
    bundle = LegacyN1Adapter.to_visual_feature_bundle(result, envelope("IF-05"))
    assert bundle.producer_kind == "legacy_ore_n1_adapter"
    assert {tensor.name for tensor in bundle.feature_tensors} == {"normalized_energy", "feature_vector"}


def test_legacy_n3_adapter_keeps_visual_probability_and_uncalibrated_status():
    result = SimpleNamespace(
        visual_probabilities=np.array([[0.7, 0.2, 0.1]], dtype=np.float32),
        class_names=("a", "b", "<UNK>"),
        candidates=[{"patch_id": "p0", "segmentation_confidence": 0.8}],
    )
    evidence = LegacyN3Adapter.to_grapheme_evidence(result, [envelope("IF-06")])
    assert evidence[0].unknown_probability == pytest.approx(0.1)
    assert evidence[0].calibration_status == "UNCALIBRATED"


def test_legacy_n4_adapter_does_not_forge_position_encodings():
    bundle = LegacyN4Adapter.position_code_bundle(envelope("IF-07"), patch_count=2)
    assert all(not channel.available for channel in bundle.channels.values())


def test_legacy_n5_adapter_separates_functional_memory_namespace():
    model = SimpleNamespace(
        class_names=("a", "b"),
        fast_forward_counts=np.array([[2.0, 1.0], [1.0, 3.0]]),
        slow_forward_counts=np.array([[1.0, 1.0], [1.0, 1.0]]),
        exposure_counts=np.array([3.0, 4.0]),
        to_dict=lambda: {},
        fingerprint="legacy-model",
        version=2,
    )
    snapshot = LegacyN5Adapter.to_memory_snapshot(model, envelope("IF-08"))
    assert snapshot.memory_namespace.startswith("legacy.")
    assert snapshot.transition_probabilities.shape == (2, 2)
    assert snapshot.exposure_counts == {"a": 3, "b": 4}


def test_legacy_n6_adapter_exports_sequence_hypotheses():
    result = SimpleNamespace(
        sequence_id="s1", class_names=("a", "b"),
        decoded_probabilities=np.array([[0.9, 0.1], [0.2, 0.8]]),
        decoded_graphemes=("a", "b"), diagnostics={"abstained": False},
    )
    hypotheses = LegacyN6Adapter.to_hypotheses(result, envelope("IF-12"))
    assert hypotheses.candidates[0].graphemes == ("a", "b")
    assert hypotheses.candidates[0].component_scores["S_ortho"] == pytest.approx(0.85)


class FakeFunctional:
    def run(self, payload, context):
        return {"engine": "functional", "payload": payload}


class FakeBiological:
    def run(self, invocation):
        return {"phase": invocation.phase_id, "authorized": invocation.learning_authorized}


def make_context(mode):
    return RunContext("run-controller", mode, 7, ComputeBudget(5.0, 4, 128.0))


def test_p8_functional_biological_hybrid_modes_keep_branches_separate():
    registry = CanonicalBackendRegistry()
    registry.register_functional(FakeFunctional())
    registry.register_biological("B-N1", FakeBiological())
    controller = CanonicalOREController(registry)
    result = controller.run(make_context(OperationMode.HYBRID), {"image": "immutable"})
    assert result.functional_output["engine"] == "functional"
    assert result.biological_outputs["B-N1"]["phase"] == "B-N1"
    assert result.fused_output is None
    assert result.channel_status["FUSION"] == "NOT_RUN_NO_FROZEN_VALIDATION_POLICY"
    assert result.diagnostics["valid_trace_order"] is True
    assert result.acceptance_status["VAL-BIOLOGICAL"] == "PENDING"


def test_p8_fusion_requires_fitted_and_frozen_policy_and_keeps_components():
    registry = CanonicalBackendRegistry()
    registry.register_functional(FakeFunctional())
    registry.register_biological("B-N1", FakeBiological())
    policy = FrozenFusionPolicy(
        "combiner-v1", "manifest://validation", HASH,
        lambda functional, biological, context: {"score": 0.5},
    )
    registry.register_fusion_policy(policy)
    result = CanonicalOREController(registry).run(
        make_context(OperationMode.HYBRID), {"input": 1}
    )
    assert result.fused_output["component_outputs_preserved"] is True
    assert result.fused_output["policy_id"] == "combiner-v1"
    bad = CanonicalBackendRegistry()
    with pytest.raises(ValueError, match="validation-fitted"):
        bad.register_fusion_policy(FrozenFusionPolicy(
            "bad", "manifest", HASH, lambda *args: {}, validation_fitted=False
        ))


def test_p8_biological_mode_reports_missing_phases_instead_of_inventing_them():
    registry = CanonicalBackendRegistry()
    registry.register_biological("B-N1", FakeBiological())
    result = CanonicalOREController(registry).run(
        make_context(OperationMode.BIOLOGICAL), {"stimulus": "x"}
    )
    assert result.channel_status["B-N1"] == "EXECUTED"
    assert result.channel_status["B-N2"] == "CHANNEL_UNAVAILABLE"
    assert result.biological_outputs["B-N2"]["status"] == "CHANNEL_UNAVAILABLE"
    assert result.channel_status["B-N9"] == "EXECUTED"


def test_p8_validation_runs_only_the_offline_b_n8_backend():
    class Validator:
        def run(self, invocation):
            assert invocation.phase_id == "B-N8"
            assert invocation.validation_spec.experiment_id == "exp"
            return {"status": "NOT_TESTED", "independent_reference": False}

    registry = CanonicalBackendRegistry()
    registry.register_biological("B-N8", Validator())
    spec = ExperimentSpec(
        "exp", "hypothesis", "validation", {}, (), (), (), "2026-11-14T00:00:00Z",
        ("a",), ("b",), ("c",), (1,), ("accuracy",), (), {}, "code", "BIOLOGICAL",
    )
    result = CanonicalOREController(registry).run(
        make_context(OperationMode.VALIDATE), {"saved_trace": "ref"}, validation_spec=spec
    )
    assert result.channel_status["B-N8"] == "EXECUTED"
    assert "B-N1" not in result.channel_status


def test_biological_state_adapter_keeps_voltage_and_rate_units_separate():
    config = NeuralSimulationConfig((("v1", 1),), np.zeros((1, 1)))
    state = RateBasedNeuralSimulator(config).initialize()
    bundle = BiologicalStateAdapter.to_visual_feature_bundle(state, envelope("IF-05"))
    by_name = {tensor.name: tensor for tensor in bundle.feature_tensors}
    assert by_name["membrane_potential"].units == "mV"
    assert by_name["firing_rate"].units == "Hz"
    assert by_name["membrane_potential"].representation_kind != by_name["firing_rate"].representation_kind


def test_fixed_seed_experiment_runner_records_each_replicate():
    spec = ExperimentSpec(
        "replicate-exp", "hypothesis", "task", {}, (), (), (),
        "2026-11-14T00:00:00Z", ("train",), ("validation",), ("test",),
        (3, 7, 11), ("accuracy",), (), {}, "analysis-v1", "TASK",
    )
    report = run_seeded_replicates(spec, lambda current, seed: {"accuracy": seed / 20.0})
    assert report.successful_seeds == (3, 7, 11)
    assert report.failed_seeds == ()
    assert report.replicates[1].metrics["accuracy"] == pytest.approx(0.35)


def test_fixed_seed_experiment_runner_records_failures_without_fake_metrics():
    spec = ExperimentSpec(
        "replicate-fail", "hypothesis", "task", {}, (), (), (),
        "2026-11-14T00:00:00Z", ("train",), ("validation",), ("test",),
        (1, 2), ("accuracy",), (), {}, "analysis-v1", "TASK",
    )
    report = run_seeded_replicates(
        spec, lambda _spec, seed: (_ for _ in ()).throw(RuntimeError("failed")) if seed == 2 else {"accuracy": 0.9}
    )
    assert report.replicates[0].status == "COMPLETED"
    assert report.replicates[1].status == "FAILED"
    assert report.replicates[1].metrics == {}


def test_legacy_n9_adapter_keeps_patch_level_candidates_and_legacy_provenance():
    result = SimpleNamespace(
        sequence_id="legacy-seq",
        decision={"status": "CANDIDATE", "confidence": 0.93},
        class_names=("a", "b"),
        visual_probabilities=np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
        posterior_beliefs=np.array([[0.95, 0.05], [0.1, 0.9]], dtype=np.float32),
    )
    adapted = LegacyN9Adapter.to_recognition_result(result, envelope("IF-13"))
    assert adapted.status == "RESOLVED"
    assert adapted.hypotheses.candidates[0].graphemes == ("a", "b")
    assert "canonical B-N9" in adapted.limitations[0]


def test_rate_based_backend_runs_canonical_b_n1_via_controller():
    config = NeuralSimulationConfig((("v1", 1),), np.zeros((1, 1)), dt_ms=1.0, membrane_tau_ms=10.0)
    registry = CanonicalBackendRegistry()
    registry.register_biological("B-N1", RateBasedB1PhaseBackend(config))
    payload = {
        "simulation_steps": 2,
        "neural_inputs": [NeuralInputEvent(0, 5.0, 1.5)],
    }
    result = CanonicalOREController(registry).run(
        make_context(OperationMode.BIOLOGICAL), payload
    )
    state = result.biological_outputs["B-N1"]
    assert state.step_index == 2
    assert state.simulation_time_ms == 2.0
    assert result.channel_status["B-N1"] == "EXECUTED"
