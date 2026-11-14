from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from ore_n3 import LabeledN2Patch, N3Config, N3Patch, N3Processor, N3Sequence
from ore_n4 import N4Input, N4Processor
from ore_n5.stages import clip_prototype_drift
from ore_n5 import (
    N5_ALGORITHM_STEPS,
    N5Config,
    N5Experience,
    N5LearningRequest,
    N5ModelState,
    N5Processor,
)


def make_n3_processor() -> N3Processor:
    references = [
        LabeledN2Patch("a", features=np.asarray([1.0, 0.1, 0.0, 0.0], np.float32)),
        LabeledN2Patch("a", features=np.asarray([0.9, 0.2, 0.0, 0.1], np.float32)),
        LabeledN2Patch("b", features=np.asarray([0.0, 0.1, 1.0, 0.0], np.float32)),
        LabeledN2Patch("b", features=np.asarray([0.1, 0.0, 0.9, 0.2], np.float32)),
    ]
    return N3Processor(N3Config()).fit(
        references, grapheme_sequences=[("a", "b"), ("b", "a")]
    )


def make_experiences() -> list[N5Experience]:
    return [
        N5Experience(
            experience_id="exp-1",
            patch_id="p1",
            sequence_id="s1",
            grapheme="a",
            features=np.asarray([0.95, 0.15, 0.02, 0.02], np.float32),
            verified_by="independent-reviewer",
            source_reference="annotation://sample/1",
            verified_at="2026-11-14T05:00:00Z",
            confidence=0.95,
            sequence_position=0,
        ),
        N5Experience(
            experience_id="exp-2",
            patch_id="p2",
            sequence_id="s1",
            grapheme="b",
            features=np.asarray([0.02, 0.08, 0.95, 0.10], np.float32),
            verified_by="independent-reviewer",
            source_reference="annotation://sample/2",
            verified_at="2026-11-14T05:00:00Z",
            confidence=0.90,
            sequence_position=1,
        ),
    ]


def make_n4_result(n3: N3Processor):
    n3_result = n3.recognize(N3Sequence("s1", [
        N3Patch("p1", features=np.asarray([0.95, 0.15, 0.02, 0.02], np.float32)),
        N3Patch("p2", features=np.asarray([0.02, 0.08, 0.95, 0.10], np.float32)),
    ]))
    return N4Processor().process(N4Input(n3_result))


def make_request(
    model: N5ModelState, *, n4_result=None, **overrides
) -> N5LearningRequest:
    values = {
        "parent_model": model,
        "experiences": make_experiences(),
        "independently_verified": True,
        "user_authorized": True,
        "n4_result": n4_result,
        "verified_sequences": [("a", "b")],
    }
    values.update(overrides)
    return N5LearningRequest(**values)


def test_algorithm_has_exactly_52_unique_named_operations():
    assert len(N5_ALGORITHM_STEPS) == 52
    assert len(set(N5_ALGORITHM_STEPS)) == 52


def test_learning_returns_child_and_preserves_parent_state():
    parent = N5Processor.initialize(make_n3_processor())
    before = parent.to_dict()
    result = N5Processor().learn(make_request(parent))

    assert result.diagnostics["valid"] is True
    assert result.diagnostics["stage_count"] == 52
    assert result.diagnostics["executed_operation_count"] == 52
    assert len(result.operation_trace) == 52
    assert [item["step"] for item in result.operation_trace] == list(range(1, 53))
    assert result.child_model.version == parent.version + 1
    assert result.child_model.parent_fingerprint == parent.fingerprint
    assert result.child_model.fingerprint != parent.fingerprint
    assert parent.to_dict() == before
    assert np.allclose(np.linalg.norm(result.child_model.fast_prototypes, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(result.child_model.slow_prototypes, axis=1), 1.0)
    assert result.metrics["total_exposure_increment"] > 0


def test_user_authorization_and_independent_verification_are_mandatory():
    parent = N5Processor.initialize(make_n3_processor())
    with pytest.raises(PermissionError, match="user_authorized"):
        N5Processor().learn(make_request(parent, user_authorized=False))
    with pytest.raises(PermissionError, match="independently_verified"):
        N5Processor().learn(make_request(parent, independently_verified=False))


def test_n4_attention_precision_and_class_gates_are_integrated():
    n3 = make_n3_processor()
    parent = N5Processor.initialize(n3)
    n4 = make_n4_result(n3)
    result = N5Processor().learn(make_request(parent, n4_result=n4))
    assert result.diagnostics["n4_modulation_source"].startswith("N4")
    assert result.metrics["experience_count"] == 2
    assert result.diagnostics["valid"] is True


def test_provenance_and_optional_feature_digest_are_checked():
    parent = N5Processor.initialize(make_n3_processor())
    invalid = make_experiences()
    invalid[0].verified_by = ""
    with pytest.raises(ValueError, match="verified_by"):
        N5Processor().learn(make_request(parent, experiences=invalid))

    invalid = make_experiences()
    invalid[0].feature_sha256 = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        N5Processor().learn(make_request(parent, experiences=invalid))

    valid = make_experiences()
    vector = np.ascontiguousarray(valid[0].features.astype(np.float32))
    valid[0].feature_sha256 = hashlib.sha256(vector.tobytes()).hexdigest()
    result = N5Processor().learn(make_request(parent, experiences=valid))
    assert result.diagnostics["valid"] is True


def test_unknown_labels_and_bad_feature_dimensions_are_rejected():
    parent = N5Processor.initialize(make_n3_processor())
    invalid = make_experiences()
    invalid[0].grapheme = "z"
    with pytest.raises(ValueError, match="unknown"):
        N5Processor().learn(make_request(parent, experiences=invalid))

    invalid = make_experiences()
    invalid[0].features = np.ones(3, np.float32)
    with pytest.raises(ValueError, match="length"):
        N5Processor().learn(make_request(parent, experiences=invalid))


def test_model_bundle_round_trip_and_export_to_n3():
    parent = N5Processor.initialize(make_n3_processor())
    result = N5Processor().learn(make_request(parent))
    restored = N5ModelState.from_dict(result.child_model.to_dict())
    assert restored.fingerprint == result.child_model.fingerprint
    exported = restored.to_n3_processor()
    assert exported._fitted is True
    assert tuple(exported.class_names) == (*restored.class_names, "<UNK>")
    assert np.isclose(exported.transition_model.unigram.sum(), 1.0)
    assert np.allclose(exported.transition_model.forward.sum(axis=1), 1.0)
    assert np.allclose(exported.transition_model.reverse.sum(axis=0), 1.0)


def test_replay_capacity_and_fast_slow_banks_are_independent():
    config = N5Config(replay_capacity=3, replay_max_samples=2)
    parent = N5Processor.initialize(make_n3_processor(), config)
    first = N5Processor(config).learn(make_request(parent))
    second_examples = make_experiences()
    for index, item in enumerate(second_examples):
        item.experience_id = f"second-{index}"
        item.patch_id = f"second-p{index}"
    second = N5Processor(config).learn(make_request(
        first.child_model, experiences=second_examples
    ))
    assert len(second.child_model.replay_buffer) == 3
    assert second.metrics["replay_count"] == 2
    assert not np.array_equal(
        second.child_model.fast_prototypes, second.child_model.slow_prototypes
    )


def test_serialization_is_finite_and_json_compatible():
    parent = N5Processor.initialize(make_n3_processor())
    result = N5Processor().learn(make_request(parent))
    encoded = json.dumps(result.to_dict(), allow_nan=False)
    assert "ore.n5.experience_dependent_plasticity.v1" in encoded


def test_config_validation():
    with pytest.raises(ValueError):
        N5Config(fast_learning_rate=0)
    with pytest.raises(ValueError):
        N5Config(replay_capacity=1, replay_max_samples=2)
    with pytest.raises(ValueError):
        N5Config(max_prototype_drift=3.0)


def test_parent_relative_prototype_drift_clipping_obeys_final_chord_bound():
    parent = np.asarray([1.0, 0.0], dtype=np.float32)
    candidate = np.asarray([0.6, 0.8], dtype=np.float32)
    maximum = 0.35
    clipped = clip_prototype_drift(candidate, parent, maximum, 1e-8)
    assert np.isclose(np.linalg.norm(clipped), 1.0, atol=1e-6)
    assert np.linalg.norm(clipped - parent) <= maximum + 1e-6
