from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pytest

from ore_n6 import N6_ALGORITHM_STEPS, N6Config, N6Input, N6LexiconEntry, N6Processor


@dataclass
class FakeN5Model:
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
    version: int = 2
    schema_version: str = "1.0"
    fingerprint: str = ""

    def to_dict(self, *, include_fingerprint: bool = True):
        names = (
            "feature_mean", "feature_scale", "fast_prototypes", "slow_prototypes",
            "fast_eligibility", "slow_eligibility", "exposure_counts", "class_gains",
            "fast_unigram_counts", "slow_unigram_counts", "fast_forward_counts",
            "slow_forward_counts", "fast_reverse_counts", "slow_reverse_counts",
        )
        payload = {
            "schema": "ore.n5.plasticity_model.v1",
            "schema_version": self.schema_version,
            "class_names": list(self.class_names),
            **{name: np.asarray(getattr(self, name)).tolist() for name in names},
            "replay_buffer": [],
            "version": self.version,
            "parent_fingerprint": None,
            "visual_temperature": 0.18,
            "unknown_similarity_threshold": 0.18,
            "unknown_temperature": 0.08,
            "transition_alpha": 0.5,
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload


def fingerprint(model) -> str:
    content = json.dumps(
        model.to_dict(include_fingerprint=False), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def make_n5(classes=("a", "b", "c")):
    c = len(classes)
    d = 4
    fast = np.eye(c, d, dtype=np.float32)
    slow = fast.copy()
    outputs = c + 1
    transitions = np.full((outputs, outputs), 0.2, dtype=np.float64)
    transitions[:c, :c] += np.eye(c) * 5.0
    if c > 1:
        transitions[0, 1] += 12.0
        transitions[1, 0] += 12.0
    unigram = np.ones(outputs, dtype=np.float64)
    unigram[:c] += np.arange(c, 0, -1) * 3.0
    model = FakeN5Model(
        class_names=tuple(classes),
        feature_mean=np.zeros(d, np.float32),
        feature_scale=np.ones(d, np.float32),
        fast_prototypes=fast,
        slow_prototypes=slow,
        fast_eligibility=np.zeros_like(fast),
        slow_eligibility=np.zeros_like(slow),
        exposure_counts=np.ones(c, np.float64),
        class_gains=np.ones(c, np.float64),
        fast_unigram_counts=unigram.copy(),
        slow_unigram_counts=unigram.copy(),
        fast_forward_counts=transitions.copy(),
        slow_forward_counts=transitions.copy(),
        fast_reverse_counts=transitions.copy(),
        slow_reverse_counts=transitions.copy(),
    )
    model.fingerprint = fingerprint(model)
    return model


def make_n4(classes=("a", "b", "c")):
    rows = np.asarray([
        [0.93, 0.04, 0.03],
        [0.04, 0.92, 0.04],
        [0.90, 0.06, 0.04],
        [0.06, 0.90, 0.04],
    ], dtype=np.float32)
    if len(classes) != 3:
        rows = np.tile(np.full(len(classes), 1.0 / len(classes)), (4, 1))
    return type("N4Fixture", (), {
        "sequence_id": "sequence-1",
        "class_names": tuple(classes),
        "posterior_beliefs": rows,
        "attention_allocation": np.full(4, 0.25, dtype=np.float32),
        "sensory_precision": np.ones(4, dtype=np.float32),
        "diagnostics": {"valid": True},
    })()


def make_entries():
    return [
        N6LexiconEntry("word-ab", ("a", "b"), 20.0, "reviewer-1", "dataset://words/ab"),
        N6LexiconEntry("word-ba", ("b", "a"), 12.0, "reviewer-1", "dataset://words/ba"),
        N6LexiconEntry("word-a", ("a",), 4.0, "reviewer-1", "dataset://words/a"),
        N6LexiconEntry("word-b", ("b",), 4.0, "reviewer-1", "dataset://words/b"),
    ]


def test_exactly_60_unique_named_operations_are_traceable():
    assert len(N6_ALGORITHM_STEPS) == 60
    assert len(set(N6_ALGORITHM_STEPS)) == 60
    result = N6Processor().process(N6Input(make_n4(), make_n5(), make_entries()))
    assert result.diagnostics["valid"] is True
    assert result.diagnostics["stage_count"] == 60
    assert result.diagnostics["executed_operation_count"] == 60
    assert len(result.operation_trace) == 60
    assert [item["step"] for item in result.operation_trace] == list(range(1, 61))
    assert result.diagnostics["operation_trace_complete"] is True


def test_lexical_segmentation_covers_sequence_and_preserves_unknown_fallback():
    result = N6Processor().process(N6Input(make_n4(), make_n5(), make_entries()))
    assert result.tokens
    assert result.tokens[0]["start"] == 0
    assert result.tokens[-1]["end"] == 4
    assert all(token["end"] > token["start"] for token in result.tokens)
    assert result.diagnostics["lexical_coverage"] > 0.0
    assert len(result.segmentation_alternatives) >= 1
    assert np.isclose(sum(x["probability"] for x in result.segmentation_alternatives), 1.0)


def test_no_lexicon_uses_non_destructive_unknown_grapheme_fallback():
    result = N6Processor().process(N6Input(make_n4(), make_n5()))
    assert len(result.tokens) == 4
    assert all(token["status"] == "UNLEXICALIZED_GRAPHEME" for token in result.tokens)
    assert result.diagnostics["lexical_coverage"] == 0.0


def test_n5_parent_is_read_only_and_fingerprint_is_retained():
    model = make_n5()
    before = model.to_dict()
    result = N6Processor().process(N6Input(make_n4(), model, make_entries()))
    assert model.to_dict() == before
    assert result.metadata["n5_parent_fingerprint"] == model.fingerprint
    assert result.diagnostics["N5_parent_unchanged"] is True


def test_explicit_boundary_probabilities_are_validated_and_preserved():
    boundaries = np.asarray([0.2, 0.9, 0.15, 0.85, 0.3], dtype=np.float32)
    result = N6Processor().process(N6Input(make_n4(), make_n5(), make_entries(), boundaries))
    assert result.metadata["boundary_source"] == "caller_supplied"
    assert result.boundary_probabilities[0] == 1.0
    assert result.boundary_probabilities[-1] == 1.0
    assert result.boundary_probabilities[1] == pytest.approx(0.9)


def test_bad_n5_fingerprint_and_vocab_mismatch_are_rejected():
    model = make_n5()
    model.fingerprint = "bad"
    with pytest.raises(ValueError, match="fingerprint"):
        N6Processor().process(N6Input(make_n4(), model))
    with pytest.raises(ValueError, match="vocabularies"):
        N6Processor().process(N6Input(make_n4(("a", "b", "x")), make_n5()))


def test_invalid_lexicon_provenance_and_graphemes_are_rejected():
    bad = N6LexiconEntry("bad", ("a", "b"), 1.0, "", "dataset://bad")
    with pytest.raises(ValueError, match="verified_by"):
        N6Processor().process(N6Input(make_n4(), make_n5(), [bad]))
    bad_label = N6LexiconEntry("bad-label", ("x",), 1.0, "reviewer", "dataset://bad")
    with pytest.raises(ValueError, match="unknown graphemes"):
        N6Processor().process(N6Input(make_n4(), make_n5(), [bad_label]))


def test_non_normalized_and_nonfinite_posteriors_are_rejected_or_normalized_safely():
    n4 = make_n4()
    n4.posterior_beliefs = n4.posterior_beliefs * 0.5
    with pytest.raises(ValueError, match="probability"):
        N6Processor().process(N6Input(n4, make_n5()))
    n4 = make_n4()
    n4.posterior_beliefs[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        N6Processor().process(N6Input(n4, make_n5()))


def test_config_validation_and_json_serialization():
    with pytest.raises(ValueError):
        N6Config(transition_smoothing=0)
    with pytest.raises(ValueError):
        N6Config(top_k_segmentations=0)
    result = N6Processor().process(N6Input(make_n4(), make_n5(), make_entries()))
    encoded = json.dumps(result.to_dict(), allow_nan=False)
    assert "ore.n6.hierarchical_orthographic_access.v1" in encoded


def test_optional_n2_features_activate_fast_slow_n5_prototype_evidence():
    model = make_n5()
    features = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float32)
    before = model.to_dict()
    result = N6Processor().process(N6Input(
        make_n4(), model, make_entries(), n2_features=features
    ))
    assert result.metadata["n5_prototype_evidence_used"] is True
    assert model.to_dict() == before
    assert result.diagnostics["valid"] is True


def test_bad_n2_feature_shape_and_zero_activity_are_rejected():
    with pytest.raises(ValueError, match="n2_features must have shape"):
        N6Processor().process(N6Input(
            make_n4(), make_n5(), n2_features=np.ones((4, 3), np.float32)
        ))
    with pytest.raises(ValueError, match="non-zero standardized activity"):
        N6Processor().process(N6Input(
            make_n4(), make_n5(), n2_features=np.zeros((4, 4), np.float32)
        ))


def test_single_class_sequence_is_supported():
    result = N6Processor().process(N6Input(
        make_n4(("a",)), make_n5(("a",)),
        [N6LexiconEntry("only-a", ("a",), 1.0, "reviewer", "dataset://a")],
    ))
    assert result.decoded_graphemes == ("a", "a", "a", "a")
    assert result.diagnostics["valid"] is True
