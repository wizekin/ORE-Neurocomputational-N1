from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ore_n4 import N4_ALGORITHM_STEPS, N4Config, N4Input, N4Processor


def make_n3_result() -> SimpleNamespace:
    visual = np.asarray([
        [0.80, 0.15, 0.05],
        [0.42, 0.48, 0.10],
        [0.08, 0.17, 0.75],
    ], dtype=np.float32)
    prior = np.asarray([
        [0.65, 0.25, 0.10],
        [0.30, 0.55, 0.15],
        [0.10, 0.25, 0.65],
    ], dtype=np.float32)
    posterior = 0.7 * visual + 0.3 * prior
    posterior /= posterior.sum(axis=1, keepdims=True)
    error = posterior * (np.log(posterior) - np.log(prior))
    return SimpleNamespace(
        sequence_id="test-sequence",
        class_names=("a", "b", "c"),
        visual_probabilities=visual,
        contextual_priors=prior,
        posterior_beliefs=posterior.astype(np.float32),
        classwise_prediction_error=error.astype(np.float32),
        candidates=[
            {"patch_id": "p1", "segmentation_confidence": 0.95, "attention": 0.1},
            {"patch_id": "p2", "segmentation_confidence": 0.55, "attention": 0.2},
            {"patch_id": "p3", "segmentation_confidence": 0.85, "attention": 0.0},
        ],
        diagnostics={"valid": True},
        metadata={"pipeline": "test N3"},
    )


def test_algorithm_has_exactly_40_unique_named_operations() -> None:
    assert len(N4_ALGORITHM_STEPS) == 40
    assert len(set(N4_ALGORITHM_STEPS)) == 40


def test_basic_n4_pipeline_returns_valid_normalized_hypotheses() -> None:
    n3 = make_n3_result()
    original_posterior = n3.posterior_beliefs.copy()
    result = N4Processor().process(N4Input(n3))

    assert result.diagnostics["valid"] is True
    assert result.diagnostics["stage_count"] == 40
    assert result.diagnostics["executed_operation_count"] == 40
    assert len(result.operation_trace) == 40
    assert result.posterior_beliefs.shape == (3, 3)
    assert np.allclose(result.posterior_beliefs.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(result.attention_allocation >= 0)
    assert np.isclose(result.attention_allocation.sum(), 1.0)
    assert np.isfinite(result.sensory_precision).all()
    assert np.isfinite(result.prior_precision).all()
    assert result.class_attention_gate.shape == (3, 3)
    assert [item["step"] for item in result.operation_trace] == list(range(1, 41))
    assert len(result.candidates) == 3
    assert np.array_equal(n3.posterior_beliefs, original_posterior)


def test_external_cues_change_attention_without_changing_visual_input() -> None:
    n3 = make_n3_result()
    plain = N4Processor().process(N4Input(n3))
    cued = N4Processor().process(N4Input(
        n3,
        bottom_up_salience=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        task_relevance=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        goal_class_weights=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    ))
    assert np.array_equal(plain.visual_probabilities, cued.visual_probabilities)
    assert cued.attention_allocation[2] > plain.attention_allocation[2]
    assert cued.top_down_relevance[2] >= plain.top_down_relevance[2]


def test_attention_ablation_is_neutral_over_positions() -> None:
    n3 = make_n3_result()
    config = N4Config(enable_spatial_attention=False, seed_attention_weight=0.0, spatial_smoothing=0.0)
    result = N4Processor(config).process(N4Input(n3))
    assert np.allclose(result.attention_allocation, np.full(3, 1 / 3), atol=1e-6)


def test_rejects_bad_probability_rows_and_bad_cue_shapes() -> None:
    n3 = make_n3_result()
    n3.visual_probabilities = n3.visual_probabilities.copy()
    n3.visual_probabilities[0] = 0.0
    with pytest.raises(ValueError, match="all-zero"):
        N4Processor().process(N4Input(n3))

    with pytest.raises(ValueError, match="task_relevance"):
        N4Processor().process(N4Input(make_n3_result(), task_relevance=np.asarray([0.5])))


def test_result_serializes_as_json() -> None:
    result = N4Processor().process(N4Input(make_n3_result()))
    encoded = json.dumps(result.to_dict(), allow_nan=False)
    assert '"schema": "ore.n4.attention_precision_gating.v1"' in encoded


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        N4Config(attention_temperature=0)
    with pytest.raises(ValueError):
        N4Config(damping=1.5)
    with pytest.raises(ValueError, match="internal bottom-up"):
        N4Config(
            entropy_salience_weight=0,
            error_salience_weight=0,
            surprise_salience_weight=0,
            external_salience_weight=1,
        )
