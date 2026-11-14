import numpy as np
import pytest
from ore_n3 import (
    LabeledN2Patch, N3Config, N3Patch, N3Processor, N3Sequence,
    N3_ALGORITHM_STEPS, UNKNOWN_LABEL,
)

def fake_n2_result(seed: int, family: int):
    rng = np.random.default_rng(seed)
    h = w = 12
    base = rng.normal(0.0, 0.025, size=(h, w)).astype(np.float32)
    base += 0.20 * family
    ventral = np.stack([
        base + 0.12 * family,
        base * 0.5 + 0.03,
        np.abs(base),
        np.flip(base, axis=0),
    ]).astype(np.float32)
    representations = {
        "ventral": ventral,
        "contour": np.abs(base + 0.10 * family),
        "boundary": np.abs(np.roll(base, 1, axis=0)),
        "junction": np.abs(base * (family + 1)),
        "curvature": np.abs(np.roll(base, 1, axis=1) - base),
        "crowding": np.clip(np.abs(base), 0, 1),
        "figure_probability": 1.0 / (1.0 + np.exp(-base)),
        "invariant": np.abs(base) + family * 0.05,
        "dorsal": np.abs(np.gradient(base)[0]),
        "shape_fragments": (base > np.median(base)).astype(np.int32),
    }
    structured = {"shape_graph": {
        "node_count": family + 1, "edge_count": family,
        "beta0": 1, "beta1": 0, "euler_characteristic": 1,
    }}
    return {"representations": representations, "structured_output": structured}

def training_engine(config=None):
    samples = [
        LabeledN2Patch("a", n2_result=fake_n2_result(1, 0)),
        LabeledN2Patch("a", n2_result=fake_n2_result(2, 0)),
        LabeledN2Patch("b", n2_result=fake_n2_result(3, 1)),
        LabeledN2Patch("b", n2_result=fake_n2_result(4, 1)),
    ]
    return N3Processor(config).fit(
        samples, grapheme_sequences=[("a", "b", "a"), ("b", "b", "a")]
    )

def test_n3_runs_29_operations_and_preserves_uncertainty():
    engine = training_engine()
    query = N3Sequence("sample", [
        N3Patch("p1", n2_result=fake_n2_result(10, 0), segmentation_confidence=0.9),
        N3Patch("p2", n2_result=fake_n2_result(11, 1), segmentation_confidence=0.8),
        N3Patch("p3", n2_result=fake_n2_result(12, 0), segmentation_confidence=0.95),
    ])
    result = engine.process(query)
    assert result.diagnostics["valid"]
    assert result.diagnostics["stage_count"] == 29
    assert len(N3_ALGORITHM_STEPS) == 29
    assert len(result.trace) <= engine.config.max_iterations
    assert result.posterior_beliefs.shape == (3, 3)
    assert result.class_names[-1] == UNKNOWN_LABEL
    np.testing.assert_allclose(result.posterior_beliefs.sum(axis=1), 1.0, atol=1e-5)
    assert result.feature_residuals.shape == (3, 68)
    assert len(result.candidates) == 3
    assert result.to_dict()["schema"] == "ore.n3.recurrent_orthographic_prediction.v1"

def test_transition_matrices_are_normalized_in_correct_directions():
    engine = training_engine()
    model = engine.transition_model
    assert model is not None
    np.testing.assert_allclose(model.unigram.sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(model.forward.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(model.reverse.sum(axis=0), 1.0, atol=1e-6)

def test_attention_increases_sensory_precision():
    from ore_n3.stages import precision_from_reliability
    low = precision_from_reliability(
        np.array([0.8], np.float32), np.array([0.0], np.float32), 1.0, 1.0
    )
    high = precision_from_reliability(
        np.array([0.8], np.float32), np.array([1.0], np.float32), 1.0, 1.0
    )
    assert high[0] > low[0]

def test_n3_rejects_nonfinite_or_out_of_range_reliability():
    engine = training_engine()
    bad = N3Sequence("bad", [
        N3Patch("p1", features=np.zeros(68, np.float32), segmentation_confidence=1.5)
    ])
    with pytest.raises(ValueError, match="segmentation_confidence"):
        engine.process(bad)

def test_n3_rejects_incompatible_feature_dimension():
    engine = training_engine()
    bad = N3Sequence("bad", [N3Patch("p1", features=np.zeros(7, np.float32))])
    with pytest.raises(ValueError, match="trained dimension"):
        engine.process(bad)

def test_n3_requires_two_grapheme_classes():
    with pytest.raises(ValueError, match="two distinct"):
        N3Processor().fit([
            LabeledN2Patch("a", features=np.ones(4, np.float32)),
            LabeledN2Patch("a", features=np.ones(4, np.float32)),
        ])

def test_n3_recurrent_updates_are_bounded_and_finite():
    engine = training_engine(N3Config(max_iterations=3))
    result = engine.process(N3Sequence("bounded", [
        N3Patch("p1", n2_result=fake_n2_result(20, 0)),
        N3Patch("p2", n2_result=fake_n2_result(21, 1)),
    ]))
    assert len(result.trace) <= 3
    assert np.isfinite(result.posterior_beliefs).all()
    assert np.isfinite(result.classwise_prediction_error).all()
