import numpy as np
import pytest

from ore_n7 import N7Config, N7Input, N7Processor, N7_ALGORITHM_STEPS


class FakeN6:
    sequence_id = "fixture-001"
    class_names = ("a", "b", "c")
    input_probabilities = np.array([[0.7,0.2,0.1],[0.6,0.3,0.1]], dtype=float)
    contextual_probabilities = np.array([[0.6,0.3,0.1],[0.5,0.4,0.1]], dtype=float)
    decoded_probabilities = np.array([[0.8,0.1,0.1],[0.1,0.8,0.1]], dtype=float)
    decoded_graphemes = ("a","b")
    boundary_probabilities = np.array([1.0,0.4,1.0])
    tokens = [{"orthography":"ab"}]
    metadata = {"schema":"ore.n6.hierarchical_orthographic_access.v1"}


def make_request(**kwargs):
    return N7Input(n6_result=FakeN6(), **kwargs)


def test_algorithm_has_exactly_72_unique_ordered_operations():
    assert len(N7_ALGORITHM_STEPS) == 72
    assert len(set(N7_ALGORITHM_STEPS)) == 72
    assert N7_ALGORITHM_STEPS[0].startswith("01_")
    assert N7_ALGORITHM_STEPS[-1].startswith("72_")


def test_full_pipeline_returns_72_operation_trace_and_separate_evidence():
    result = N7Processor().process(make_request(
        context_state={
            "intent_prior":{"recognize":0.8,"compare":0.2},
            "domain_prior":{"orthography":0.9,"language":0.1},
            "memory":{"activation":0.2},
        },
        provider_evidence={
            "visual_strength":0.8,"orthographic_strength":0.7,
            "context_strength":0.4,"task_strength":0.9,
        },
        script_profile="latin_basic",
        task_context={"goal":"recognize"},
    ))
    assert len(result.operation_trace) == 72
    assert [x["operation"] for x in result.operation_trace] == list(range(1,73))
    assert result.diagnostics["operation_trace_complete"]
    assert set(result.evidence) == {"visual","orthographic","n6_contextual","boundary_mean"}
    assert np.isclose(sum(result.posterior),1.0)
    assert np.isclose(sum(result.style_posterior.values()),1.0)


def test_invalid_provider_probability_is_rejected():
    with pytest.raises(ValueError):
        N7Processor().process(make_request(provider_evidence={"visual_strength":1.5}))


def test_unknown_provider_cue_is_ignored_but_recorded_only_if_configured():
    result = N7Processor().process(make_request(provider_evidence={"visual_strength":0.5}))
    assert result.diagnostics["valid"]


def test_invalid_context_prior_is_rejected():
    with pytest.raises(ValueError):
        N7Processor().process(make_request(context_state={"intent_prior":{"recognize":-1}}))


def test_config_rejects_incompatible_cue_maps():
    with pytest.raises(ValueError):
        N7Config(observation_variance={"other":0.2})


def test_prediction_errors_and_precision_are_finite():
    result = N7Processor().process(make_request())
    assert np.isfinite(result.prediction_errors).all()
    assert np.isfinite(result.precision).all()
    assert all(v >= 0 for v in result.layer_errors.values())


def test_trace_records_each_operation_as_completed():
    result = N7Processor().process(make_request())
    assert all(row["status"] == "completed" for row in result.operation_trace)
    assert result.operation_trace[-1]["details"]["result_returned"]


def test_context_memory_is_not_mutated():
    memory = {"activation":0.4,"prior_topic":0.7}
    before = dict(memory)
    N7Processor().process(make_request(context_state={"memory":memory}))
    assert memory == before


def test_n6_result_is_required():
    with pytest.raises(ValueError):
        N7Processor().process(N7Input(n6_result=None))
