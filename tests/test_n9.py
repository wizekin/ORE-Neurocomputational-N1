from __future__ import annotations

import json

import numpy as np
import pytest

from ore_n9 import (
    N9_ALGORITHM_STEPS,
    N9Adapters,
    N9Config,
    N9Patch,
    N9Processor,
    N9Request,
)


def simple_request() -> N9Request:
    return N9Request(
        sequence_id="n9-test",
        patches=[
            N9Patch("p0", grapheme_probabilities={"a": 0.92, "b": 0.08}),
            N9Patch("p1", grapheme_probabilities={"a": 0.12, "b": 0.88}),
        ],
        word_spans=[(0, 2)],
        context={"domain": "test"},
    )


def test_manifest_has_exactly_120_unique_operations() -> None:
    assert len(N9_ALGORITHM_STEPS) == 120
    assert len(set(N9_ALGORITHM_STEPS)) == 120


def test_pipeline_executes_all_operations_and_preserves_evidence_channels() -> None:
    result = N9Processor().process(simple_request())

    assert result.diagnostics["valid"] is True
    assert result.diagnostics["stage_count"] == 120
    assert result.diagnostics["executed_operation_count"] == 120
    assert [entry["step"] for entry in result.operation_trace] == list(range(1, 121))
    assert result.visual_probabilities.shape == (2, 2)
    assert result.posterior_beliefs.shape == (2, 2)
    assert np.allclose(result.visual_probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.posterior_beliefs.sum(axis=1), 1.0)
    assert np.isclose(result.attention_allocation.sum(), 1.0)
    assert result.metadata["evidence_decomposition"]["visual_is_preserved_separately"] is True
    assert result.semantic_graph == {"nodes": [], "edges": []}


def test_unknown_fallback_is_explicit_and_does_not_invent_a_grapheme() -> None:
    request = N9Request(
        sequence_id="unknown",
        patches=[N9Patch("p0", features=np.asarray([0.1, 0.2], dtype=np.float32))],
    )
    result = N9Processor().process(request)

    assert result.class_names == ("<UNK>",)
    assert result.decision["status"] == "ABSTAIN"
    assert result.lexical_hypotheses[0]["form"] == ""
    assert result.operation_trace[25]["status"] == "abstention_fallback"


def test_image_patch_runs_existing_n1_and_n2_frontends() -> None:
    yy, xx = np.mgrid[0:48, 0:48]
    image = (0.5 + 0.35 * np.sin(xx / 3.0)).astype(np.float32)
    request = N9Request(sequence_id="image", patches=[N9Patch("patch", image=image)])
    result = N9Processor().process(request)

    assert result.diagnostics["valid"] is True
    assert result.operation_trace[12]["status"] == "ok"
    assert result.operation_trace[23]["status"] == "ok"
    assert result.class_names == ("<UNK>",)


def test_n8_graph_validation_discards_dangling_edges() -> None:
    def semantic_adapter(payload):
        return {
            "semantic_graph": {
                "nodes": [
                    {"id": "w1", "type": "concept", "label": "known"},
                    {"id": "w2", "type": "event", "label": "known-event"},
                ],
                "edges": [
                    {"source": "w1", "target": "w2", "relation": "agent"},
                    {"source": "missing", "target": "w2", "relation": "invented"},
                    {"source": "w1", "target": "w2", "relation": ""},
                ],
            },
            "propositions": [{"predicate": "test", "arguments": ["w1", "w2"]}],
            "completeness": 0.8,
            "ambiguity": 0.2,
        }

    result = N9Processor(adapters=N9Adapters(n8=semantic_adapter)).process(simple_request())
    assert len(result.semantic_graph["nodes"]) == 2
    assert result.semantic_graph["edges"] == [
        {"source": "w1", "target": "w2", "relation": "agent"}
    ]
    assert len(result.propositions) == 1


def test_extension_adapters_are_called_through_declared_interfaces() -> None:
    calls = {"n6": 0, "n7": 0, "syntax": 0, "n8": 0}

    def n6(payload):
        calls["n6"] += 1
        return {
            "lexical_hypotheses": [
                {"form": "ab", "start": 0, "end": 2, "score": 0.99, "source": "test-lexicon"}
            ],
            "pos_features": [{"token": "ab", "pos": "NOUN"}],
        }

    def syntax(payload):
        calls["syntax"] += 1
        return {"parses": [{"root": "ab"}], "consistency": 0.9}

    def n7(payload):
        calls["n7"] += 1
        return {"coherence": 0.85, "uncertainty": 0.1, "intents": ["test"]}

    def n8(payload):
        calls["n8"] += 1
        return {
            "semantic_graph": {"nodes": [], "edges": []},
            "propositions": [],
            "completeness": 0.5,
        }

    result = N9Processor(
        adapters=N9Adapters(n6=n6, n7=n7, n8=n8, syntax=syntax)
    ).process(simple_request())

    assert calls == {"n6": 1, "n7": 1, "syntax": 1, "n8": 1}
    assert result.lexical_hypotheses[0]["source"] == "test-lexicon"
    assert result.syntax["parses"][0]["root"] == "ab"


def test_learning_is_skipped_without_explicit_authorization() -> None:
    calls = []

    def n5(payload):
        calls.append(payload)
        return {"child_fingerprint": "child"}

    result = N9Processor(adapters=N9Adapters(n5=n5)).process(simple_request())

    assert calls == []
    assert result.learning["status"] == "skipped_without_explicit_authorization"
    assert result.learning["parent_model_mutated"] is False


def test_config_rejects_unsafe_ranges() -> None:
    with pytest.raises(ValueError):
        N9Config(damping=0)
    with pytest.raises(ValueError):
        N9Config(confidence_threshold=1.5)
    with pytest.raises(ValueError):
        N9Config(max_feedback_iterations=0)


def test_result_serializes_without_nan() -> None:
    result = N9Processor().process(simple_request())
    encoded = json.dumps(result.to_dict(), allow_nan=False)
    assert '"schema": "ore.n9.integrated_neurocomputational_ore.v1"' in encoded



def test_authorized_learning_uses_a_detached_child_adapter() -> None:
    class DummyN5:
        def __init__(self):
            self.proposals = 0

        @property
        def fingerprint(self):
            return f"parent-v{self.proposals}"

        def propose_update(self, payload):
            self.proposals += 1
            return {"child_model": {"version": 1}, "metadata": {"detached": True}}

    parent = DummyN5()
    request = simple_request()
    request.learning_authorized = True
    result = N9Processor(adapters=N9Adapters(n5=parent)).process(request)

    assert parent.proposals == 0
    assert result.learning["status"] == "child_proposed"
    assert result.learning["child_model_created"] is True
    assert result.learning["parent_model_mutated"] is False
