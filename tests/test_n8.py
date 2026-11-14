import copy
import json

import numpy as np
import pytest

from ore_n8 import (
    N8_ALGORITHM_STEPS,
    N8Config,
    N8Input,
    N8Processor,
    N8RelationEvidence,
    N8SemanticEntry,
)


class FakeN6:
    sequence_id = "sequence-008"
    decoded_graphemes = ("b", "a", "n", "k", "m", "o", "n", "e", "y")
    boundary_probabilities = np.asarray([1.0, 0.4, 1.0])
    tokens = [
        {
            "token_id": "sequence-008:token:1",
            "orthography": "bank",
            "graphemes": ["b", "a", "n", "k"],
            "start": 0,
            "end": 4,
            "confidence": 0.98,
            "boundary_start_probability": 1.0,
            "boundary_end_probability": 0.92,
            "segmentation_probability": 0.95,
            "known_lexicon_entry": True,
            "language": "en",
            "status": "LEXICAL_MATCH",
        },
        {
            "token_id": "sequence-008:token:2",
            "orthography": "money",
            "graphemes": ["m", "o", "n", "e", "y"],
            "start": 4,
            "end": 9,
            "confidence": 0.97,
            "boundary_start_probability": 0.92,
            "boundary_end_probability": 1.0,
            "segmentation_probability": 0.95,
            "known_lexicon_entry": True,
            "language": "en",
            "status": "LEXICAL_MATCH",
        },
    ]
    metadata = {"schema": "ore.n6.hierarchical_orthographic_access.v1"}


class FakeN7:
    sequence_id = "sequence-008"
    evidence = {
        "visual": [0.8, 0.2],
        "orthographic": [0.7, 0.3],
        "n6_contextual": [0.6, 0.4],
    }
    predictions = [0.6, 0.4, 0.3, 0.8]
    precision = [1.2, 0.9]
    prediction_errors = [0.1, -0.2]
    posterior = [0.8, 0.2]
    ranked_hypotheses = [
        {"intent": "recognize", "domain": "language", "probability": 0.8},
        {"intent": "compare", "domain": "visual", "probability": 0.2},
    ]
    association_ranking = [("financial", 0.75)]
    metadata = {"schema": "ore.n7.predictive_contextual_processing.v1"}


def make_entries():
    return [
        N8SemanticEntry(
            entry_id="bank-finance",
            orthography="bank",
            concept_id="FINANCIAL_INSTITUTION",
            sense_id="bank.n.01",
            gloss="institution that provides financial services",
            verified_by="reviewer-a",
            source_reference="lexicon://en/bank/finance",
            frequency=8.0,
            language="en",
            concept_type="institution",
            domain_weights={"language": 1.0, "visual": 0.1},
            intent_weights={"recognize": 1.0, "compare": 0.5},
            semantic_features={"institution": 1.0, "financial": 1.0, "geographical": 0.0},
            association_tags=("financial",),
        ),
        N8SemanticEntry(
            entry_id="bank-river",
            orthography="bank",
            concept_id="RIVER_EDGE",
            sense_id="bank.n.02",
            gloss="sloping land beside a river",
            verified_by="reviewer-b",
            source_reference="lexicon://en/bank/river",
            frequency=7.0,
            language="en",
            concept_type="landform",
            domain_weights={"language": 0.4, "visual": 1.0},
            intent_weights={"recognize": 0.7, "compare": 1.0},
            semantic_features={"institution": 0.0, "financial": 0.0, "geographical": 1.0},
        ),
        N8SemanticEntry(
            entry_id="money-entry",
            orthography="money",
            concept_id="MONEY",
            sense_id="money.n.01",
            gloss="medium of exchange",
            verified_by="reviewer-c",
            source_reference="lexicon://en/money",
            frequency=10.0,
            language="en",
            concept_type="abstract_value",
            domain_weights={"language": 1.0},
            semantic_features={"financial": 1.0, "institution": 0.2},
            association_tags=("financial",),
        ),
    ]


def make_request(**kwargs):
    return N8Input(
        n7_result=FakeN7(),
        n6_result=FakeN6(),
        semantic_lexicon=make_entries(),
        **kwargs,
    )


def test_n8_has_exactly_72_unique_ordered_operations():
    assert len(N8_ALGORITHM_STEPS) == 72
    assert len(set(N8_ALGORITHM_STEPS)) == 72
    assert N8_ALGORITHM_STEPS[0].startswith("01_")
    assert N8_ALGORITHM_STEPS[-1].startswith("72_")


def test_full_pipeline_resolves_senses_and_returns_complete_trace():
    result = N8Processor().process(make_request(
        semantic_context={"global_sense_weights": {"bank-finance": 1.0, "bank-river": 0.05}},
        relation_evidence=[N8RelationEvidence(
            source_token_index=0,
            target_token_index=1,
            relation="ASSOCIATED_WITH",
            confidence=0.95,
            verified_by="parser-reviewer",
            source_reference="relations://sequence-008/edge-1",
        )],
    ))
    assert len(result.operation_trace) == 72
    assert [row["operation"] for row in result.operation_trace] == list(range(1, 73))
    assert all(row["status"] == "completed" for row in result.operation_trace)
    assert result.diagnostics["operation_trace_complete"]
    assert result.diagnostics["valid"]
    assert result.token_semantics[0]["selected_sense_id"] == "bank.n.01"
    assert result.token_semantics[1]["selected_concept_id"] == "MONEY"
    assert result.diagnostics["supported_relation_count"] == 1
    assert result.compositional_frames[0]["composition_status"] == "EXPLICIT_RELATION_COMPONENT"
    assert np.isclose(sum(result.intent_posterior.values()), 1.0)
    assert np.isclose(sum(result.domain_posterior.values()), 1.0)


def test_unknown_token_is_preserved_without_invented_concept():
    n6 = copy.deepcopy(FakeN6())
    n6.tokens = [dict(FakeN6.tokens[0], token_id="unknown-1", orthography="xylophone",
                      known_lexicon_entry=False, status="UNLEXICALIZED_GRAPHEME")]
    n6.decoded_graphemes = tuple("xylophone")
    n7 = copy.deepcopy(FakeN7())
    n7.sequence_id = n6.sequence_id
    n7.posterior = [0.8, 0.2]
    n7.ranked_hypotheses = FakeN7.ranked_hypotheses
    result = N8Processor().process(N8Input(n7_result=n7, n6_result=n6, semantic_lexicon=[]))
    token = result.token_semantics[0]
    assert token["status"] == "UNKNOWN_OR_UNMAPPED"
    assert token["selected_concept_id"] is None
    assert result.diagnostics["unknown_token_count"] == 1
    assert result.compositional_frames == []
    assert result.diagnostics["sequence_semantic_confidence"] == 0.0


def test_input_objects_are_not_mutated_and_output_is_json_serializable():
    n6, n7 = FakeN6(), FakeN7()
    before_n6 = copy.deepcopy(n6.tokens)
    before_n7 = copy.deepcopy(n7.posterior)
    result = N8Processor().process(N8Input(
        n7_result=n7,
        n6_result=n6,
        semantic_lexicon=make_entries(),
    ))
    assert n6.tokens == before_n6
    assert n7.posterior == before_n7
    json.dumps(result.to_dict(), ensure_ascii=False)
    assert result.diagnostics["upstream_inputs_unchanged"]


def test_unverified_semantic_entries_are_rejected():
    entry = {
        "entry_id": "unsafe",
        "orthography": "bank",
        "concept_id": "FINANCIAL_INSTITUTION",
        "sense_id": "bank.n.01",
        "gloss": "financial institution",
        "frequency": 1.0,
    }
    with pytest.raises(ValueError, match="provenance"):
        N8Processor().process(N8Input(
            n7_result=FakeN7(), n6_result=FakeN6(), semantic_lexicon=[entry]
        ))


def test_relation_with_unknown_endpoint_is_retained_but_not_composed():
    relation = N8RelationEvidence(0, 1, "RELATED_TO", 0.9, "reviewer", "rel://1")
    n6 = copy.deepcopy(FakeN6())
    n6.tokens[0]["orthography"] = "not-in-lexicon"
    result = N8Processor().process(N8Input(
        n7_result=FakeN7(), n6_result=n6, semantic_lexicon=make_entries(),
        relation_evidence=[relation],
    ))
    assert result.semantic_graph["edges"][0]["status"] == "UNRESOLVED_ENDPOINT_OR_LOW_CONFIDENCE"
    assert result.diagnostics["supported_relation_count"] == 0


def test_mismatched_sequences_and_invalid_configuration_are_rejected():
    n7 = copy.deepcopy(FakeN7())
    n7.sequence_id = "different"
    with pytest.raises(ValueError, match="sequence"):
        N8Processor().process(N8Input(n7_result=n7, n6_result=FakeN6()))
    with pytest.raises(ValueError):
        N8Config(abstain_threshold=1.5)


def test_relation_endpoint_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        N8Processor().process(make_request(relation_evidence=[
            N8RelationEvidence(0, 9, "RELATED_TO", 0.8, "reviewer", "rel://bad")
        ]))
