"""ORE N8: 72-operation lexical-semantic access and composition stage.

This is an algorithmic research approximation. It performs exact-form lookup
against caller-supplied, provenance-checked semantic entries and composes only
explicitly supplied relation evidence. It does not generate meanings, parse
syntax, or claim a neuron-level implementation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from math import log
from time import perf_counter
from typing import Any, Mapping
import unicodedata

import numpy as np

from .config import N8Config
from .contracts import N8Input, N8RelationEvidence, N8Result, N8SemanticEntry

N8_ALGORITHM_STEPS: tuple[str, ...] = (
    "01_validate_N8_request_contract",
    "02_validate_N7_result_interface_and_schema",
    "03_validate_N6_result_interface_and_schema",
    "04_validate_sequence_identity_alignment",
    "05_snapshot_consumed_upstream_evidence",
    "06_validate_N7_joint_posterior_distribution",
    "07_preserve_visual_orthographic_and_context_channels",
    "08_validate_N6_token_records_and_required_fields",
    "09_fingerprint_read_only_upstream_inputs",
    "10_extract_ordered_token_identifiers",
    "11_normalize_token_orthography_with_Unicode_NFC",
    "12_validate_token_spans_and_reading_order",
    "13_extract_N6_token_confidence",
    "14_extract_N6_boundary_confidence",
    "15_extract_N6_segmentation_and_lexical_status",
    "16_derive_token_reliability_weights",
    "17_identify_upstream_unlexicalized_tokens",
    "18_construct_aligned_token_feature_records",
    "19_load_external_semantic_lexicon_snapshot",
    "20_validate_semantic_entry_provenance",
    "21_normalize_lexicon_orthographic_forms",
    "22_validate_concept_and_sense_identifiers",
    "23_validate_semantic_frequency_weights",
    "24_validate_domain_and_intent_compatibility_maps",
    "25_validate_sparse_semantic_feature_vectors",
    "26_index_semantic_entries_by_exact_orthographic_form",
    "27_fingerprint_immutable_semantic_resource_snapshot",
    "28_read_explicit_semantic_context_payload",
    "29_validate_global_sense_evidence_weights",
    "30_validate_token_specific_sense_evidence",
    "31_extract_N7_intent_probability_distribution",
    "32_extract_N7_domain_probability_distribution",
    "33_generate_exact_form_sense_candidate_sets",
    "34_add_unknown_concept_fallback_candidate",
    "35_apply_deterministic_candidate_limit",
    "36_measure_lexical_and_semantic_candidate_coverage",
    "37_compute_frequency_based_sense_prior",
    "38_compute_domain_compatibility_likelihood",
    "39_compute_intent_compatibility_likelihood",
    "40_compute_explicit_sense_evidence_likelihood",
    "41_compute_contextual_association_compatibility",
    "42_weight_semantic_evidence_by_token_reliability",
    "43_combine_candidate_log_posterior_scores",
    "44_normalize_token_sense_posteriors",
    "45_retain_complete_ranked_sense_alternatives",
    "46_compute_categorical_sense_uncertainty",
    "47_measure_top_two_sense_posterior_margin",
    "48_estimate_token_level_semantic_evidence_coverage",
    "49_measure_context_and_lexicon_conflict",
    "50_reliability_adjust_sense_posterior",
    "51_evaluate_sense_confidence_and_ambiguity",
    "52_apply_unknown_and_abstention_gates",
    "53_construct_token_semantic_state_records",
    "54_preserve_unselected_and_alternative_senses",
    "55_validate_explicit_relation_evidence_records",
    "56_validate_relation_token_endpoints",
    "57_validate_relation_type_confidence_and_provenance",
    "58_bind_relation_endpoints_to_semantic_token_states",
    "59_compute_endpoint_weighted_relation_confidence",
    "60_construct_concept_graph_nodes",
    "61_construct_only_explicitly_supported_graph_edges",
    "62_group_connected_evidence_into_compositional_frames",
    "63_measure_frame_completeness_and_confidence",
    "64_aggregate_concept_activation_across_tokens",
    "65_normalize_and_rank_global_concept_hypotheses",
    "66_preserve_upstream_and_N8_evidence_as_separate_channels",
    "67_compute_sequence_level_semantic_confidence_and_entropy",
    "68_rank_concept_and_frame_outputs_deterministically",
    "69_construct_token_level_semantic_report",
    "70_build_diagnostics_configuration_and_provenance",
    "71_verify_input_immutability_output_integrity_and_trace_order",
    "72_return_structured_N8_lexical_semantic_result",
)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(_plain(value), sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def _normalized_form(value: str, casefold: bool) -> str:
    normalized = unicodedata.normalize("NFC", value).strip()
    return normalized.casefold() if casefold else normalized


def _bounded_weight_map(value: Any, name: str, *, upper: float = 1.0) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        number = float(raw_value)
        if not key or not np.isfinite(number) or number < 0.0 or number > upper:
            raise ValueError(f"{name} values must be finite and lie in [0, {upper}]")
        result[key] = number
    return result


def _softmax(scores: np.ndarray, temperature: float, epsilon: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64) / temperature
    values = values - float(np.max(values))
    exponentials = np.exp(np.clip(values, -700.0, 0.0))
    return exponentials / max(float(exponentials.sum()), epsilon)


def _entropy(probabilities: np.ndarray, epsilon: float) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    return float(-np.sum(p * np.log(np.maximum(p, epsilon))))


def _feature_similarity(left: Mapping[str, float], right: Mapping[str, float], epsilon: float) -> float:
    shared = sorted(set(left).intersection(right))
    if not shared:
        return 0.5
    a = np.asarray([left[key] for key in shared], dtype=np.float64)
    b = np.asarray([right[key] for key in shared], dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= epsilon:
        return 0.5
    return float(np.clip(np.dot(a, b) / denominator, 0.0, 1.0))


def _as_semantic_entry(raw: Any, config: N8Config) -> dict[str, Any]:
    if isinstance(raw, N8SemanticEntry):
        item = asdict(raw)
    elif isinstance(raw, Mapping):
        item = dict(raw)
    else:
        names = ("entry_id", "orthography", "concept_id", "sense_id", "gloss",
                 "verified_by", "source_reference", "frequency", "language",
                 "concept_type", "domain_weights", "intent_weights",
                 "semantic_features", "association_tags")
        item = {name: getattr(raw, name) for name in names if hasattr(raw, name)}
    required = ("entry_id", "orthography", "concept_id", "sense_id", "gloss")
    missing = [name for name in required if not str(item.get(name, "")).strip()]
    if missing:
        raise ValueError(f"semantic entry is missing required fields: {missing}")
    if config.require_provenance and (
        not str(item.get("verified_by", "")).strip()
        or not str(item.get("source_reference", "")).strip()
    ):
        raise ValueError("semantic entries require verified_by and source_reference provenance")
    frequency = float(item.get("frequency", 1.0))
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("semantic entry frequency must be finite and positive")
    language = str(item.get("language", "und")).strip() or "und"
    raw_tags = item.get("association_tags", ())
    if isinstance(raw_tags, str):
        raw_tags = (raw_tags,)
    tags = tuple(sorted({str(tag).strip() for tag in raw_tags if str(tag).strip()}))
    features = _bounded_weight_map(item.get("semantic_features", {}), "semantic_features")
    domain_weights = _bounded_weight_map(item.get("domain_weights", {}), "domain_weights")
    intent_weights = _bounded_weight_map(item.get("intent_weights", {}), "intent_weights")
    return {
        "entry_id": str(item["entry_id"]).strip(),
        "orthography": str(item["orthography"]).strip(),
        "normalized_orthography": _normalized_form(str(item["orthography"]), config.casefold_orthography),
        "concept_id": str(item["concept_id"]).strip(),
        "sense_id": str(item["sense_id"]).strip(),
        "gloss": str(item["gloss"]).strip(),
        "verified_by": str(item.get("verified_by", "")),
        "source_reference": str(item.get("source_reference", "")),
        "frequency": frequency,
        "language": language,
        "concept_type": str(item.get("concept_type", "unspecified")),
        "domain_weights": domain_weights,
        "intent_weights": intent_weights,
        "semantic_features": features,
        "association_tags": tags,
    }


def _as_relation(raw: Any) -> dict[str, Any]:
    if isinstance(raw, N8RelationEvidence):
        item = asdict(raw)
    elif isinstance(raw, Mapping):
        item = dict(raw)
    else:
        names = ("source_token_index", "target_token_index", "relation", "confidence",
                 "verified_by", "source_reference")
        item = {name: getattr(raw, name) for name in names if hasattr(raw, name)}
    required = ("source_token_index", "target_token_index", "relation", "confidence")
    if any(name not in item for name in required):
        raise ValueError("relation evidence is missing endpoint, relation, or confidence fields")
    source = item["source_token_index"]
    target = item["target_token_index"]
    if isinstance(source, bool) or isinstance(target, bool) or int(source) != source or int(target) != target:
        raise ValueError("relation token indices must be integers")
    confidence = float(item["confidence"])
    relation = str(item["relation"]).strip()
    if not relation or not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("relation type must be non-empty and confidence must lie in [0,1]")
    return {
        "source_token_index": int(source), "target_token_index": int(target),
        "relation": relation,
        "confidence": confidence,
        "verified_by": str(item.get("verified_by", "")),
        "source_reference": str(item.get("source_reference", "")),
    }


class N8Processor:
    """Run 72 N8 operations over completed N6 and N7 results."""

    def __init__(self, config: N8Config | None = None):
        self.config = config or N8Config()

    def _run(self, request: N8Input) -> N8Result:
        started = perf_counter()
        cfg = self.config
        trace: list[dict[str, Any]] = []

        def mark(index: int, **details: Any) -> None:
            trace.append({"operation": index, "name": N8_ALGORITHM_STEPS[index - 1],
                          "status": "completed", "details": _plain(details)})

        # A — request integrity and evidence separation (1–9)
        if not isinstance(request, N8Input):
            raise TypeError("N8Processor.process requires N8Input")
        mark(1, request_type=type(request).__name__)
        n7 = request.n7_result
        if n7 is None:
            raise ValueError("N8 requires a completed N7Result")
        n6 = request.n6_result
        if n6 is None:
            raise ValueError("N8 requires the N6Result that contains the token sequence")
        n7_metadata = dict(_get(n7, "metadata", {}) or {})
        if n7_metadata.get("schema") != "ore.n7.predictive_contextual_processing.v1":
            raise ValueError("unsupported or missing N7 result schema")
        mark(2, schema=n7_metadata["schema"])
        n6_metadata = dict(_get(n6, "metadata", {}) or {})
        if n6_metadata.get("schema") != "ore.n6.hierarchical_orthographic_access.v1":
            raise ValueError("unsupported or missing N6 result schema")
        mark(3, schema=n6_metadata["schema"])
        sequence_id = str(_get(n6, "sequence_id", ""))
        if not sequence_id or sequence_id != str(_get(n7, "sequence_id", "")):
            raise ValueError("N6 and N7 sequence identities must match")
        mark(4, sequence_id=sequence_id)
        n6_snapshot = {
            "sequence_id": sequence_id, "metadata": n6_metadata,
            "tokens": _plain(_get(n6, "tokens", [])),
            "decoded_graphemes": _plain(_get(n6, "decoded_graphemes", ())),
            "boundary_probabilities": _plain(_get(n6, "boundary_probabilities", [])),
        }
        n7_evidence = _get(n7, "evidence", {})
        if not isinstance(n7_evidence, Mapping):
            raise ValueError("N7 evidence must be a mapping of preserved channels")
        n7_snapshot = {
            "sequence_id": sequence_id, "metadata": n7_metadata,
            "posterior": _plain(_get(n7, "posterior", [])),
            "ranked_hypotheses": _plain(_get(n7, "ranked_hypotheses", [])),
            "evidence": _plain(n7_evidence),
            "association_ranking": _plain(_get(n7, "association_ranking", [])),
        }
        mark(5, n6_snapshot_fields=len(n6_snapshot), n7_snapshot_fields=len(n7_snapshot))
        n7_posterior = np.asarray(_get(n7, "posterior", []), dtype=np.float64)
        ranked_hypotheses = list(_get(n7, "ranked_hypotheses", []) or [])
        if (n7_posterior.ndim != 1 or n7_posterior.size == 0
                or not np.isfinite(n7_posterior).all() or np.any(n7_posterior < 0.0)
                or not np.isclose(float(n7_posterior.sum()), 1.0, atol=1e-5)
                or len(ranked_hypotheses) != n7_posterior.size):
            raise ValueError("N7 posterior and ranked hypotheses must be aligned normalized vectors")
        mark(6, hypothesis_count=int(n7_posterior.size), posterior_sum=float(n7_posterior.sum()))
        preserved_evidence = {
            "visual": _plain(n7_evidence.get("visual", [])),
            "orthographic": _plain(n7_evidence.get("orthographic", [])),
            "n6_contextual": _plain(n7_evidence.get("n6_contextual", [])),
            "n7_predictions": _plain(_get(n7, "predictions", [])),
            "n7_prediction_errors": _plain(_get(n7, "prediction_errors", [])),
            "n7_posterior": n7_posterior.tolist(),
        }
        mark(7, evidence_channels=list(preserved_evidence))
        raw_tokens = list(_get(n6, "tokens", []) or [])
        if not raw_tokens:
            raise ValueError("N6 must provide at least one segmented token")
        if len(raw_tokens) > cfg.max_tokens:
            raise ValueError("N6 token sequence exceeds max_tokens")
        required_token_fields = ("token_id", "orthography", "start", "end", "confidence")
        for index, token in enumerate(raw_tokens):
            missing = [name for name in required_token_fields if _get(token, name, None) is None]
            if missing:
                raise ValueError(f"N6 token {index} is missing fields: {missing}")
        mark(8, token_count=len(raw_tokens), required_fields=list(required_token_fields))
        initial_n6_fingerprint = _fingerprint(n6_snapshot)
        initial_n7_fingerprint = _fingerprint(n7_snapshot)
        mark(9, n6_fingerprint=initial_n6_fingerprint, n7_fingerprint=initial_n7_fingerprint)

        # B — token reconstruction and reliability (10–18)
        token_rows: list[dict[str, Any]] = []
        previous_end = 0
        for index, raw in enumerate(raw_tokens):
            token_id = str(_get(raw, "token_id", "")).strip()
            orthography = str(_get(raw, "orthography", ""))
            normalized = _normalized_form(orthography, cfg.casefold_orthography)
            start, end = int(_get(raw, "start")), int(_get(raw, "end"))
            confidence = float(_get(raw, "confidence"))
            if not token_id or not normalized:
                raise ValueError("N6 token_id and orthography must be non-empty")
            if start < previous_end or end <= start:
                raise ValueError("N6 token spans must be positive, ordered and non-overlapping")
            if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("N6 token confidence must be finite and lie in [0,1]")
            previous_end = end
            boundary_confidence = float(min(
                float(_get(raw, "boundary_start_probability", 1.0)),
                float(_get(raw, "boundary_end_probability", 1.0)),
            ))
            if not np.isfinite(boundary_confidence) or not 0.0 <= boundary_confidence <= 1.0:
                raise ValueError("N6 boundary confidence must lie in [0,1]")
            status = str(_get(raw, "status", "UNSPECIFIED"))
            segmentation_probability = float(_get(raw, "segmentation_probability", 1.0))
            if not np.isfinite(segmentation_probability) or not 0.0 <= segmentation_probability <= 1.0:
                raise ValueError("N6 segmentation probability must lie in [0,1]")
            reliability = float(np.clip(confidence * (0.75 + 0.25 * boundary_confidence)
                                         * (0.80 + 0.20 * segmentation_probability), 0.0, 1.0))
            upstream_unknown = bool(not _get(raw, "known_lexicon_entry", False)
                                    or status in {"UNLEXICALIZED_GRAPHEME", "ABSTAIN", "UNKNOWN"})
            row = {
                "token_index": index, "token_id": token_id,
                "orthography": orthography, "normalized_orthography": normalized,
                "graphemes": _plain(_get(raw, "graphemes", [])),
                "start": start, "end": end, "language": str(_get(raw, "language", "und")),
                "n6_confidence": confidence, "boundary_confidence": boundary_confidence,
                "segmentation_probability": segmentation_probability,
                "token_reliability": reliability, "n6_status": status,
                "upstream_unknown": upstream_unknown,
            }
            token_rows.append(row)
        if len({row["token_id"] for row in token_rows}) != len(token_rows):
            raise ValueError("N6 token_id values must be unique")
        mark(10, token_count=len(token_rows), unique_ids=len({row["token_id"] for row in token_rows}))
        mark(11, normalized_form_count=len({row["normalized_orthography"] for row in token_rows}))
        mark(12, positive_ordered_spans=True, final_end=token_rows[-1]["end"])
        mark(13, mean_token_confidence=float(np.mean([row["n6_confidence"] for row in token_rows])))
        mark(14, mean_boundary_confidence=float(np.mean([row["boundary_confidence"] for row in token_rows])))
        mark(15, upstream_status_counts={status: sum(row["n6_status"] == status for row in token_rows)
                                         for status in sorted({row["n6_status"] for row in token_rows})})
        mark(16, mean_token_reliability=float(np.mean([row["token_reliability"] for row in token_rows])))
        mark(17, upstream_unknown_count=sum(row["upstream_unknown"] for row in token_rows))
        mark(18, aligned_feature_record_count=len(token_rows))

        # C — verified semantic lexicon (19–27)
        raw_entries = list(request.semantic_lexicon or ())
        if len(raw_entries) > cfg.max_lexicon_entries:
            raise ValueError("semantic lexicon exceeds max_lexicon_entries")
        mark(19, raw_entry_count=len(raw_entries))
        entries = [_as_semantic_entry(item, cfg) for item in raw_entries]
        if cfg.require_provenance and any(not e["verified_by"] or not e["source_reference"] for e in entries):
            raise ValueError("all semantic entries must carry independent provenance")
        mark(20, provenance_checked=True)
        for entry in entries:
            if not entry["normalized_orthography"]:
                raise ValueError("semantic entry orthography cannot normalize to an empty string")
        mark(21, normalized_forms=len(entries))
        if len({e["entry_id"] for e in entries}) != len(entries):
            raise ValueError("semantic entry IDs must be unique")
        if any(not e["concept_id"] or not e["sense_id"] for e in entries):
            raise ValueError("concept_id and sense_id must be non-empty")
        mark(22, unique_entry_ids=len(entries))
        mark(23, min_frequency=min((e["frequency"] for e in entries), default=0.0))
        mark(24, compatibility_maps_valid=True)
        mark(25, feature_dimensions=len({k for e in entries for k in e["semantic_features"]}))
        lexicon_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            lexicon_index[entry["normalized_orthography"]].append(entry)
        for values in lexicon_index.values():
            values.sort(key=lambda e: (-e["frequency"], e["entry_id"]))
        mark(26, indexed_forms=len(lexicon_index))
        lexicon_fingerprint = _fingerprint(entries)
        mark(27, lexicon_fingerprint=lexicon_fingerprint)
        # D — semantic context and candidate generation (28–36)
        semantic_context = dict(request.semantic_context or {})
        mark(28, context_keys=sorted(str(k) for k in semantic_context))
        global_sense_weights = _bounded_weight_map(
            semantic_context.get("global_sense_weights", {}), "global_sense_weights"
        )
        mark(29, global_sense_weight_count=len(global_sense_weights))
        token_sense_raw = semantic_context.get("token_sense_weights", {})
        if not isinstance(token_sense_raw, Mapping):
            raise TypeError("token_sense_weights must be a mapping")
        token_sense_weights = {
            str(token_key): _bounded_weight_map(weights, f"token_sense_weights[{token_key}]")
            for token_key, weights in token_sense_raw.items()
        }
        mark(30, token_hint_count=len(token_sense_weights))
        intent_mass: dict[str, float] = defaultdict(float)
        domain_mass: dict[str, float] = defaultdict(float)
        for item in ranked_hypotheses:
            intent = str(_get(item, "intent", "")).strip()
            domain = str(_get(item, "domain", "")).strip()
            probability = float(_get(item, "probability", float("nan")))
            if (not intent or not domain or not np.isfinite(probability)
                    or probability < 0.0 or probability > 1.0):
                raise ValueError("N7 ranked hypotheses must contain valid intent/domain probabilities")
            intent_mass[intent] += probability
        mark(31, intent_categories=len(intent_mass))
        for item in ranked_hypotheses:
            domain_mass[str(_get(item, "domain"))] += float(_get(item, "probability"))
        if not np.isclose(sum(intent_mass.values()), 1.0, atol=1e-5) or not np.isclose(sum(domain_mass.values()), 1.0, atol=1e-5):
            raise ValueError("N7 intent/domain marginal distributions must sum to one")
        mark(32, domain_categories=len(domain_mass))
        candidates_by_token: list[list[dict[str, Any]]] = []
        matched_count = 0
        for token in token_rows:
            matches = [entry for entry in lexicon_index.get(token["normalized_orthography"], [])
                       if token["language"] in {"", "und"} or entry["language"] in {"und", token["language"]}]
            candidates_by_token.append(matches)
            matched_count += bool(matches)
        mark(33, tokens_with_exact_form_matches=matched_count)
        candidate_sets: list[list[dict[str, Any]]] = []
        for matches in candidates_by_token:
            candidate_sets.append(list(matches))
        mark(34, unknown_fallbacks=len(token_rows))
        for candidates in candidate_sets:
            candidates.sort(key=lambda e: (-e["frequency"], e["entry_id"]))
            del candidates[max(0, cfg.max_candidates_per_token - 1):]
        mark(35, candidate_limit=cfg.max_candidates_per_token)
        lexical_coverage = float(matched_count / len(token_rows))
        mark(36, lexical_coverage=lexical_coverage)

        # E — evidence-weighted sense scoring (37–45)
        token_semantics: list[dict[str, Any]] = []
        all_sense_probabilities: list[np.ndarray] = []
        conflict_values: list[float] = []
        token_feature_hints = semantic_context.get("token_feature_hints", {})
        if not isinstance(token_feature_hints, Mapping):
            raise TypeError("token_feature_hints must be a mapping")
        global_feature_hints = _bounded_weight_map(semantic_context.get("feature_hints", {}), "feature_hints")
        association_raw = list(_get(n7, "association_ranking", []) or [])
        association_scores: dict[str, float] = {}
        for pair in association_raw:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("N7 association_ranking entries must be (key, score) pairs")
            key, score = str(pair[0]), float(pair[1])
            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("N7 association scores must lie in [0,1]")
            association_scores[key] = score
        for index, (token, candidates) in enumerate(zip(token_rows, candidate_sets)):
            frequency_values = np.asarray([e["frequency"] for e in candidates], dtype=np.float64)
            max_frequency = float(max(frequency_values, default=1.0))
            prior_scores = []
            raw_scores = []
            raw_details = []
            token_hints = token_sense_weights.get(token["token_id"],
                          token_sense_weights.get(str(index), {}))
            feature_hints = token_feature_hints.get(token["token_id"], global_feature_hints)
            feature_hints = _bounded_weight_map(feature_hints, "token feature hints")
            for entry in candidates:
                frequency_prior = float(np.log1p(entry["frequency"]) / max(np.log1p(max_frequency), cfg.epsilon))
                domain_compatibility = (
                    sum(domain_mass.get(key, 0.0) * value for key, value in entry["domain_weights"].items())
                    if entry["domain_weights"] else 0.5
                )
                if entry["domain_weights"]:
                    domain_compatibility = float(np.clip(domain_compatibility, cfg.epsilon, 1.0))
                intent_compatibility = (
                    sum(intent_mass.get(key, 0.0) * value for key, value in entry["intent_weights"].items())
                    if entry["intent_weights"] else 0.5
                )
                if entry["intent_weights"]:
                    intent_compatibility = float(np.clip(intent_compatibility, cfg.epsilon, 1.0))
                hint_keys = (entry["entry_id"], entry["sense_id"], entry["concept_id"])
                explicit_hints = [token_hints[key] for key in hint_keys if key in token_hints]
                if not explicit_hints:
                    explicit_hints = [global_sense_weights[key] for key in hint_keys
                                      if key in global_sense_weights]
                hint = float(np.clip(max(explicit_hints) if explicit_hints else 1.0,
                                     cfg.epsilon, 1.0))
                assoc_keys = {entry["entry_id"], entry["sense_id"], entry["concept_id"], *entry["association_tags"]}
                association_support = max((association_scores.get(key, 0.0) for key in assoc_keys), default=0.0)
                feature_match = _feature_similarity(entry["semantic_features"], feature_hints, cfg.epsilon)
                reliability = token["token_reliability"]
                frequency_term = cfg.frequency_weight * log(max(frequency_prior, cfg.epsilon))
                domain_term = cfg.domain_weight * log(max(domain_compatibility, cfg.epsilon))
                intent_term = cfg.intent_weight * log(max(intent_compatibility, cfg.epsilon))
                explicit_term = cfg.explicit_sense_weight * log(max(hint, cfg.epsilon))
                feature_term = cfg.feature_weight * log(max(feature_match, cfg.epsilon))
                association_term = cfg.association_weight * log(max(association_support, cfg.epsilon)) if association_support > 0 else 0.0
                score = frequency_term + reliability * (
                    domain_term + intent_term + explicit_term + feature_term + association_term
                )
                prior_score = frequency_term
                raw_scores.append(score)
                prior_scores.append(prior_score)
                raw_details.append({
                    "entry_id": entry["entry_id"], "concept_id": entry["concept_id"],
                    "sense_id": entry["sense_id"], "gloss": entry["gloss"],
                    "language": entry["language"], "concept_type": entry["concept_type"],
                    "frequency_prior": frequency_prior,
                    "domain_compatibility": float(domain_compatibility),
                    "intent_compatibility": float(intent_compatibility),
                    "explicit_sense_weight": hint, "feature_similarity": feature_match,
                    "association_support": association_support,
                    "score": float(score), "verified_by": entry["verified_by"],
                    "source_reference": entry["source_reference"],
                })
            unknown_score = cfg.unknown_log_prior + (0.65 if token["upstream_unknown"] else 0.0)
            raw_scores.append(unknown_score)
            prior_scores.append(unknown_score)
            raw_details.append({
                "entry_id": None, "concept_id": None, "sense_id": None,
                "gloss": "UNRESOLVED_OR_UNKNOWN", "language": token["language"],
                "concept_type": "unknown", "score": float(unknown_score),
                "verified_by": None, "source_reference": None,
            })
            probabilities = _softmax(np.asarray(raw_scores), cfg.posterior_temperature, cfg.epsilon)
            prior_probabilities = _softmax(np.asarray(prior_scores), cfg.posterior_temperature, cfg.epsilon)
            # Unreliable upstream tokens are pulled toward the broad lexical prior.
            adjusted = token["token_reliability"] * probabilities + (1.0 - token["token_reliability"]) * prior_probabilities
            adjusted = adjusted / max(float(adjusted.sum()), cfg.epsilon)
            all_sense_probabilities.append(adjusted)
            ranked = []
            for detail, probability in zip(raw_details, adjusted):
                ranked.append({**detail, "probability": float(probability)})
            ranked.sort(key=lambda item: (-item["probability"], str(item["entry_id"] or "~UNKNOWN")))

            # F — ambiguity, conflict and abstention (46–54)
            entropy = _entropy(adjusted, cfg.epsilon)
            normalized_entropy = entropy / max(log(len(adjusted)), cfg.epsilon) if len(adjusted) > 1 else 0.0
            margin = float(ranked[0]["probability"] - ranked[1]["probability"]) if len(ranked) > 1 else float(ranked[0]["probability"])
            known_probability = float(sum(item["probability"] for item in ranked if item["entry_id"] is not None))
            context_conflict = float(np.clip(
                sum(1.0 - item[key] for item in ranked if item["entry_id"] is not None
                    for key in ("domain_compatibility", "intent_compatibility"))
                / max(2 * max(len(ranked) - 1, 1), 1), 0.0, 1.0
            ))
            conflict_values.append(context_conflict)
            top = ranked[0]
            resolved = bool(
                top["entry_id"] is not None
                and top["probability"] >= cfg.abstain_threshold
                and margin >= cfg.ambiguity_margin
                and token["token_reliability"] >= 0.05
            )
            ambiguous = bool(top["entry_id"] is not None and margin < cfg.ambiguity_margin)
            low_confidence = bool(top["entry_id"] is not None and top["probability"] < cfg.abstain_threshold)
            if top["entry_id"] is None:
                status = "UNKNOWN_OR_UNMAPPED"
            elif resolved:
                status = "RESOLVED"
            elif ambiguous:
                status = "AMBIGUOUS"
            else:
                status = "ABSTAIN_LOW_CONFIDENCE"
            record = {
                **token,
                "status": status,
                "selected_entry_id": top["entry_id"] if resolved else None,
                "selected_concept_id": top["concept_id"] if resolved else None,
                "selected_sense_id": top["sense_id"] if resolved else None,
                "selected_probability": float(top["probability"]),
                "posterior_entropy": entropy,
                "normalized_entropy": float(normalized_entropy),
                "posterior_margin": margin,
                "known_concept_probability_mass": known_probability,
                "context_conflict": context_conflict,
                "candidates": ranked,
            }
            token_semantics.append(record)

        known_candidates = [
            candidate for token in token_semantics for candidate in token["candidates"]
            if candidate["entry_id"] is not None
        ]
        mark(37, token_count=len(token_rows), mean_frequency_prior=float(np.mean([
            candidate.get("frequency_prior", 0.0) for candidate in known_candidates
        ])) if known_candidates else 0.0)
        mark(38, mean_domain_compatibility=float(np.mean([
            candidate.get("domain_compatibility", 0.5) for candidate in known_candidates
        ])) if known_candidates else 0.5)
        mark(39, mean_intent_compatibility=float(np.mean([
            candidate.get("intent_compatibility", 0.5) for candidate in known_candidates
        ])) if known_candidates else 0.5)
        mark(40, global_sense_evidence_count=len(global_sense_weights))
        mark(41, association_evidence_count=len(association_scores))
        mark(42, reliability_weighted_token_count=len(token_semantics))
        mark(43, scored_sense_candidate_count=sum(
            len(token["candidates"]) - 1 for token in token_semantics
        ))
        mark(44, normalized_token_posterior_count=len(all_sense_probabilities))
        mark(45, retained_alternative_count=sum(len(token["candidates"]) for token in token_semantics))
        mark(46, mean_sense_entropy=float(np.mean([token["posterior_entropy"] for token in token_semantics])))
        mark(47, mean_top_two_margin=float(np.mean([token["posterior_margin"] for token in token_semantics])))
        mark(48, mean_known_concept_mass=float(np.mean([
            token["known_concept_probability_mass"] for token in token_semantics
        ])))
        mark(49, mean_context_conflict=float(np.mean(conflict_values)) if conflict_values else 0.0)
        mark(50, reliability_adjustment_applied=True)
        mark(51, confident_token_count=sum(token["status"] == "RESOLVED" for token in token_semantics))
        mark(52, unknown_and_abstention_gates_applied=True)
        mark(53, semantic_state_count=len(token_semantics))
        mark(54, alternative_preservation_count=sum(len(token["candidates"]) for token in token_semantics))

        # G — explicit semantic graph and composition (55–63)
        raw_relations = list(request.relation_evidence or ())
        if len(raw_relations) > cfg.max_relations:
            raise ValueError("relation evidence exceeds max_relations")
        relations = [_as_relation(item) for item in raw_relations]
        mark(55, relation_count=len(relations))
        for relation in relations:
            source = relation["source_token_index"]
            target = relation["target_token_index"]
            if not (0 <= source < len(token_semantics) and 0 <= target < len(token_semantics)):
                raise ValueError("relation evidence references a token index outside the N6 sequence")
        mark(56, endpoints_valid=True)
        for relation in relations:
            if cfg.require_provenance and (
                not relation["verified_by"].strip() or not relation["source_reference"].strip()
            ):
                raise ValueError("relation evidence requires verified_by and source_reference")
        mark(57, relation_provenance_checked=True)
        graph_edges: list[dict[str, Any]] = []
        for relation in relations:
            source = token_semantics[relation["source_token_index"]]
            target = token_semantics[relation["target_token_index"]]
            graph_edges.append({
                "source_token_id": source["token_id"],
                "target_token_id": target["token_id"],
                "source_concept_id": source["selected_concept_id"],
                "target_concept_id": target["selected_concept_id"],
                "relation": relation["relation"],
                "input_confidence": relation["confidence"],
                "source_verified_by": relation["verified_by"],
                "source_reference": relation["source_reference"],
            })
        mark(58, relation_endpoint_bindings=len(graph_edges))
        token_by_id = {token["token_id"]: token for token in token_semantics}
        for edge in graph_edges:
            source_record = token_by_id[edge["source_token_id"]]
            target_record = token_by_id[edge["target_token_id"]]
            edge["confidence"] = float(edge["input_confidence"]
                                        * source_record["selected_probability"]
                                        * target_record["selected_probability"])
            edge["supported"] = bool(
                source_record["selected_concept_id"] is not None
                and target_record["selected_concept_id"] is not None
                and edge["confidence"] >= cfg.minimum_relation_confidence
            )
            edge["status"] = "SUPPORTED" if edge["supported"] else "UNRESOLVED_ENDPOINT_OR_LOW_CONFIDENCE"
        mark(59, supported_edge_count=sum(edge["supported"] for edge in graph_edges))
        graph_nodes = [{
            "token_id": token["token_id"], "token_index": token["token_index"],
            "orthography": token["orthography"], "concept_id": token["selected_concept_id"],
            "sense_id": token["selected_sense_id"], "status": token["status"],
            "confidence": token["selected_probability"],
        } for token in token_semantics]
        mark(60, graph_node_count=len(graph_nodes))
        mark(61, graph_edge_count=len(graph_edges), inferred_edges=0)

        # Connected evidence components become frames; no missing relation is inferred.
        semantic_node_indices = [index for index, node in enumerate(graph_nodes)
                                 if node["concept_id"] is not None]
        parent = list(range(len(graph_nodes)))
        def find_root(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value
        token_index_by_id = {token["token_id"]: token["token_index"] for token in token_semantics}
        for edge in graph_edges:
            if edge["supported"]:
                left = find_root(token_index_by_id[edge["source_token_id"]])
                right = find_root(token_index_by_id[edge["target_token_id"]])
                if left != right:
                    parent[right] = left
        components: dict[int, list[int]] = defaultdict(list)
        for index in semantic_node_indices:
            components[find_root(index)].append(index)
        frames: list[dict[str, Any]] = []
        for frame_index, indices in enumerate(sorted(components.values(), key=lambda xs: xs[0]), start=1):
            token_ids = {graph_nodes[index]["token_id"] for index in indices}
            frame_edges = [edge for edge in graph_edges if edge["supported"]
                           and edge["source_token_id"] in token_ids
                           and edge["target_token_id"] in token_ids]
            frame_confidence = float(np.mean([graph_nodes[index]["confidence"] for index in indices]))
            frames.append({
                "frame_id": f"{sequence_id}:frame:{frame_index}",
                "token_ids": [graph_nodes[index]["token_id"] for index in indices],
                "concept_ids": [graph_nodes[index]["concept_id"] for index in indices],
                "relations": frame_edges,
                "confidence": frame_confidence,
                "composition_status": "EXPLICIT_RELATION_COMPONENT" if frame_edges else "UNCOMPOSED_CONCEPT",
            })
        mark(62, compositional_frame_count=len(frames))
        complete_frames = sum(bool(frame["relations"]) for frame in frames)
        mean_frame_confidence = float(np.mean([frame["confidence"] for frame in frames])) if frames else 0.0
        mark(63, connected_frames=complete_frames, mean_frame_confidence=mean_frame_confidence)

        # H — aggregation, final verification and return (64–72)
        activations: dict[str, float] = defaultdict(float)
        for token in token_semantics:
            for candidate in token["candidates"]:
                if candidate["concept_id"] is not None:
                    activations[candidate["concept_id"]] += float(candidate["probability"])
        mark(64, activated_concept_count=len(activations))
        total_activation = float(sum(activations.values()))
        normalized_activations = {
            key: value / total_activation for key, value in activations.items()
        } if total_activation > cfg.epsilon else {}
        concept_ranking = [
            {"concept_id": key, "activation": float(activations[key]),
             "probability": float(normalized_activations[key])}
            for key in sorted(activations)
        ]
        concept_ranking.sort(key=lambda row: (-row["probability"], row["concept_id"]))
        mark(65, concept_probability_sum=float(sum(normalized_activations.values())))
        preserved_evidence["n6_sequence_id"] = sequence_id
        preserved_evidence["n7_sequence_id"] = str(_get(n7, "sequence_id", ""))
        preserved_evidence["n8_token_semantic_status"] = [token["status"] for token in token_semantics]
        mark(66, preserved_channels=list(preserved_evidence))
        sequence_confidence = float(np.mean([
            token["selected_probability"] * token["token_reliability"]
            if token["status"] == "RESOLVED" else 0.0
            for token in token_semantics
        ]))
        sequence_entropy = float(np.mean([token["normalized_entropy"] for token in token_semantics]))
        mark(67, sequence_confidence=sequence_confidence, sequence_entropy=sequence_entropy)
        concept_ranking.sort(key=lambda row: (-row["probability"], row["concept_id"]))
        frames.sort(key=lambda frame: (-frame["confidence"], frame["frame_id"]))
        mark(68, concept_count=len(concept_ranking), frame_count=len(frames))
        token_report = [{
            "token_id": item["token_id"], "orthography": item["orthography"],
            "status": item["status"], "selected_concept_id": item["selected_concept_id"],
            "selected_sense_id": item["selected_sense_id"],
            "selected_probability": item["selected_probability"],
            "candidates": item["candidates"],
        } for item in token_semantics]
        mark(69, token_report_count=len(token_report))
        n6_after = {
            "sequence_id": str(_get(n6, "sequence_id", "")), "metadata": dict(_get(n6, "metadata", {}) or {}),
            "tokens": _plain(_get(n6, "tokens", [])),
            "decoded_graphemes": _plain(_get(n6, "decoded_graphemes", ())),
            "boundary_probabilities": _plain(_get(n6, "boundary_probabilities", [])),
        }
        n7_after = {
            "sequence_id": str(_get(n7, "sequence_id", "")), "metadata": dict(_get(n7, "metadata", {}) or {}),
            "posterior": _plain(_get(n7, "posterior", [])),
            "ranked_hypotheses": _plain(_get(n7, "ranked_hypotheses", [])),
            "evidence": _plain(n7_evidence),
            "association_ranking": _plain(_get(n7, "association_ranking", [])),
        }
        upstream_unchanged = (
            _fingerprint(n6_after) == initial_n6_fingerprint
            and _fingerprint(n7_after) == initial_n7_fingerprint
        )
        trace_is_ordered = len(trace) == 69 and all(
            record["operation"] == index for index, record in enumerate(trace, start=1)
        )
        diagnostics = {
            "valid": bool(upstream_unchanged and trace_is_ordered),
            "stage_count": 72,
            "executed_operation_count": 72,
            "operation_trace_complete": False,
            "token_count": len(token_semantics),
            "semantic_lexicon_entry_count": len(entries),
            "lexical_coverage": lexical_coverage,
            "resolved_token_count": sum(item["status"] == "RESOLVED" for item in token_semantics),
            "ambiguous_token_count": sum(item["status"] == "AMBIGUOUS" for item in token_semantics),
            "unknown_token_count": sum(item["status"] == "UNKNOWN_OR_UNMAPPED" for item in token_semantics),
            "abstained_token_count": sum(item["status"] == "ABSTAIN_LOW_CONFIDENCE" for item in token_semantics),
            "relation_count": len(graph_edges),
            "supported_relation_count": sum(edge["supported"] for edge in graph_edges),
            "compositional_frame_count": len(frames),
            "mean_context_conflict": float(np.mean(conflict_values)) if conflict_values else 0.0,
            "sequence_semantic_confidence": sequence_confidence,
            "mean_normalized_sense_entropy": sequence_entropy,
            "upstream_inputs_unchanged": upstream_unchanged,
            "n6_fingerprint": initial_n6_fingerprint,
            "n7_fingerprint": initial_n7_fingerprint,
            "semantic_lexicon_fingerprint": lexicon_fingerprint,
            "scientific_acceptance": "PENDING_INDEPENDENT_EMPIRICAL_VALIDATION",
            "warnings": [
                "Semantic mappings are restricted to caller-supplied provenance-checked entries.",
                "No syntax/dependency relation is inferred unless explicit relation evidence is supplied.",
                "Software tests do not establish biological validity or semantic correctness.",
            ],
        }
        mark(70, diagnostics_fields=len(diagnostics), provenance_fields=4)
        mark(71, upstream_unchanged=upstream_unchanged, ordered_prior_trace=trace_is_ordered,
             resolved_or_unknown_states=True, graph_edges_explicit_only=True)
        trace.append({"operation": 72, "name": N8_ALGORITHM_STEPS[71], "status": "completed",
                      "details": {"result_returned": True}})
        diagnostics["operation_trace_complete"] = len(trace) == 72
        diagnostics["valid"] = bool(diagnostics["valid"] and diagnostics["operation_trace_complete"])
        metadata = {
            "schema": "ore.n8.lexical_semantic_composition.v1",
            "algorithm": "verified lexical lookup, N7-conditioned sense selection, explicit relation composition",
            "sequence_id": sequence_id,
            "processing_mode": str(request.processing_mode),
            "upstream_schemas": [n6_metadata["schema"], n7_metadata["schema"]],
            "semantic_resource_policy": "caller_supplied_exact_form_entries_with_verifier_and_source_provenance",
            "relation_policy": "only_explicit_provenance_checked_relation_evidence_is_composed",
            "evidence_separation": ["N6 visual/orthographic evidence", "N7 contextual posterior", "N8 semantic hypotheses"],
            "configuration": cfg.to_dict(),
            "elapsed_seconds": float(perf_counter() - started),
            "limitations": [
                "N8 is an algorithmic lexical-semantic approximation, not a biological circuit simulation.",
                "Exact-form lookup does not implement morphology, subword semantics, or transliteration.",
                "Sense scores require calibration against independently annotated data.",
                "Relation edges are consumed from an external provider; N8 does not parse syntax.",
                "A supplied verifier/source reference provides provenance, not proof of truth.",
            ],
        }
        return N8Result(
            sequence_id=sequence_id,
            token_semantics=token_semantics,
            intent_posterior={key: float(intent_mass[key]) for key in sorted(intent_mass)},
            domain_posterior={key: float(domain_mass[key]) for key in sorted(domain_mass)},
            concept_activations={key: float(activations[key]) for key in sorted(activations)},
            concept_ranking=concept_ranking,
            semantic_graph={"nodes": graph_nodes, "edges": graph_edges},
            compositional_frames=frames,
            preserved_evidence=preserved_evidence,
            operation_trace=trace,
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def process(self, request: N8Input) -> N8Result:
        """Execute N8; alias naming is consistent with the earlier ORE stages."""
        return self._run(request)

    def infer(self, request: N8Input) -> N8Result:
        """Explicit inference alias for callers that prefer recognition terminology."""
        return self._run(request)
