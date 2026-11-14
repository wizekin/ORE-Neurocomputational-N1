"""The 120-operation ORE N9 integration controller.

The controller coordinates existing N1-N4 stages when supplied and exposes
explicit extension contracts for N5-N8. Missing stages remain marked as
unavailable. All inference is read-only; optional learning creates a detached
child model only after explicit authorization.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    N9Adapters,
    N9Config,
    N9Input,
    N9Patch,
    N9Request,
    N9Result,
)

N9_ALGORITHM_STEPS: tuple[str, ...] = (
    # Part A — input orchestration and system-state initialization (1–12)
    "validate_incoming_observation_contract",
    "identify_input_pathway_and_script_profile",
    "preserve_raw_observation_and_source_identity",
    "construct_missing_data_and_occlusion_masks",
    "establish_spatial_and_temporal_coordinates",
    "estimate_initial_observation_reliability",
    "initialize_separated_latent_state",
    "validate_upstream_module_contract_compatibility",
    "snapshot_read_only_model_resources",
    "encode_task_and_operating_mode",
    "load_context_and_working_memory_snapshots",
    "build_dependency_graph_and_schedule_branches",
    # Part B — early visual processing and visual organization (13–24)
    "convert_or_retrieve_luminance_and_ink_signal",
    "compute_local_center_surround_contrast",
    "normalize_local_contrast",
    "extract_orientation_selective_energy",
    "separate_spatial_frequency_bands",
    "detect_edge_contour_and_junction_evidence",
    "pool_local_features_into_shape_descriptors",
    "construct_visual_population_features",
    "retain_alternative_region_segmentation_hypotheses",
    "estimate_figure_ground_and_stroke_grouping",
    "align_visual_fragments_with_candidate_regions",
    "emit_n2_visual_evidence_package",
    # Part C — position coding, orthographic populations and sequence (25–36)
    "create_script_specific_grapheme_vocabulary",
    "compute_distributed_grapheme_activations",
    "estimate_novelty_and_unknown_probability",
    "build_absolute_position_code",
    "build_relative_position_code",
    "compute_position_overlap_and_competition",
    "preserve_identity_and_position_channels",
    "establish_script_aware_logical_order",
    "load_read_only_transition_memory",
    "generate_top_k_sequence_candidate_paths",
    "align_sequence_candidates_with_word_boundaries",
    "decompose_visual_and_contextual_evidence",
    # Part D — attention, precision and resource allocation (37–48)
    "measure_grapheme_entropy",
    "measure_orthographic_prediction_error",
    "derive_task_and_goal_salience",
    "construct_attention_logits",
    "allocate_competitive_attention",
    "smooth_and_floor_attention",
    "estimate_sensory_precision",
    "estimate_prior_precision",
    "compute_relative_precision_weights",
    "form_precision_weighted_error_signals",
    "pass_attention_metadata_to_downstream_modules",
    "prepare_feedback_triggered_attention_reallocation",
    # Part E — sequence decoding and lexical/phonological interfaces (49–60)
    "initialize_recurrent_sequence_beliefs",
    "generate_forward_sequence_predictions",
    "generate_backward_sequence_predictions",
    "combine_visual_and_bidirectional_predictions",
    "invoke_n6_lexical_access_adapter",
    "retrieve_morphological_hypotheses",
    "retrieve_phonological_form_hypotheses",
    "combine_lexical_and_phonological_routes",
    "rank_word_and_sense_candidates",
    "apply_lexical_abstention_rule",
    "refine_word_boundary_alignment",
    "emit_n6_interface_package",
    # Part F — syntax, sentence structure and semantic composition (61–72)
    "extract_part_of_speech_and_morphosyntactic_features",
    "construct_syntactic_hypotheses",
    "load_explicit_grammar_constraints",
    "invoke_syntax_parser_adapter",
    "assign_supported_thematic_roles",
    "compose_hierarchical_phrase_representations",
    "resolve_explicit_negation_modality_and_quantifier_scope",
    "represent_connective_and_discourse_relations",
    "resolve_coreference_only_with_evidence",
    "construct_predicate_argument_propositions",
    "construct_and_validate_typed_semantic_graph",
    "measure_semantic_completeness_and_ambiguity",
    # Part G — context, discourse and situation model (73–84)
    "load_discourse_context",
    "retrieve_read_only_semantic_memory",
    "retrieve_explicit_episodic_document_context",
    "invoke_n7_intent_and_domain_adapter",
    "measure_proposition_context_compatibility",
    "generate_context_conditioned_upcoming_predictions",
    "link_temporal_and_causal_relations",
    "construct_situation_model",
    "measure_discourse_coherence",
    "detect_world_knowledge_conflicts",
    "quantify_contextual_uncertainty",
    "emit_contextual_state_snapshot",
    # Part H — hierarchical feedback and conflict resolution (85–96)
    "generate_hierarchical_top_down_predictions",
    "project_semantic_predictions_to_lexical_candidates",
    "project_lexical_predictions_to_grapheme_candidates",
    "request_predicted_visual_feature_evidence",
    "compute_level_specific_prediction_errors",
    "compute_precision_weighted_prediction_errors",
    "evaluate_global_free_energy_proxy",
    "perform_bounded_posterior_revision",
    "detect_cross_channel_conflicts",
    "schedule_targeted_visual_reprocessing",
    "measure_feedback_convergence",
    "preserve_channel_specific_evidence",
    # Part I — global integration, decision and output (97–108)
    "integrate_orthographic_lexical_syntactic_semantic_graphs",
    "construct_joint_candidate_hypotheses",
    "evaluate_cross_level_consistency",
    "aggregate_candidate_evidence_and_conflicts",
    "calibrate_level_specific_confidence",
    "choose_answer_ambiguous_or_abstain",
    "build_structured_integrated_result",
    "build_task_specific_output_representation",
    "generate_evidence_linked_explanations",
    "rollback_invalid_or_unsupported_revisions",
    "estimate_resource_and_computational_costs",
    "run_final_integrity_and_provenance_checks",
    # Part J — controlled learning and scientific acceptance (109–120)
    "freeze_inference_state_before_learning",
    "detect_candidate_learning_opportunities",
    "require_explicit_learning_authorization",
    "propose_update_on_detached_child_model",
    "estimate_transition_statistics_without_mutation",
    "identify_repeated_patterns_for_mdl_review",
    "apply_homeostatic_and_anti_forgetting_constraints",
    "record_learning_provenance_and_version",
    "prepare_ablation_and_component_isolation_plan",
    "record_real_corpus_validation_requirements",
    "separate_software_task_and_biological_acceptance",
    "finalize_integrated_n9_result",
)

if len(N9_ALGORITHM_STEPS) != 120 or len(set(N9_ALGORITHM_STEPS)) != 120:
    raise RuntimeError("N9 operation manifest must contain 120 unique operations")


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exp = np.exp(np.clip(shifted, -80.0, 80.0))
    return (exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)).astype(np.float32)


def _normalize_rows(values: np.ndarray, epsilon: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array = np.maximum(array, 0.0)
    totals = array.sum(axis=1, keepdims=True)
    zero = totals[:, 0] <= epsilon
    if np.any(zero):
        array[zero] = 1.0 / max(1, array.shape[1])
        totals = array.sum(axis=1, keepdims=True)
    return (array / np.maximum(totals, epsilon)).astype(np.float32)


def _entropy_rows(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    return (-np.sum(p * np.log(np.maximum(p, epsilon)), axis=1)).astype(np.float32)


def _digest(value: Any) -> str:
    """Stable digest for arrays and JSON-like provenance values."""
    h = sha256()

    def visit(item: Any) -> None:
        if isinstance(item, np.ndarray):
            arr = np.ascontiguousarray(item)
            h.update(str(arr.dtype).encode())
            h.update(repr(arr.shape).encode())
            h.update(arr.tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=lambda x: str(x)):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(b"[")
            for part in item:
                visit(part)
            h.update(b"]")
        elif isinstance(item, (str, int, float, bool)) or item is None:
            h.update(json.dumps(item, sort_keys=True, allow_nan=False).encode("utf-8"))
        else:
            h.update(repr(item).encode("utf-8", errors="replace"))

    visit(value)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return repr(value)


def _module_fingerprint(module: Any | None) -> str | None:
    if module is None:
        return None
    for attr in ("fingerprint", "model_fingerprint"):
        value = getattr(module, attr, None)
        if isinstance(value, str) and value:
            return value
    pieces: dict[str, Any] = {
        "class": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
        "version": getattr(module, "__version__", None),
    }
    for attr in ("config", "class_names", "transition_model", "visual_population"):
        if hasattr(module, attr):
            value = getattr(module, attr)
            if attr == "config":
                pieces[attr] = repr(value)
            elif attr == "class_names":
                pieces[attr] = list(value)
            elif attr == "transition_model" and value is not None:
                pieces[attr] = getattr(value, "fingerprint", repr(value))
            elif attr == "visual_population" and value is not None:
                pieces[attr] = getattr(value, "fingerprint", repr(value))
    return _digest(pieces)


def _call_adapter(adapter: Any, payload: Mapping[str, Any], names: Sequence[str]) -> Any:
    if adapter is None:
        return None
    if callable(adapter):
        result = adapter(payload)
    else:
        method = next((getattr(adapter, name) for name in names if callable(getattr(adapter, name, None))), None)
        if method is None:
            raise TypeError(f"adapter must be callable or expose one of {tuple(names)}")
        result = method(payload)
    if result is None:
        return {}
    if not isinstance(result, Mapping):
        raise TypeError("N9 extension adapters must return a mapping")
    return dict(result)


def _entropy(probability: np.ndarray, epsilon: float) -> float:
    p = np.asarray(probability, dtype=np.float64)
    return float(-np.sum(p * np.log(np.maximum(p, epsilon))))


class N9Processor:
    """Run the integrated 120-operation N9 controller.

    Image patches use the repository's N1 and N2 processors. A fitted N3
    processor and N4 processor are optional. N5-N8 are injected explicitly:
    N5 may only propose a detached child model; N6 resolves lexical candidates;
    N7 supplies contextual predictions; N8 composes semantic structures.
    """

    def __init__(
        self,
        config: N9Config | None = None,
        adapters: N9Adapters | None = None,
    ) -> None:
        self.config = config or N9Config()
        self.adapters = adapters or N9Adapters()

    def process(self, request: N9Request) -> N9Result:
        started = time.perf_counter()
        if not isinstance(request, N9Request):
            raise TypeError("request must be an N9Request")
        cfg = self.config
        trace: list[dict[str, Any]] = []
        warnings: list[str] = []
        state: dict[str, Any] = {}
        operation_cursor = 0

        def mark(number: int, status: str = "ok", **details: Any) -> None:
            nonlocal operation_cursor
            expected = operation_cursor + 1
            if number != expected:
                raise RuntimeError(f"N9 operation order error: expected {expected}, received {number}")
            operation_cursor = number
            trace.append({
                "step": number,
                "operation": N9_ALGORITHM_STEPS[number - 1],
                "status": status,
                **_jsonable(details),
            })

        # A. Operations 1–12: input, provenance and execution state.
        mark(1, patch_count=len(request.patches), sequence_id=request.sequence_id)
        profile = {
            "script": request.script_profile.get("script", "unspecified"),
            "reading_direction": request.script_profile.get("reading_direction", "ltr"),
            "grapheme_policy": request.script_profile.get("grapheme_policy", "caller_defined"),
            **dict(request.script_profile),
        }
        mark(2, script=profile["script"], reading_direction=profile["reading_direction"])
        source_digest = _digest({
            "sequence_id": request.sequence_id,
            "source_id": request.source_id,
            "patches": [
                {
                    "patch_id": p.patch_id,
                    "image": p.image,
                    "features": p.features,
                    "probabilities": dict(p.grapheme_probabilities or {}),
                    "n2": getattr(p.n2_result, "metadata", None),
                    "source_metadata": dict(p.source_metadata),
                }
                for p in request.patches
            ],
        })
        mark(3, source_digest=source_digest)
        masks: list[np.ndarray] = []
        for patch in request.patches:
            if patch.valid_mask is not None:
                mask = np.asarray(patch.valid_mask, dtype=bool)
            elif patch.image is not None:
                mask = np.ones(np.asarray(patch.image).shape, dtype=bool)
            else:
                mask = np.asarray([bool(patch.observed)], dtype=bool)
            masks.append(mask)
        visible_fraction = np.asarray(
            [float(np.mean(mask)) if patch.observed else 0.0
             for patch, mask in zip(request.patches, masks)],
            dtype=np.float32,
        )
        mark(4, visible_fraction=visible_fraction)
        raw_x = np.asarray([
            float(p.x) if p.x is not None else float(i)
            for i, p in enumerate(request.patches)
        ], dtype=np.float32)
        raw_y = np.asarray([
            float(p.y) if p.y is not None else 0.0
            for p in request.patches
        ], dtype=np.float32)
        denom = float(np.ptp(raw_x))
        normalized_x = (raw_x - float(np.min(raw_x))) / (denom if denom > cfg.epsilon else 1.0)
        positions = np.column_stack((normalized_x, raw_y, np.arange(len(request.patches)))).astype(np.float32)
        mark(5, coordinate_count=len(positions), input_order_preserved=True)
        reliability = np.clip(
            np.asarray([p.segmentation_confidence for p in request.patches], dtype=np.float32)
            * visible_fraction,
            0.0, 1.0,
        )
        mark(6, mean_reliability=float(np.mean(reliability)))
        state.update({
            "visual_features": [],
            "n1_results": [],
            "n2_results": [],
            "grapheme_beliefs": None,
            "lexical_hypotheses": [],
            "syntax": {},
            "semantic_graph": {"nodes": [], "edges": []},
            "propositions": [],
            "context": {},
            "errors": {},
        })
        mark(7, separated_channels=["visual", "orthographic", "lexical", "phonological", "syntax", "semantic", "context"])
        adapter_status = {}
        for name in ("n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "syntax"):
            adapter = getattr(self.adapters, name)
            adapter_status[name] = "available" if adapter is not None else "not_configured"
        if self.adapters.n3 is not None and getattr(self.adapters.n3, "_fitted", True) is False:
            raise RuntimeError("configured N3Processor is not fitted; call fit() before N9 processing")
        mark(8, adapters=adapter_status)
        model_before = {
            name: _module_fingerprint(getattr(self.adapters, name))
            for name in ("n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "syntax")
        }
        mark(9, model_fingerprints=model_before)
        task_state = {"task": request.task, "learning_authorized": bool(request.learning_authorized)}
        mark(10, task=request.task)
        context_snapshot = deepcopy(dict(request.context))
        mark(11, context_keys=sorted(str(k) for k in context_snapshot))
        dependency_graph = {
            "visual": ["input"],
            "orthographic": ["visual"],
            "attention": ["orthographic"],
            "lexical": ["orthographic", "attention"],
            "syntax": ["lexical"],
            "semantic": ["syntax", "lexical"],
            "discourse": ["semantic", "context"],
            "feedback": ["orthographic", "lexical", "semantic", "discourse"],
            "decision": ["feedback"],
            "learning": ["decision"],
        }
        mark(12, dependency_nodes=len(dependency_graph), recurrent_edges="bounded feedback loop")

        # Prepare existing N1/N2 modules without mutating upstream inputs.
        n1_processor = self.adapters.n1
        n2_processor = self.adapters.n2
        if n1_processor is None:
            from ore_n1 import N1Processor
            n1_processor = N1Processor()
        if n2_processor is None:
            from ore_n2 import N2Processor
            n2_processor = N2Processor()

        prepared: list[dict[str, Any]] = []
        for patch in request.patches:
            n1_result = None
            n2_result = patch.n2_result
            feature = None if patch.features is None else np.asarray(patch.features, dtype=np.float32).copy()
            if patch.image is not None:
                n1_result = n1_processor.process(np.asarray(patch.image))
                n2_result = n2_processor.process(n1_result)
            if n2_result is not None and feature is None:
                from ore_n3.visual import feature_vector_from_n2
                feature = feature_vector_from_n2(n2_result)
            prepared.append({
                "patch": patch,
                "n1": n1_result,
                "n2": n2_result,
                "features": feature,
                "image": None if patch.image is None else np.asarray(patch.image),
            })
            state["n1_results"].append(n1_result)
            state["n2_results"].append(n2_result)

        # B. Operations 13–24: visual evidence and retained segmentation alternatives.
        luminance_records = []
        for item in prepared:
            image = item["image"]
            if image is not None:
                luminance_records.append({
                    "mean": float(np.mean(image)),
                    "std": float(np.std(image)),
                    "range": float(np.max(image) - np.min(image)),
                })
            else:
                luminance_records.append(None)
        mark(13, status="ok" if any(v is not None for v in luminance_records) else "not_applicable",
             image_patch_count=sum(v is not None for v in luminance_records))
        center_surround = [
            None if item["n1"] is None else {
                "on": item["n1"].stage_outputs["ganglion_on"],
                "off": item["n1"].stage_outputs["ganglion_off"],
            }
            for item in prepared
        ]
        mark(14, status="ok" if any(v is not None for v in center_surround) else "not_applicable")
        normalized_maps = [
            None if item["n1"] is None else item["n1"].stage_outputs["normalized_energy"]
            for item in prepared
        ]
        mark(15, status="ok" if any(v is not None for v in normalized_maps) else "not_applicable")
        oriented_energy = [
            None if item["n1"] is None else item["n1"].stage_outputs["complex_cell_energy"]
            for item in prepared
        ]
        mark(16, status="ok" if any(v is not None for v in oriented_energy) else "not_applicable")
        frequency_bands = []
        for item in prepared:
            n2 = item["n2"]
            frequency_bands.append(
                n2.representations.get("multiscale") if n2 is not None else None
            )
        mark(17, status="ok" if any(v is not None for v in frequency_bands) else "not_applicable")
        contours = [
            n2.representations.get("contour") if n2 is not None else None
            for n2 in (item["n2"] for item in prepared)
        ]
        mark(18, status="ok" if any(v is not None for v in contours) else "not_applicable")
        shape_descriptors = []
        for item in prepared:
            feature = item["features"]
            if feature is not None:
                shape_descriptors.append(feature.copy())
            elif item["n2"] is not None:
                shape_descriptors.append(np.asarray(item["n2"].representations["ventral"]).mean(axis=(1, 2)))
            else:
                shape_descriptors.append(np.zeros(1, dtype=np.float32))
        mark(19, descriptor_dimension=int(max((v.size for v in shape_descriptors), default=0)))
        state["visual_features"] = shape_descriptors
        mark(20, feature_source=[
            "N2_descriptor" if item["n2"] is not None else
            "caller_features" if item["features"] is not None else
            "probability_only"
            for item in prepared
        ])
        segmentation_hypotheses = [{
            "patch_id": p.patch_id,
            "start": i,
            "end": i + 1,
            "confidence": float(p.segmentation_confidence),
            "source": "caller_localized_patch",
        } for i, p in enumerate(request.patches)]
        mark(21, status="ok", auto_segmentation=False, hypothesis_count=len(segmentation_hypotheses))
        n = len(request.patches)
        grouping = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                distance = float(np.linalg.norm(positions[i, :2] - positions[j, :2]))
                grouping[i, j] = math.exp(-distance)
        mark(22, grouping_shape=grouping.shape)
        alignments = [{
            "patch_id": item["patch"].patch_id,
            "feature_available": item["features"] is not None,
            "region_index": i,
        } for i, item in enumerate(prepared)]
        mark(23, alignment_count=len(alignments))
        mark(24, evidence_channels=["raw_observation", "N1_visual", "N2_structural", "caller_grapheme"])
        state.update({
            "luminance": luminance_records,
            "center_surround": center_surround,
            "normalized_maps": normalized_maps,
            "oriented_energy": oriented_energy,
            "frequency_bands": frequency_bands,
            "contours": contours,
            "segmentation_hypotheses": segmentation_hypotheses,
        })

        # C. Operations 25–36: orthographic probabilities and sequence paths.
        direct_probabilities = [p.grapheme_probabilities for p in request.patches]
        n3_result = None
        if all(value is not None for value in direct_probabilities):
            class_names = tuple(sorted({str(k) for row in direct_probabilities for k in row}))
            raw = np.zeros((n, len(class_names)), dtype=np.float32)
            for i, row in enumerate(direct_probabilities):
                for key, value in row.items():
                    raw[i, class_names.index(str(key))] = float(value)
            visual_probabilities = _normalize_rows(raw, cfg.epsilon)
            mark(25, vocabulary_size=len(class_names), source="caller_probability_contract")
            mark(26, status="ok", rows=n)
        elif self.adapters.n3 is not None and all(
            item["n2"] is not None or item["features"] is not None for item in prepared
        ):
            from ore_n3 import N3Patch, N3Sequence
            n3_patches = [
                N3Patch(
                    patch_id=item["patch"].patch_id,
                    n2_result=item["n2"],
                    features=item["features"] if item["n2"] is None else None,
                    segmentation_confidence=float(item["patch"].segmentation_confidence),
                )
                for item in prepared
            ]
            n3_result = self.adapters.n3.recognize(N3Sequence(request.sequence_id, n3_patches))
            class_names = tuple(n3_result.class_names)
            visual_probabilities = _normalize_rows(n3_result.visual_probabilities, cfg.epsilon)
            mark(25, vocabulary_size=len(class_names), source="fitted_N3_processor")
            mark(26, status="ok", rows=n)
        else:
            class_names = ("<UNK>",)
            visual_probabilities = np.ones((n, 1), dtype=np.float32)
            mark(25, vocabulary_size=1, source="explicit_unknown_fallback")
            mark(26, status="abstention_fallback", reason="no grapheme distribution or fitted N3 adapter")
        if visual_probabilities.shape[0] != n:
            raise ValueError("grapheme distribution row count must equal patch count")
        mark(27, unknown_mass=float(np.mean([
            float(row[class_names.index("<UNK>")]) if "<UNK>" in class_names else 0.0
            for row in visual_probabilities
        ])))
        absolute_positions = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            absolute_positions[i, i] = 1.0
        mark(28, absolute_code_shape=absolute_positions.shape)
        relative_positions = positions[:, 0:1] - positions[:, 0][None, :]
        mark(29, relative_code_shape=relative_positions.shape)
        feature_matrix = np.zeros((n, max(1, max(v.size for v in shape_descriptors))), dtype=np.float32)
        for i, descriptor in enumerate(shape_descriptors):
            feature_matrix[i, :descriptor.size] = descriptor
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        normalized_features = feature_matrix / np.maximum(norms, cfg.epsilon)
        overlap = normalized_features @ normalized_features.T
        mark(30, overlap_shape=overlap.shape)
        orthographic_channel = visual_probabilities.copy()
        position_channel = {"absolute": absolute_positions, "relative": relative_positions}
        mark(31, channels_independent=True)
        mark(32, logical_order="caller_input_order", reading_direction=profile["reading_direction"])
        transition_model = getattr(self.adapters.n3, "transition_model", None) if self.adapters.n3 else None
        transition_fingerprint = getattr(transition_model, "fingerprint", None)
        mark(33, status="ok" if transition_model is not None else "not_configured",
             transition_fingerprint=transition_fingerprint)
        beam: list[tuple[float, list[str]]] = [(1.0, [])]
        for row in visual_probabilities:
            top_indices = np.argsort(row)[::-1][:min(cfg.top_k, len(class_names))]
            expanded = [
                (score * float(row[index]), path + [class_names[int(index)]])
                for score, path in beam for index in top_indices
            ]
            beam = sorted(expanded, key=lambda pair: pair[0], reverse=True)[:cfg.beam_width]
        sequence_paths = [
            {"symbols": path, "score": float(score)}
            for score, path in beam
        ]
        mark(34, path_count=len(sequence_paths), beam_width=cfg.beam_width)
        spans = list(request.word_spans) if request.word_spans is not None else [(0, n)]
        covered = {idx for start, end in spans for idx in range(start, end)}
        spans.extend((idx, idx + 1) for idx in range(n) if idx not in covered)
        spans = sorted(spans)
        mark(35, word_span_count=len(spans))
        evidence_decomposition = {
            "visual_probabilities": visual_probabilities.copy(),
            "position_code": position_channel,
            "transition_model_available": transition_model is not None,
            "n3_result_available": n3_result is not None,
        }
        mark(36, evidence_sources=list(evidence_decomposition))
        state.update({
            "n3_result": n3_result,
            "class_names": class_names,
            "visual_probabilities": visual_probabilities,
            "orthographic_channel": orthographic_channel,
            "position_channel": position_channel,
            "transition_model": transition_model,
            "sequence_paths": sequence_paths,
            "word_spans": spans,
        })

        # D. Operations 37–48: attention and precision allocation.
        entropy = _entropy_rows(visual_probabilities, cfg.epsilon)
        normalized_entropy = entropy / max(math.log(max(2, len(class_names))), cfg.epsilon)
        mark(37, mean_entropy=float(np.mean(entropy)))
        if n3_result is not None:
            n3_error = np.asarray(n3_result.classwise_prediction_error, dtype=np.float32)
            error_signal = np.mean(np.abs(n3_error), axis=1)
        else:
            error_signal = normalized_entropy.copy()
        error_signal = error_signal / max(float(np.max(error_signal)), cfg.epsilon)
        mark(38, mean_error=float(np.mean(error_signal)))
        task_relevance = np.asarray(
            context_snapshot.get("task_relevance", np.zeros(n)), dtype=np.float32
        )
        if task_relevance.shape != (n,) or not np.isfinite(task_relevance).all():
            task_relevance = np.zeros(n, dtype=np.float32)
            warnings.append("invalid task_relevance ignored; zero relevance used")
        task_relevance = np.clip(task_relevance, 0.0, 1.0)
        bottom_up_salience = np.clip(0.5 * normalized_entropy + 0.5 * error_signal, 0, 1)
        attention_logits = bottom_up_salience + task_relevance
        mark(39, mean_salience=float(np.mean(bottom_up_salience)))
        mark(40, mean_logit=float(np.mean(attention_logits)))
        attention = _softmax(attention_logits / 0.7, axis=0).reshape(-1)
        mark(41, attention_sum=float(attention.sum()))
        uniform = np.full(n, 1.0 / n, dtype=np.float32)
        attention = 0.90 * attention + 0.10 * uniform
        if n > 2:
            smoothed = attention.copy()
            smoothed[1:-1] = 0.15 * attention[:-2] + 0.70 * attention[1:-1] + 0.15 * attention[2:]
            attention = smoothed / max(float(smoothed.sum()), cfg.epsilon)
        mark(42, attention_floor=float(np.min(attention)))
        sensory_precision = np.clip(
            (0.5 + 0.5 * reliability) * (1.0 + n * attention),
            0.25, 4.0,
        ).astype(np.float32)
        mark(43, mean_sensory_precision=float(np.mean(sensory_precision)))
        n4_result = None
        if n3_result is not None:
            from ore_n4 import N4Input, N4Processor
            n4_processor = self.adapters.n4 or N4Processor()
            n4_result = n4_processor.process(N4Input(
                n3_result,
                task_relevance=task_relevance,
                segmentation_confidence=reliability,
            ))
            attention = np.asarray(n4_result.attention_allocation, dtype=np.float32).copy()
            sensory_precision = np.asarray(n4_result.sensory_precision, dtype=np.float32).copy()
            prior_precision = np.asarray(n4_result.prior_precision, dtype=np.float32).copy()
            attended_beliefs = np.asarray(n4_result.posterior_beliefs, dtype=np.float32).copy()
            mark(44, status="ok", source="N4Processor")
        else:
            prior_precision = np.clip(0.5 + normalized_entropy, 0.25, 3.0).astype(np.float32)
            attended_beliefs = visual_probabilities.copy()
            mark(44, status="not_configured", source="N9_bounded_fallback")
        mark(45, attention_sum=float(attention.sum()))
        class_gate = np.ones_like(visual_probabilities, dtype=np.float32)
        if n3_result is not None:
            class_error = np.abs(np.asarray(n3_result.classwise_prediction_error, dtype=np.float32))
            scale = np.max(class_error, axis=1, keepdims=True)
            class_gate = np.clip(1.0 + class_error / np.maximum(scale, cfg.epsilon), 0.5, 2.0)
        mark(46, class_gate_shape=class_gate.shape)
        relative_precision = sensory_precision / np.maximum(prior_precision, cfg.epsilon)
        mark(47, mean_relative_precision=float(np.mean(relative_precision)))
        attention_metadata = {
            "attention": attention.copy(),
            "sensory_precision": sensory_precision.copy(),
            "prior_precision": prior_precision.copy(),
            "n4_available": n4_result is not None,
        }
        mark(48, downstream_metadata=["attention", "sensory_precision", "prior_precision"])
        state.update({
            "attention": attention,
            "sensory_precision": sensory_precision,
            "prior_precision": prior_precision,
            "class_gate": class_gate,
            "attended_beliefs": attended_beliefs,
            "attention_metadata": attention_metadata,
        })

        # E. Operations 49–60: recurrent sequence, lexical and phonological interfaces.
        beliefs = _normalize_rows(attended_beliefs, cfg.epsilon)
        mark(49, belief_shape=beliefs.shape)
        forward_prediction = np.zeros_like(beliefs)
        reverse_prediction = np.zeros_like(beliefs)
        if transition_model is not None:
            unigram = np.asarray(transition_model.unigram, dtype=np.float32)
            for i in range(n):
                forward_prediction[i] = unigram[:len(class_names)] if unigram.size >= len(class_names) else 1.0
                reverse_prediction[i] = forward_prediction[i]
            if hasattr(transition_model, "forward") and n > 1:
                for i in range(1, n):
                    previous = int(np.argmax(beliefs[i - 1]))
                    forward_prediction[i] = transition_model.forward[previous, :len(class_names)]
            if hasattr(transition_model, "reverse") and n > 1:
                for i in range(n - 1):
                    following = int(np.argmax(beliefs[i + 1]))
                    reverse_prediction[i] = transition_model.reverse[following, :len(class_names)]
            forward_prediction = _normalize_rows(forward_prediction, cfg.epsilon)
            reverse_prediction = _normalize_rows(reverse_prediction, cfg.epsilon)
        else:
            forward_prediction = beliefs.copy()
            reverse_prediction = beliefs.copy()
        mark(50, status="ok" if transition_model is not None else "not_configured")
        mark(51, status="ok" if transition_model is not None else "not_configured")
        bidirectional_prior = _normalize_rows(0.5 * forward_prediction + 0.5 * reverse_prediction, cfg.epsilon)
        proposal = _softmax(
            sensory_precision[:, None] * np.log(np.maximum(visual_probabilities, cfg.epsilon))
            + prior_precision[:, None] * np.log(np.maximum(bidirectional_prior, cfg.epsilon)),
            axis=1,
        )
        mark(52, mean_prior_confidence=float(np.mean(np.max(bidirectional_prior, axis=1))))
        lexical_payload = {
            "sequence_id": request.sequence_id,
            "class_names": list(class_names),
            "grapheme_probabilities": proposal.copy(),
            "patch_ids": [p.patch_id for p in request.patches],
            "word_spans": spans,
            "context": context_snapshot,
            "task": request.task,
        }
        n6_output = _call_adapter(self.adapters.n6, lexical_payload, ("resolve", "process", "predict"))
        if n6_output is not None:
            lexical_hypotheses = list(n6_output.get("lexical_hypotheses", n6_output.get("words", [])))
            if not lexical_hypotheses:
                lexical_hypotheses = []
            mark(53, status="ok", candidate_count=len(lexical_hypotheses))
        else:
            lexical_hypotheses = []
            for start, end in spans:
                local_rows = proposal[start:end]
                indices = np.argmax(local_rows, axis=1)
                symbols = [class_names[int(idx)] for idx in indices]
                form = "".join(symbol for symbol in symbols if symbol != "<UNK>")
                confidence = float(np.prod(np.max(local_rows, axis=1)))
                lexical_hypotheses.append({
                    "form": form,
                    "start": start,
                    "end": end,
                    "score": confidence,
                    "source": "surface_form_fallback",
                    "semantics_verified": False,
                })
            mark(53, status="fallback", candidate_count=len(lexical_hypotheses),
                 reason="N6 lexical adapter not configured")
        morphology = list(n6_output.get("morphology", [])) if n6_output else []
        mark(54, status="ok" if morphology else "not_configured", morphology_count=len(morphology))
        phonology = list(n6_output.get("phonological_forms", [])) if n6_output else []
        mark(55, status="ok" if phonology else "not_configured", phonological_form_count=len(phonology))
        combined_lexical = {
            "lexical_hypotheses": lexical_hypotheses,
            "morphology": morphology,
            "phonological_forms": phonology,
            "route_sources": ["visual_orthographic"] + (["lexical_adapter"] if n6_output else []),
        }
        mark(56, route_sources=combined_lexical["route_sources"])
        lexical_hypotheses = sorted(
            lexical_hypotheses,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        mark(57, ranked_count=len(lexical_hypotheses))
        lexical_confidence = float(lexical_hypotheses[0].get("score", 0.0)) if lexical_hypotheses else 0.0
        lexical_abstain = lexical_confidence < cfg.confidence_threshold
        mark(58, status="abstain" if lexical_abstain else "ok",
             top_score=lexical_confidence, threshold=cfg.confidence_threshold)
        mark(59, word_span_alignment=spans)
        state.update({
            "beliefs": beliefs,
            "forward_prediction": forward_prediction,
            "reverse_prediction": reverse_prediction,
            "lexical_hypotheses": lexical_hypotheses,
            "morphology": morphology,
            "phonology": phonology,
            "n6_output": n6_output or {},
            "lexical_abstain": lexical_abstain,
        })
        mark(60, status="ok", lexical_count=len(lexical_hypotheses))

        # F. Operations 61–72: explicit syntax and compositional semantics.
        pos_features = list(
            (n6_output or {}).get("pos_features", (n6_output or {}).get("morphosyntactic_features", []))
        )
        mark(61, status="ok" if pos_features else "not_configured", feature_count=len(pos_features))
        syntax_payload = {
            "lexical_hypotheses": lexical_hypotheses,
            "pos_features": pos_features,
            "grammar": context_snapshot.get("grammar", {}),
            "context": context_snapshot,
        }
        syntax_output = _call_adapter(self.adapters.syntax, syntax_payload, ("parse", "process", "predict"))
        if syntax_output is None and n6_output:
            syntax_output = dict(n6_output.get("syntax", {}))
        syntax_output = syntax_output or {}
        mark(62, status="ok" if syntax_output else "not_configured",
             parse_count=len(syntax_output.get("parses", [])))
        grammar = dict(context_snapshot.get("grammar", {}))
        mark(63, grammar_rule_count=len(grammar.get("rules", [])) if isinstance(grammar, Mapping) else 0)
        mark(64, status="ok" if syntax_output.get("parses") else "not_configured")
        thematic_roles = list(syntax_output.get("thematic_roles", []))
        mark(65, status="ok" if thematic_roles else "not_configured", role_count=len(thematic_roles))
        phrase_representations = list(syntax_output.get("phrases", []))
        mark(66, status="ok" if phrase_representations else "not_configured")
        scope = dict(syntax_output.get("scope", {}))
        mark(67, status="ok" if scope else "not_configured")
        discourse_relations = list(syntax_output.get("discourse_relations", []))
        mark(68, status="ok" if discourse_relations else "not_configured")
        coreference = list(syntax_output.get("coreference", []))
        mark(69, status="ok" if coreference else "not_configured")
        semantic_payload = {
            "lexical_hypotheses": lexical_hypotheses,
            "syntax": syntax_output,
            "morphology": morphology,
            "context": context_snapshot,
        }
        n8_output = _call_adapter(self.adapters.n8, semantic_payload, ("compose", "process", "resolve"))
        if n8_output is None and n6_output:
            candidate_semantics = n6_output.get("semantics")
            n8_output = dict(candidate_semantics) if isinstance(candidate_semantics, Mapping) else None
        if n8_output is not None:
            graph = dict(n8_output.get("semantic_graph", {"nodes": [], "edges": []}))
            propositions = list(n8_output.get("propositions", []))
        else:
            graph = {"nodes": [], "edges": []}
            propositions = []
        mark(70, status="ok" if propositions else "not_configured", proposition_count=len(propositions))
        graph = self._validate_semantic_graph(graph)
        mark(71, node_count=len(graph["nodes"]), edge_count=len(graph["edges"]))
        semantic_completeness = float(n8_output.get("completeness", 0.0)) if n8_output else 0.0
        semantic_ambiguity = float(n8_output.get("ambiguity", 1.0)) if n8_output else 1.0
        mark(72, completeness=semantic_completeness, ambiguity=semantic_ambiguity)
        state.update({
            "syntax": syntax_output,
            "thematic_roles": thematic_roles,
            "scope": scope,
            "discourse_relations": discourse_relations,
            "coreference": coreference,
            "semantic_graph": graph,
            "propositions": propositions,
            "semantic_completeness": semantic_completeness,
        })

        # G. Operations 73–84: discourse and situation state.
        discourse_context = {
            "prior_turns": context_snapshot.get("prior_turns", []),
            "document_context": context_snapshot.get("document_context", {}),
            "domain": context_snapshot.get("domain"),
            "topic": context_snapshot.get("topic"),
        }
        mark(73, context_fields=[k for k, v in discourse_context.items() if v not in (None, {}, [])])
        semantic_memory = dict(context_snapshot.get("semantic_memory", {}))
        mark(74, memory_entry_count=len(semantic_memory))
        episodic_context = dict(context_snapshot.get("episodic_context", {}))
        mark(75, episodic_context_available=bool(episodic_context))
        n7_payload = {
            "propositions": propositions,
            "semantic_graph": graph,
            "lexical_hypotheses": lexical_hypotheses,
            "discourse_context": discourse_context,
        }
        n7_output = _call_adapter(self.adapters.n7, n7_payload, ("predict", "process", "infer"))
        if n7_output is None:
            n7_output = {}
        mark(76, status="ok" if n7_output else "not_configured",
             intent_count=len(n7_output.get("intents", [])))
        compatibility = n7_output.get("proposition_compatibility", [])
        mark(77, compatibility_count=len(compatibility))
        temporal_links = list(n7_output.get("temporal_links", []))
        causal_links = list(n7_output.get("causal_links", []))
        mark(78, temporal_link_count=len(temporal_links), causal_link_count=len(causal_links))
        situation_model = dict(n7_output.get("situation_model", {}))
        mark(79, situation_model_available=bool(situation_model))
        coherence = float(n7_output.get("coherence", 0.0))
        mark(80, coherence=coherence)
        conflicts = list(n7_output.get("knowledge_conflicts", []))
        mark(81, conflict_count=len(conflicts))
        context_uncertainty = float(n7_output.get("uncertainty", 1.0 if not n7_output else 0.5))
        context_uncertainty = float(np.clip(context_uncertainty, 0.0, 1.0))
        mark(82, uncertainty=context_uncertainty)
        contextual_state = {
            **discourse_context,
            "intents": n7_output.get("intents", []),
            "predictions": n7_output.get("predictions", {}),
            "situation_model": situation_model,
            "coherence": coherence,
            "uncertainty": context_uncertainty,
            "temporal_links": temporal_links,
            "causal_links": causal_links,
            "knowledge_conflicts": conflicts,
        }
        mark(83, state_fields=len(contextual_state))
        mark(84, status="ok" if n7_output else "fallback", source="N7_adapter" if n7_output else "caller_context")
        state["context"] = contextual_state

        # H. Operations 85–96: bounded top-down feedback, never replacing visual evidence.
        semantic_predictions = dict(n7_output.get("lexical_predictions", {}))
        if n8_output:
            semantic_predictions.update(n8_output.get("lexical_predictions", {}))
        mark(85, prediction_count=len(semantic_predictions))
        lexical_feedback = np.zeros_like(visual_probabilities)
        for i, row in enumerate(lexical_feedback):
            lexical_feedback[i] = 0.0
        for i, patch in enumerate(request.patches):
            for symbol, probability in semantic_predictions.items():
                if symbol in class_names:
                    lexical_feedback[i, class_names.index(symbol)] += float(probability)
        if lexical_feedback.sum() > cfg.epsilon:
            lexical_feedback = _normalize_rows(lexical_feedback + cfg.epsilon, cfg.epsilon)
        mark(86, feedback_mass=float(lexical_feedback.sum()))
        lexical_char_feedback = np.zeros_like(visual_probabilities)
        for hypothesis in lexical_hypotheses:
            form = str(hypothesis.get("form", ""))
            start = int(hypothesis.get("start", 0))
            end = int(hypothesis.get("end", min(n, start + len(form))))
            for offset, char in enumerate(form[:max(0, end - start)]):
                index = start + offset
                if 0 <= index < n and char in class_names:
                    lexical_char_feedback[index, class_names.index(char)] += float(hypothesis.get("score", 0.0))
        if lexical_char_feedback.sum() > cfg.epsilon:
            lexical_char_feedback = _normalize_rows(lexical_char_feedback + cfg.epsilon, cfg.epsilon)
        mark(87, feedback_mass=float(lexical_char_feedback.sum()))
        predicted_visual_features = n8_output.get("predicted_visual_features") if n8_output else None
        if predicted_visual_features is not None:
            predicted_visual_features = np.asarray(predicted_visual_features, dtype=np.float32)
            if not np.isfinite(predicted_visual_features).all():
                predicted_visual_features = None
                warnings.append("non-finite predicted visual features discarded")
        mark(88, status="ok" if predicted_visual_features is not None else "not_configured")
        level_errors = {
            "visual_to_posterior": np.abs(visual_probabilities - proposal),
            "lexical_feedback": np.abs(proposal - lexical_char_feedback)
            if lexical_char_feedback.sum() > 0 else np.zeros_like(proposal),
        }
        mark(89, mean_visual_error=float(np.mean(level_errors["visual_to_posterior"])))
        weighted_errors = {
            "visual": level_errors["visual_to_posterior"] * sensory_precision[:, None],
            "lexical": level_errors["lexical_feedback"] * prior_precision[:, None],
        }
        mark(90, mean_weighted_visual_error=float(np.mean(weighted_errors["visual"])))
        free_energy = float(np.mean(
            -np.sum(proposal * np.log(np.maximum(visual_probabilities, cfg.epsilon)), axis=1)
            + cfg.context_gain * np.sum(
                proposal * np.log(np.maximum(proposal, cfg.epsilon)
                                  / np.maximum(bidirectional_prior, cfg.epsilon)),
                axis=1,
            )
        ))
        mark(91, free_energy_proxy=free_energy)
        posterior = proposal.copy()
        feedback_distribution = lexical_char_feedback if lexical_char_feedback.sum() > cfg.epsilon else bidirectional_prior
        feedback_trace = []
        converged = False
        mean_change = float("inf")
        for iteration in range(1, cfg.max_feedback_iterations + 1):
            old = posterior.copy()
            logits = (
                np.log(np.maximum(visual_probabilities, cfg.epsilon))
                + cfg.feedback_gain * np.log(np.maximum(feedback_distribution, cfg.epsilon))
                + cfg.context_gain * np.log(np.maximum(bidirectional_prior, cfg.epsilon))
            )
            proposed = _softmax(logits, axis=1)
            posterior = _normalize_rows(
                (1.0 - cfg.damping) * posterior + cfg.damping * proposed,
                cfg.epsilon,
            )
            mean_change = float(np.mean(np.abs(posterior - old)))
            feedback_trace.append({"iteration": iteration, "mean_change": mean_change})
            if mean_change <= 1e-5:
                converged = True
                break
        mark(92, iterations=len(feedback_trace), converged=converged, mean_change=mean_change)
        def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
            mid = 0.5 * (left + right)
            kl_l = np.sum(left * np.log(np.maximum(left, cfg.epsilon) / np.maximum(mid, cfg.epsilon)), axis=1)
            kl_r = np.sum(right * np.log(np.maximum(right, cfg.epsilon) / np.maximum(mid, cfg.epsilon)), axis=1)
            return float(np.mean(0.5 * (kl_l + kl_r)))
        channel_conflict = js_divergence(visual_probabilities, posterior)
        mark(93, mean_jensen_shannon_conflict=channel_conflict)
        reprocess_indices = [
            int(i) for i, (uncertainty, error) in enumerate(zip(normalized_entropy, error_signal))
            if uncertainty > 0.8 or error > 0.8
        ]
        mark(94, reprocess_patch_indices=reprocess_indices, automatic_reprocessing=False)
        mark(95, feedback_converged=converged, trace_length=len(feedback_trace))
        channel_snapshot = {
            "visual": visual_probabilities.copy(),
            "orthographic": orthographic_channel.copy(),
            "posterior": posterior.copy(),
            "contextual_prior": bidirectional_prior.copy(),
        }
        mark(96, retained_channels=list(channel_snapshot))
        state.update({
            "posterior": posterior,
            "free_energy": free_energy,
            "feedback_trace": feedback_trace,
            "channel_conflict": channel_conflict,
            "reprocess_indices": reprocess_indices,
            "channel_snapshot": channel_snapshot,
        })

        # I. Operations 97–108: integrated decision and output.
        integrated_graph = {
            "orthographic_sequence": sequence_paths[:cfg.top_k],
            "lexical_hypotheses": lexical_hypotheses,
            "syntax": syntax_output,
            "semantic_graph": graph,
            "propositions": propositions,
            "context": contextual_state,
        }
        mark(97, graph_components=list(integrated_graph))
        joint_candidates = []
        for hypothesis in lexical_hypotheses[:cfg.top_k]:
            joint_candidates.append({
                **dict(hypothesis),
                "semantic_support": semantic_completeness,
                "context_support": float(np.clip(coherence, 0, 1)),
            })
        mark(98, joint_candidate_count=len(joint_candidates))
        syntax_score = float(syntax_output.get("consistency", 0.0))
        semantic_score = float(np.clip(semantic_completeness, 0.0, 1.0))
        context_score = float(np.clip(coherence, 0.0, 1.0))
        consistency = float(np.mean([syntax_score, semantic_score, context_score]))
        mark(99, consistency=consistency)
        evidence_support = {
            "visual": float(np.mean(np.max(visual_probabilities, axis=1))),
            "attended": float(np.mean(np.max(posterior, axis=1))),
            "lexical": lexical_confidence,
            "semantic": semantic_score,
            "context": context_score,
        }
        mark(100, support=evidence_support)
        posterior_confidence = float(np.mean(np.max(posterior, axis=1)))
        sorted_means = np.sort(np.mean(posterior, axis=0))[::-1]
        candidate_margin = float(sorted_means[0] - sorted_means[1]) if len(sorted_means) > 1 else posterior_confidence
        calibrated_confidence = float(np.clip(
            0.70 * posterior_confidence + 0.20 * lexical_confidence + 0.10 * (1.0 - context_uncertainty),
            0.0, 1.0,
        ))
        mark(101, posterior_confidence=posterior_confidence, calibrated_confidence=calibrated_confidence,
             candidate_margin=candidate_margin)
        if not lexical_hypotheses or calibrated_confidence < cfg.confidence_threshold:
            decision_status = "ABSTAIN"
        elif candidate_margin < cfg.ambiguity_margin:
            decision_status = "AMBIGUOUS"
        else:
            decision_status = "CANDIDATE"
        best_form = str(lexical_hypotheses[0].get("form", "")) if lexical_hypotheses else None
        decision = {
            "status": decision_status,
            "candidate": best_form if decision_status == "CANDIDATE" else None,
            "top_hypotheses": joint_candidates,
            "confidence": calibrated_confidence,
            "margin": candidate_margin,
        }
        mark(102, decision=decision_status)
        structured_result = {
            "sequence_id": request.sequence_id,
            "decision": decision,
            "graphemes": [
                {"patch_id": patch.patch_id,
                 "alternatives": [
                     {"symbol": class_names[int(index)], "probability": float(posterior[i, index])}
                     for index in np.argsort(posterior[i])[::-1][:cfg.top_k]
                 ]}
                for i, patch in enumerate(request.patches)
            ],
        }
        mark(103, output_fields=list(structured_result))
        rendered_text = " ".join(
            str(item.get("form", "")) for item in lexical_hypotheses if item.get("form")
        )
        task_output = {
            "task": request.task,
            "text": rendered_text if request.task in {"read", "transcribe", "recognize"} else None,
            "candidate": best_form,
            "status": decision_status,
        }
        mark(104, rendered_text_available=bool(rendered_text))
        explanation = {
            "decision_status": decision_status,
            "primary_evidence": "preserved visual/orthographic posterior",
            "context_used": bool(n7_output),
            "semantic_relations_used": bool(graph["edges"]),
            "warnings": list(warnings),
            "limitations": [
                "N9 is a computational integration proposal, not a validated neural circuit model",
                "unconfigured N5-N8 adapters are not treated as executed stages",
                "surface-form fallback is not lexical or semantic verification",
            ],
        }
        mark(105, explanation_sections=len(explanation))
        integrity_ok = bool(np.isfinite(posterior).all() and np.allclose(posterior.sum(axis=1), 1.0, atol=1e-5))
        if not integrity_ok and cfg.strict_integrity:
            posterior = visual_probabilities.copy()
            decision_status = "ABSTAIN"
            decision.update({"status": "ABSTAIN", "candidate": None})
            warnings.append("posterior rollback to preserved visual evidence after integrity failure")
        mark(106, rollback_applied=not integrity_ok)
        resource_costs = {
            "patch_count": n,
            "class_count": len(class_names),
            "feedback_iterations": len(feedback_trace),
            "feature_scalar_count": int(feature_matrix.size),
            "operation_count": 108,
            "elapsed_seconds_so_far": time.perf_counter() - started,
        }
        mark(107, resource_costs=resource_costs)
        final_finite = (
            np.isfinite(visual_probabilities).all()
            and np.isfinite(posterior).all()
            and np.isfinite(attention).all()
            and np.allclose(visual_probabilities.sum(axis=1), 1.0, atol=1e-5)
            and np.allclose(posterior.sum(axis=1), 1.0, atol=1e-5)
            and np.isclose(float(attention.sum()), 1.0, atol=1e-5)
        )
        if not final_finite:
            raise FloatingPointError("N9 final integrity checks failed")
        mark(108, integrity_valid=True, source_digest=source_digest)

        # J. Operations 109–120: inference freeze, opt-in learning and acceptance.
        frozen_inference_digest = _digest({
            "visual": visual_probabilities,
            "posterior": posterior,
            "class_names": class_names,
            "source_digest": source_digest,
        })
        mark(109, inference_frozen=True, digest=frozen_inference_digest)
        learning_opportunities = {
            "high_uncertainty_patch_indices": [int(i) for i, value in enumerate(normalized_entropy) if value > 0.7],
            "unknown_patch_indices": [
                int(i) for i, row in enumerate(visual_probabilities)
                if "<UNK>" in class_names and row[class_names.index("<UNK>")] > 0.5
            ],
            "cross_channel_conflict": channel_conflict,
        }
        mark(110, opportunity_count=sum(len(v) for v in learning_opportunities.values() if isinstance(v, list)))
        mark(111, status="authorized" if request.learning_authorized else "not_authorized")
        learning_result: dict[str, Any] = {
            "status": "not_requested",
            "parent_model_mutated": False,
            "child_model_created": False,
        }
        if request.learning_authorized and self.adapters.n5 is not None:
            parent_before = _module_fingerprint(self.adapters.n5)
            proposal_result = _call_adapter(
                self.adapters.n5,
                {
                    "operation": "propose_detached_child_update",
                    "parent_fingerprint": parent_before,
                    "learning_opportunities": learning_opportunities,
                    "transition_statistics_source": sequence_paths,
                    "frozen_inference_digest": frozen_inference_digest,
                },
                ("propose_update", "train_child", "process"),
            ) or {}
            parent_after = _module_fingerprint(self.adapters.n5)
            parent_unchanged = parent_before == parent_after
            child = proposal_result.get("child_model")
            child_digest = _module_fingerprint(child) if child is not None else proposal_result.get("child_fingerprint")
            learning_result = {
                "status": "child_proposed" if child is not None and parent_unchanged else "rejected",
                "parent_model_mutated": not parent_unchanged,
                "child_model_created": child is not None and parent_unchanged,
                "parent_fingerprint_before": parent_before,
                "parent_fingerprint_after": parent_after,
                "child_fingerprint": child_digest,
                "adapter_metadata": proposal_result.get("metadata", {}),
            }
            if not parent_unchanged:
                warnings.append("N5 adapter changed its parent model; proposal rejected")
        elif request.learning_authorized:
            learning_result["status"] = "authorized_but_N5_unavailable"
        else:
            learning_result["status"] = "skipped_without_explicit_authorization"
        mark(112, learning_status=learning_result["status"])
        transition_statistics: dict[str, int] = {}
        for path in sequence_paths:
            symbols = path["symbols"]
            for left, right in zip(symbols, symbols[1:]):
                key = f"{left}→{right}"
                transition_statistics[key] = transition_statistics.get(key, 0) + 1
        mark(113, transition_pair_count=len(transition_statistics))
        mdl_candidates = [
            {"pattern": key, "count": count}
            for key, count in sorted(transition_statistics.items(), key=lambda item: (-item[1], item[0]))
            if count > 1
        ]
        mark(114, candidate_pattern_count=len(mdl_candidates), promotion_requires_review=True)
        regularization = {
            "parent_model_frozen": learning_result.get("parent_model_mutated") is False,
            "automatic_forgetting": False,
            "normalization_check": final_finite,
        }
        mark(115, regularization=regularization)
        provenance = {
            "source_id": request.source_id,
            "source_digest": source_digest,
            "model_fingerprints_before": model_before,
            "upstream_stages": adapter_status,
            "script_profile": profile,
            "sequence_id": request.sequence_id,
        }
        mark(116, provenance_fields=len(provenance))
        ablation_plan = [
            "disable_visual_to_orthographic_link",
            "disable_position_and_transition_priors",
            "disable_attention_and_precision_gating",
            "disable_lexical_adapter",
            "disable_syntactic_composition",
            "disable_semantic_composition",
            "disable_discourse_context",
            "disable_top_down_feedback",
        ]
        mark(117, ablation_count=len(ablation_plan))
        validation_requirements = {
            "synthetic_software_tests": "required_and_executable",
            "real_corpus_validation": "pending_external_evaluation",
            "writer_font_script_disjoint_splits": "required",
            "independent_replication": "required",
        }
        mark(118, real_corpus_status="PENDING_EXTERNAL_EVIDENCE")
        acceptance = {
            "software_contracts": "passed" if final_finite and operation_cursor == 118 else "failed",
            "task_accuracy": "not_established_without_held_out_corpus",
            "biological_validity": "not_established",
            "independent_validation": "pending",
        }
        mark(119, acceptance=acceptance)
        mark(120, status="complete", final_decision=decision_status)
        if operation_cursor != 120 or len(trace) != 120:
            raise RuntimeError("N9 must execute and record exactly 120 operations")

        model_after = {
            name: _module_fingerprint(getattr(self.adapters, name))
            for name in ("n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "syntax")
        }
        changed_models = [
            name for name in model_before
            if model_before[name] != model_after[name]
            and name != "n5"
        ]
        if changed_models:
            message = f"inference adapters changed during N9 processing: {changed_models}"
            if cfg.strict_integrity:
                raise RuntimeError(message)
            warnings.append(message)
        provenance["model_fingerprints_after"] = model_after

        diagnostics = {
            "valid": bool(final_finite),
            "stage_count": 120,
            "executed_operation_count": len(trace),
            "operation_status_counts": {
                status: sum(1 for item in trace if item["status"] == status)
                for status in sorted({item["status"] for item in trace})
            },
            "feedback": {
                "converged": converged,
                "iterations": len(feedback_trace),
                "trace": feedback_trace,
            },
            "mean_normalized_entropy": float(np.mean(normalized_entropy)),
            "mean_attention": float(np.mean(attention)),
            "attention_sum": float(attention.sum()),
            "free_energy_proxy": free_energy,
            "channel_conflict": channel_conflict,
            "warnings": warnings,
            "elapsed_seconds": time.perf_counter() - started,
            "scientific_status": "software prototype; empirical and biological validation pending",
        }
        metadata = {
            "pipeline": "ORE N9 Integrated Neurocomputational ORE",
            "version": "0.1.0",
            "operation_count": 120,
            "subsystem_count": 10,
            "patch_count": n,
            "class_count": len(class_names),
            "feature_shapes": [list(value.shape) for value in feature_matrix.shape] if False else list(feature_matrix.shape),
            "task_output": task_output,
            "integrated_graph": integrated_graph,
            "structured_result": structured_result,
            "ablation_plan": ablation_plan,
            "validation_requirements": validation_requirements,
            "acceptance": acceptance,
            "learning_opportunities": learning_opportunities,
            "transition_statistics": transition_statistics,
            "mdl_candidates": mdl_candidates,
            "resource_costs": resource_costs,
            "explanation": explanation,
            "evidence_decomposition": {
                "visual_mass": float(visual_probabilities.sum()),
                "posterior_mass": float(posterior.sum()),
                "visual_is_preserved_separately": True,
            },
        }
        return N9Result(
            sequence_id=request.sequence_id,
            class_names=class_names,
            visual_probabilities=visual_probabilities.copy(),
            posterior_beliefs=posterior.astype(np.float32, copy=True),
            attention_allocation=attention.astype(np.float32, copy=True),
            lexical_hypotheses=lexical_hypotheses,
            syntax=syntax_output,
            semantic_graph=graph,
            propositions=propositions,
            context_model=contextual_state,
            decision=decision,
            operation_trace=trace,
            diagnostics=diagnostics,
            provenance=provenance,
            learning=learning_result,
            metadata=metadata,
        )

    @staticmethod
    def _validate_semantic_graph(value: Mapping[str, Any]) -> dict[str, Any]:
        """Validate adapter-returned typed graph edges; discard dangling relations."""
        raw_nodes = list(value.get("nodes", []))
        nodes: list[dict[str, Any]] = []
        node_ids: set[str] = set()
        for index, raw in enumerate(raw_nodes):
            if not isinstance(raw, Mapping):
                continue
            node_id = str(raw.get("id", "")).strip()
            node_type = str(raw.get("type", "")).strip()
            if not node_id or not node_type or node_id in node_ids:
                continue
            node_ids.add(node_id)
            nodes.append(dict(raw))
        edges: list[dict[str, Any]] = []
        for raw in value.get("edges", []):
            if not isinstance(raw, Mapping):
                continue
            source = str(raw.get("source", ""))
            target = str(raw.get("target", ""))
            relation = str(raw.get("relation", "")).strip()
            if source in node_ids and target in node_ids and relation:
                edges.append(dict(raw))
        return {"nodes": nodes, "edges": edges}


__all__ = ["N9_ALGORITHM_STEPS", "N9Processor"]
