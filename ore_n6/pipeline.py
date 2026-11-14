"""The complete 60-operation ORE N6 hierarchical orthographic-access algorithm."""
from __future__ import annotations

from collections import defaultdict
from math import log
from time import perf_counter
from typing import Any

import numpy as np

from .config import N6Config
from .contracts import N6Input, N6LexiconEntry, N6Result
from .stages import (
    entropy_rows,
    k_best_paths,
    logit,
    n5_fingerprint,
    normalize_rows,
    stable_softmax,
)

UNKNOWN_LABEL = "<UNK>"

N6_ALGORITHM_STEPS: tuple[str, ...] = (
    "01_validate_N6_request_contract",
    "02_validate_N4_result_interface",
    "03_validate_N5_model_interface",
    "04_validate_sequence_identity",
    "05_validate_grapheme_vocabulary",
    "06_validate_patch_posterior_dimensions",
    "07_normalize_and_preserve_N4_patch_posteriors",
    "08_validate_attention_precision_and_upstream_diagnostics",
    "09_validate_N5_model_schema_version",
    "10_align_N4_and_N5_grapheme_vocabularies",
    "11_validate_N5_feature_normalization_state",
    "12_validate_fast_and_slow_prototype_banks",
    "13_validate_fast_and_slow_transition_matrices",
    "14_validate_all_N5_numeric_state",
    "15_recompute_and_verify_N5_fingerprint",
    "16_build_grapheme_to_class_index",
    "17_combine_fast_and_slow_forward_transition_memory",
    "18_combine_fast_and_slow_reverse_transition_memory",
    "19_smooth_and_normalize_forward_transition_probabilities",
    "20_smooth_and_normalize_reverse_transition_probabilities",
    "21_consolidate_fast_and_slow_unigram_exposure_memory",
    "22_derive_N5_population_class_prior",
    "23_compute_bounded_experience_dependent_class_gains",
    "24_calibrate_N4_evidence_with_N5_gains_and_optional_prototypes",
    "25_convert_calibrated_evidence_to_log_likelihoods",
    "26_estimate_left_to_right_contextual_support",
    "27_estimate_right_to_left_contextual_support",
    "28_combine_bidirectional_contextual_support",
    "29_integrate_context_and_visual_evidence",
    "30_normalize_context_conditioned_patch_posteriors",
    "31_construct_symmetric_pairwise_transition_scores",
    "32_initialize_sequence_Viterbi_state",
    "33_propagate_recurrent_sequence_hypotheses",
    "34_backtrace_best_grapheme_sequence",
    "35_compute_forward_sequence_messages",
    "36_compute_backward_sequence_messages",
    "37_estimate_forward_backward_grapheme_marginals",
    "38_fuse_marginals_with_context_conditioned_evidence",
    "39_normalize_final_grapheme_posteriors",
    "40_measure_positionwise_orthographic_uncertainty",
    "41_measure_sequence_transition_coherence",
    "42_construct_ranked_positionwise_grapheme_candidates",
    "43_validate_optional_word_boundary_evidence",
    "44_validate_lexicon_entry_provenance_and_labels",
    "45_index_verified_orthographic_lexicon_entries",
    "46_match_lexicon_entries_to_grapheme_spans",
    "47_compute_lexical_emission_and_transition_evidence",
    "48_estimate_boundary_affinity_from_cues_and_lexicon_support",
    "49_add_unknown_grapheme_fallback_edges",
    "50_score_word_and_fallback_segmentation_edges",
    "51_run_k_best_hierarchical_segmentation_dynamic_program",
    "52_normalize_segmentation_alternative_scores",
    "53_select_highest_scoring_orthographic_segmentation",
    "54_bind_grapheme_positions_to_word_level_units",
    "55_compute_token_and_word_form_confidence",
    "56_apply_lexical_abstention_and_unknown_handling",
    "57_assemble_token_lattice_and_sequence_alternatives",
    "58_verify_position_coverage_and_N5_model_immutability",
    "59_build_diagnostics_operation_trace_and_provenance",
    "60_return_structured_N6_orthographic_access_result",
)


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    finite_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    summed = np.sum(np.exp(array - finite_maximum), axis=axis, keepdims=True)
    result = finite_maximum + np.log(np.maximum(summed, 1e-300))
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    return result


def _validate_probability_vector(
    name: str, value: Any, expected: int, epsilon: float
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (expected,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector with shape ({expected},)")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError(f"{name} values must lie in [0, 1]")
    return np.clip(array, epsilon, 1.0 - epsilon).astype(np.float32)


def _as_lexicon_entry(item: Any) -> dict[str, Any]:
    if isinstance(item, N6LexiconEntry):
        values = {
            "entry_id": item.entry_id,
            "graphemes": item.graphemes,
            "frequency": item.frequency,
            "verified_by": item.verified_by,
            "source_reference": item.source_reference,
            "language": item.language,
        }
    elif isinstance(item, dict):
        values = dict(item)
    else:
        raise TypeError("lexicon_entries must contain N6LexiconEntry objects or dictionaries")

    for field_name in ("entry_id", "verified_by", "source_reference"):
        value = values.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"lexicon {field_name} is required")
    graphemes = values.get("graphemes")
    if not isinstance(graphemes, (tuple, list)) or not graphemes:
        raise ValueError("lexicon graphemes must be a non-empty tuple or list")
    graphemes = tuple(graphemes)
    if any(not isinstance(item, str) or not item for item in graphemes):
        raise ValueError("lexicon graphemes must be non-empty strings")
    frequency = float(values.get("frequency", 1.0))
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("lexicon frequency must be finite and positive")
    language = values.get("language", "und")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("lexicon language must be a non-empty string")
    return {
        "entry_id": values["entry_id"],
        "graphemes": graphemes,
        "frequency": frequency,
        "verified_by": values["verified_by"],
        "source_reference": values["source_reference"],
        "language": language,
    }


class N6Processor:
    """Decode an N4 patch sequence into grapheme and word-form hypotheses.

    N6 is read-only with respect to N4 and N5. It uses an optional externally
    verified orthographic lexicon; it does not infer word meaning, silently add
    labels, or modify upstream model parameters.
    """

    def __init__(self, config: N6Config | None = None) -> None:
        self.config = config or N6Config()

    def process(self, request: N6Input) -> N6Result:
        started = perf_counter()
        cfg = self.config
        trace: list[dict[str, Any]] = []
        warnings: list[str] = []

        def mark(step: int, **summary: Any) -> None:
            trace.append({
                "step": step,
                "operation": N6_ALGORITHM_STEPS[step - 1],
                **summary,
            })

        # 01-08: N4 input and upstream integrity.
        if not isinstance(request, N6Input):
            raise TypeError("request must be an N6Input")
        mark(1, status="valid")
        n4 = request.n4_result
        n5 = request.n5_model
        if n4 is None or any(
            not hasattr(n4, name)
            for name in ("sequence_id", "class_names", "posterior_beliefs")
        ):
            raise TypeError("n4_result must expose sequence_id, class_names, and posterior_beliefs")
        mark(2, status="valid")
        required_n5 = (
            "class_names", "feature_mean", "feature_scale", "fast_prototypes",
            "slow_prototypes", "fast_eligibility", "slow_eligibility", "class_gains",
            "fast_unigram_counts", "slow_unigram_counts", "fast_forward_counts",
            "slow_forward_counts", "fast_reverse_counts", "slow_reverse_counts",
            "to_dict", "fingerprint", "schema_version", "version",
        )
        missing_n5 = [name for name in required_n5 if not hasattr(n5, name)]
        if n5 is None or missing_n5:
            raise TypeError(f"n5_model is missing required N5 fields: {missing_n5}")
        mark(3, status="valid")

        sequence_id = n4.sequence_id
        if not isinstance(sequence_id, str) or not sequence_id.strip():
            raise ValueError("N4 sequence_id must be a non-empty string")
        mark(4, sequence_id=sequence_id)

        class_names = tuple(n4.class_names)
        if (
            not class_names
            or any(not isinstance(name, str) or not name.strip() for name in class_names)
            or len(set(class_names)) != len(class_names)
            or UNKNOWN_LABEL in class_names
        ):
            raise ValueError("N4 class_names must be unique known grapheme labels")
        class_count = len(class_names)
        mark(5, class_count=class_count)

        posterior = np.asarray(n4.posterior_beliefs, dtype=np.float64)
        if (
            posterior.ndim != 2
            or posterior.shape[0] < 1
            or posterior.shape[1] != class_count
            or posterior.shape[0] > cfg.max_sequence_length
        ):
            raise ValueError("N4 posterior_beliefs must have shape [1..max_sequence_length, classes]")
        if not np.isfinite(posterior).all() or np.any(posterior < 0):
            raise ValueError("N4 posterior beliefs must be finite and non-negative")
        row_sums = posterior.sum(axis=1)
        if np.any(row_sums <= cfg.epsilon) or not np.allclose(row_sums, 1.0, atol=1e-3):
            raise ValueError("N4 posterior probability rows must sum to one")
        patch_count = int(posterior.shape[0])
        mark(6, patch_count=patch_count, posterior_shape=list(posterior.shape))
        posterior = normalize_rows(posterior, cfg.epsilon)
        input_probabilities = posterior.copy()
        mark(7, maximum_row_sum_error=float(np.max(np.abs(posterior.sum(axis=1) - 1.0))))

        attention = getattr(n4, "attention_allocation", None)
        sensory_precision = getattr(n4, "sensory_precision", None)
        if attention is None:
            attention_values = np.full(patch_count, 1.0 / patch_count, dtype=np.float32)
            warnings.append("N4 attention allocation missing; uniform attention was used.")
        else:
            attention_values = _validate_probability_vector(
                "attention_allocation", attention, patch_count, cfg.epsilon
            )
            if not np.isclose(float(attention_values.sum()), 1.0, atol=1e-3):
                raise ValueError("N4 attention_allocation must sum to one")
        if sensory_precision is None:
            precision_values = np.ones(patch_count, dtype=np.float32)
            warnings.append("N4 sensory precision missing; neutral precision was used.")
        else:
            precision_values = np.asarray(sensory_precision, dtype=np.float32)
            if (
                precision_values.shape != (patch_count,)
                or not np.isfinite(precision_values).all()
                or np.any(precision_values <= 0)
            ):
                raise ValueError("N4 sensory_precision must be finite, positive, and patch-aligned")
        upstream_valid = bool(getattr(n4, "diagnostics", {}).get("valid", True))
        if not upstream_valid:
            message = "N4 diagnostics report an invalid upstream result."
            if cfg.strict_upstream_validity:
                raise ValueError(message)
            warnings.append(message)
        mark(8, attention_mass=float(attention_values.sum()), mean_sensory_precision=float(precision_values.mean()), upstream_valid=upstream_valid)

        # 09-16: validate and align the N5 model snapshot.
        if str(n5.schema_version) != "1.0" or int(n5.version) < 1:
            raise ValueError("unsupported N5 model schema or version")
        mark(9, schema_version=str(n5.schema_version), model_version=int(n5.version))
        n5_names = tuple(n5.class_names)
        if n5_names != class_names:
            raise ValueError("N4 and N5 grapheme vocabularies must match exactly and in order")
        mark(10, vocabulary_aligned=True)

        mean = np.asarray(n5.feature_mean, dtype=np.float32)
        scale = np.asarray(n5.feature_scale, dtype=np.float32)
        if mean.ndim != 1 or mean.size < 1 or scale.shape != mean.shape or np.any(scale <= 0):
            raise ValueError("N5 feature_mean and positive feature_scale must be aligned vectors")
        feature_dimension = int(mean.size)
        mark(11, feature_dimension=feature_dimension)

        prototypes: dict[str, np.ndarray] = {}
        for name in ("fast_prototypes", "slow_prototypes", "fast_eligibility", "slow_eligibility"):
            value = np.asarray(getattr(n5, name), dtype=np.float32)
            if value.shape != (class_count, feature_dimension):
                raise ValueError(f"N5 {name} must have shape ({class_count}, {feature_dimension})")
            prototypes[name] = value.copy()
        if any(np.any(np.linalg.norm(prototypes[name], axis=1) <= cfg.epsilon)
               for name in ("fast_prototypes", "slow_prototypes")):
            raise ValueError("N5 prototype rows must be non-zero")
        mark(12, prototype_shape=[class_count, feature_dimension])

        output_count = class_count + 1
        transition_fields = (
            "fast_forward_counts", "slow_forward_counts",
            "fast_reverse_counts", "slow_reverse_counts",
        )
        transitions_raw: dict[str, np.ndarray] = {}
        for name in transition_fields:
            value = np.asarray(getattr(n5, name), dtype=np.float64)
            if value.shape != (output_count, output_count):
                raise ValueError(f"N5 {name} must have shape ({output_count}, {output_count})")
            transitions_raw[name] = value.copy()
        mark(13, transition_shape=[output_count, output_count])

        exposure_counts = np.asarray(n5.exposure_counts, dtype=np.float64)
        class_gains_state = np.asarray(n5.class_gains, dtype=np.float64)
        fast_unigram_state = np.asarray(n5.fast_unigram_counts, dtype=np.float64)
        slow_unigram_state = np.asarray(n5.slow_unigram_counts, dtype=np.float64)
        if exposure_counts.shape != (class_count,) or class_gains_state.shape != (class_count,):
            raise ValueError("N5 exposure_counts and class_gains must align with the grapheme vocabulary")
        if fast_unigram_state.shape != (output_count,) or slow_unigram_state.shape != (output_count,):
            raise ValueError("N5 fast/slow unigram arrays must include the reserved unknown class")
        numeric_fields = [mean, scale, *prototypes.values(), exposure_counts,
                          class_gains_state, *transitions_raw.values(),
                          fast_unigram_state, slow_unigram_state]
        if any(not np.isfinite(array).all() for array in numeric_fields):
            raise ValueError("N5 model contains non-finite numeric state")
        if np.any(scale <= 0) or np.any(class_gains_state <= 0):
            raise ValueError("N5 feature scales and class gains must be positive")
        if np.any(exposure_counts < 0) or any(np.any(array < 0) for array in [
            *transitions_raw.values(), fast_unigram_state, slow_unigram_state
        ]):
            raise ValueError("N5 exposure, transition, and unigram counts cannot be negative")
        mark(14, finite_numeric_state=True)

        parent_fingerprint = str(n5.fingerprint)
        if not parent_fingerprint or n5_fingerprint(n5) != parent_fingerprint:
            raise ValueError("N5 model fingerprint mismatch")
        mark(15, parent_fingerprint_valid=True)
        class_index = {name: index for index, name in enumerate(class_names)}
        mark(16, indexed_class_count=len(class_index))

        # 17-24: consolidate N5 learned statistics into evidence priors.
        forward_counts = transitions_raw["fast_forward_counts"][:class_count, :class_count] + transitions_raw["slow_forward_counts"][:class_count, :class_count]
        reverse_counts = transitions_raw["fast_reverse_counts"][:class_count, :class_count] + transitions_raw["slow_reverse_counts"][:class_count, :class_count]
        mark(17, forward_transition_mass=float(forward_counts.sum()))
        mark(18, reverse_transition_mass=float(reverse_counts.sum()))

        forward_counts += cfg.transition_smoothing
        forward_transition = forward_counts / np.maximum(
            forward_counts.sum(axis=1, keepdims=True), cfg.epsilon
        )
        mark(19, forward_rows_normalized=True)
        reverse_counts += cfg.transition_smoothing
        reverse_transition = reverse_counts / np.maximum(
            reverse_counts.sum(axis=0, keepdims=True), cfg.epsilon
        )
        mark(20, reverse_columns_normalized=True)

        unigram_counts = fast_unigram_state[:class_count] + slow_unigram_state[:class_count]
        mark(21, unigram_mass=float(unigram_counts.sum()))
        unigram_prior = unigram_counts + cfg.unigram_smoothing
        unigram_prior /= max(float(unigram_prior.sum()), cfg.epsilon)
        mark(22, prior_entropy=float(-np.sum(unigram_prior * np.log(np.maximum(unigram_prior, cfg.epsilon)))))

        class_gains = class_gains_state
        if class_gains.shape != (class_count,):
            raise ValueError("N5 class_gains must align with the grapheme vocabulary")
        gain_factors = np.power(np.clip(class_gains, 0.25, 4.0), cfg.class_gain_strength)
        gain_factors = np.clip(gain_factors, 0.5, 2.0)
        mark(23, min_class_gain=float(gain_factors.min()), max_class_gain=float(gain_factors.max()))
        calibrated = posterior.copy()
        prototype_evidence_used = False
        if request.n2_features is not None and cfg.prototype_evidence_weight > 0.0:
            n2_features = np.asarray(request.n2_features, dtype=np.float32)
            if n2_features.shape != (patch_count, feature_dimension):
                raise ValueError(
                    f"n2_features must have shape ({patch_count}, {feature_dimension})"
                )
            if not np.isfinite(n2_features).all():
                raise ValueError("n2_features contain non-finite values")
            standardized = (n2_features - mean[None, :]) / scale[None, :]
            norms = np.linalg.norm(standardized, axis=1, keepdims=True)
            if np.any(norms <= cfg.epsilon):
                raise ValueError("N2 feature vectors must have non-zero standardized activity")
            normalized_features = standardized / np.maximum(norms, cfg.epsilon)
            effective_prototypes = 0.5 * prototypes["fast_prototypes"] + 0.5 * prototypes["slow_prototypes"]
            prototype_norms = np.linalg.norm(effective_prototypes, axis=1, keepdims=True)
            effective_prototypes = effective_prototypes / np.maximum(prototype_norms, cfg.epsilon)
            similarities = np.clip(normalized_features @ effective_prototypes.T, -1.0, 1.0)
            prototype_probabilities = stable_softmax(
                similarities, temperature=cfg.prototype_temperature
            )
            weight = cfg.prototype_evidence_weight
            calibrated = (1.0 - weight) * calibrated + weight * prototype_probabilities
            prototype_evidence_used = True
        if cfg.use_class_gains:
            calibrated = normalize_rows(calibrated * gain_factors[None, :], cfg.epsilon)
        else:
            calibrated = normalize_rows(calibrated, cfg.epsilon)
        mark(24, class_gain_modulation=bool(cfg.use_class_gains), n5_prototype_evidence_used=prototype_evidence_used)
        log_emission = np.log(np.maximum(calibrated.astype(np.float64), cfg.epsilon))
        mark(25, log_emission_shape=list(log_emission.shape))

        # 26-31: bidirectional contextual integration.
        left_support = np.empty_like(calibrated, dtype=np.float64)
        right_support = np.empty_like(calibrated, dtype=np.float64)
        for index in range(patch_count):
            left_support[index] = unigram_prior if index == 0 else calibrated[index - 1] @ forward_transition
            right_support[index] = unigram_prior if index == patch_count - 1 else reverse_transition @ calibrated[index + 1]
        mark(26, mean_left_support=float(left_support.mean()))
        mark(27, mean_right_support=float(right_support.mean()))
        if cfg.use_bidirectional_context:
            contextual_log = (
                cfg.forward_context_weight * np.log(np.maximum(left_support, cfg.epsilon))
                + cfg.reverse_context_weight * np.log(np.maximum(right_support, cfg.epsilon))
            )
        else:
            contextual_log = np.zeros_like(log_emission)
        mark(28, bidirectional_context=bool(cfg.use_bidirectional_context))
        contextual_logits = log_emission + contextual_log
        mark(29, context_log_evidence_mean=float(contextual_log.mean()))
        contextual_probabilities = stable_softmax(
            contextual_logits, temperature=cfg.context_temperature
        )
        mark(30, contextual_rows_normalized=True)

        pair_transition = np.sqrt(np.maximum(forward_transition * reverse_transition, cfg.epsilon))
        pair_transition /= np.maximum(pair_transition.sum(axis=1, keepdims=True), cfg.epsilon)
        log_transition = np.log(np.maximum(pair_transition, cfg.epsilon))
        mark(31, pair_transition_shape=list(pair_transition.shape))

        # 32-42: recurrent sequence decoding and uncertainty-preserving output.
        delta = np.empty_like(log_emission)
        backpointer = np.zeros(log_emission.shape, dtype=np.int64)
        delta[0] = log_emission[0]
        mark(32, initial_state_count=class_count)
        for index in range(1, patch_count):
            candidates = delta[index - 1][:, None] + log_transition
            backpointer[index] = np.argmax(candidates, axis=0)
            delta[index] = log_emission[index] + np.max(candidates, axis=0)
        mark(33, recurrent_steps=max(patch_count - 1, 0))
        best_path = np.zeros(patch_count, dtype=np.int64)
        best_path[-1] = int(np.argmax(delta[-1]))
        for index in range(patch_count - 1, 0, -1):
            best_path[index - 1] = backpointer[index, best_path[index]]
        decoded_graphemes = tuple(class_names[int(index)] for index in best_path)
        mark(34, decoded_path_length=len(decoded_graphemes))

        alpha = np.empty_like(log_emission)
        alpha[0] = log_emission[0]
        for index in range(1, patch_count):
            alpha[index] = log_emission[index] + _logsumexp(
                alpha[index - 1][:, None] + log_transition, axis=0
            )
        mark(35, forward_messages=int(alpha.size))
        beta = np.zeros_like(log_emission)
        for index in range(patch_count - 2, -1, -1):
            beta[index] = _logsumexp(
                log_transition + log_emission[index + 1][None, :] + beta[index + 1][None, :],
                axis=1,
            )
        mark(36, backward_messages=int(beta.size))
        marginals = stable_softmax(alpha + beta, temperature=1.0)
        mark(37, marginal_shape=list(marginals.shape))
        fused = 0.70 * marginals + 0.30 * contextual_probabilities
        mark(38, marginal_context_blend=0.70)
        decoded_probabilities = normalize_rows(fused, cfg.epsilon)
        mark(39, decoded_rows_normalized=True)
        uncertainty = entropy_rows(decoded_probabilities, cfg.epsilon)
        normalized_uncertainty = uncertainty / max(log(class_count), cfg.epsilon) if class_count > 1 else np.zeros(patch_count, dtype=np.float32)
        mark(40, mean_normalized_uncertainty=float(np.mean(normalized_uncertainty)))
        if patch_count > 1:
            chosen_pairs = [pair_transition[best_path[i - 1], best_path[i]] for i in range(1, patch_count)]
            sequence_coherence = float(np.exp(np.mean(np.log(np.maximum(chosen_pairs, cfg.epsilon)))))
        else:
            sequence_coherence = 1.0
        mark(41, sequence_transition_coherence=sequence_coherence)
        position_candidates: list[dict[str, Any]] = []
        for position, row in enumerate(decoded_probabilities):
            order = np.argsort(row)[::-1][:cfg.top_k_position_candidates]
            position_candidates.append({
                "position": position,
                "candidates": [
                    {"grapheme": class_names[int(index)], "probability": float(row[index])}
                    for index in order
                ],
            })
        mark(42, candidate_positions=len(position_candidates))

        # 43-49: validate cues and build the orthographic segmentation lattice.
        if request.boundary_probabilities is None:
            supplied_boundaries = None
        else:
            supplied_boundaries = _validate_probability_vector(
                "boundary_probabilities", request.boundary_probabilities,
                patch_count + 1, cfg.epsilon,
            )
            supplied_boundaries[0] = 1.0
            supplied_boundaries[-1] = 1.0
        mark(43, boundary_cues_supplied=supplied_boundaries is not None)

        raw_entries = list(request.lexicon_entries)
        if len(raw_entries) > cfg.max_lexicon_entries:
            raise ValueError("lexicon exceeds max_lexicon_entries")
        lexicon = [_as_lexicon_entry(item) for item in raw_entries]
        identifiers = [entry["entry_id"] for entry in lexicon]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("lexicon entry_id values must be unique")
        for entry in lexicon:
            if len(entry["graphemes"]) > cfg.max_lexicon_graphemes:
                raise ValueError("lexicon entry exceeds max_lexicon_graphemes")
            missing_labels = [g for g in entry["graphemes"] if g not in class_index]
            if missing_labels:
                raise ValueError(f"lexicon entry contains unknown graphemes: {missing_labels}")
        mark(44, validated_entry_count=len(lexicon))

        lexicon_by_first: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in lexicon:
            lexicon_by_first[entry["graphemes"][0]].append(entry)
        for values in lexicon_by_first.values():
            values.sort(key=lambda item: (-item["frequency"], item["entry_id"]))
        mark(45, first_grapheme_buckets=len(lexicon_by_first))

        edge_lists: list[list[dict[str, Any]]] = [[] for _ in range(patch_count)]
        lexical_support = np.zeros(patch_count + 1, dtype=np.float64)
        entry_matches = 0
        for start in range(patch_count):
            first_order = np.argsort(decoded_probabilities[start])[::-1]
            first_order = first_order[:cfg.max_lexicon_first_graphemes]
            candidate_entries: list[dict[str, Any]] = []
            for first_index in first_order:
                candidate_entries.extend(lexicon_by_first.get(class_names[int(first_index)], []))
            candidate_entries.sort(key=lambda item: (-item["frequency"], item["entry_id"]))
            for entry in candidate_entries[:cfg.max_lexicon_candidates_per_position]:
                graphemes = entry["graphemes"]
                end = start + len(graphemes)
                if end > patch_count:
                    continue
                indices = [class_index[g] for g in graphemes]
                emission_values = [decoded_probabilities[start + offset, idx] for offset, idx in enumerate(indices)]
                emission_log = float(np.sum(np.log(np.maximum(emission_values, cfg.epsilon))))
                mean_evidence = float(np.mean(emission_values))
                if len(indices) > 1:
                    transition_values = [pair_transition[a, b] for a, b in zip(indices[:-1], indices[1:])]
                    transition_log = float(np.sum(np.log(np.maximum(transition_values, cfg.epsilon))))
                else:
                    transition_log = 0.0
                edge = {
                    "kind": "lexicon", "entry_id": entry["entry_id"],
                    "graphemes": list(graphemes), "start": start, "end": end,
                    "orthography": "".join(graphemes), "frequency": entry["frequency"],
                    "verified_by": entry["verified_by"],
                    "source_reference": entry["source_reference"],
                    "language": entry["language"],
                    "emission_log": emission_log, "transition_log": transition_log,
                    "mean_evidence": mean_evidence,
                }
                edge_lists[start].append(edge)
                lexical_support[end] += max(mean_evidence, cfg.epsilon)
                entry_matches += 1
        mark(46, lexical_span_matches=entry_matches)
        mark(47, mean_lexical_boundary_support=float(np.mean(lexical_support)))

        if supplied_boundaries is not None and cfg.use_boundary_evidence:
            boundary_probabilities = supplied_boundaries.copy()
            boundary_source = "caller_supplied"
        else:
            boundary_probabilities = np.full(
                patch_count + 1, cfg.default_boundary_probability, dtype=np.float32
            )
            for boundary in range(1, patch_count):
                support = lexical_support[boundary]
                boundary_probabilities[boundary] = np.clip(
                    cfg.default_boundary_probability + 0.10 * np.log1p(support),
                    0.05, 0.95,
                )
            boundary_source = "lexicon_affinity_heuristic"
        boundary_probabilities[0] = 1.0
        boundary_probabilities[-1] = 1.0
        mark(48, boundary_source=boundary_source, mean_internal_boundary=float(np.mean(boundary_probabilities[1:-1])) if patch_count > 1 else 1.0)

        for position in range(patch_count):
            grapheme_index = int(np.argmax(decoded_probabilities[position]))
            grapheme = class_names[grapheme_index]
            probability = float(decoded_probabilities[position, grapheme_index])
            edge_lists[position].append({
                "kind": "unknown", "entry_id": None,
                "graphemes": [grapheme], "start": position, "end": position + 1,
                "orthography": grapheme, "frequency": 0.0,
                "verified_by": None, "source_reference": None, "language": "und",
                "emission_log": float(np.log(max(probability, cfg.epsilon))),
                "transition_log": 0.0, "mean_evidence": probability,
            })
        mark(49, fallback_edges=patch_count)

        # 50-57: score word hypotheses and solve the k-best segmentation graph.
        edge_count = 0
        for start, edges in enumerate(edge_lists):
            for edge in edges:
                start_boundary = 0.0 if start == 0 else logit(float(boundary_probabilities[start]), cfg.epsilon)
                end_boundary = 0.0 if edge["end"] == patch_count else logit(float(boundary_probabilities[edge["end"]]), cfg.epsilon)
                max_frequency = max((item["frequency"] for item in lexicon), default=1.0)
                frequency_score = (
                    np.log1p(float(edge["frequency"])) / max(np.log1p(max_frequency), cfg.epsilon)
                    if edge["kind"] == "lexicon" else 0.0
                )
                if edge["kind"] == "lexicon":
                    edge["score"] = (
                        edge["emission_log"]
                        + cfg.lexical_transition_weight * edge["transition_log"]
                        + cfg.lexical_frequency_weight * float(frequency_score)
                        + cfg.boundary_weight * (start_boundary + end_boundary)
                    )
                else:
                    edge["score"] = (
                        edge["emission_log"] - cfg.unknown_segment_penalty
                        + cfg.boundary_weight * (start_boundary + end_boundary)
                    )
                edge_count += 1
        mark(50, segmentation_edges=edge_count)
        k_paths = k_best_paths(edge_lists, patch_count, cfg.top_k_segmentations)
        mark(51, retained_complete_paths=len(k_paths))
        path_scores = np.asarray([score for score, _ in k_paths], dtype=np.float64)
        segmentation_probabilities = stable_softmax(
            path_scores, temperature=cfg.lexical_confidence_temperature
        )
        mark(52, segmentation_probability_sum=float(segmentation_probabilities.sum()))
        best_score, best_path_edges = k_paths[0]
        mark(53, selected_token_count=len(best_path_edges), best_path_score=float(best_score))

        tokens: list[dict[str, Any]] = []
        position_to_token = [-1] * patch_count
        for token_index, edge in enumerate(best_path_edges):
            for position in range(edge["start"], edge["end"]):
                if position_to_token[position] != -1:
                    raise RuntimeError("segmentation overlaps a grapheme position")
                position_to_token[position] = token_index
        mark(54, bound_position_count=sum(index >= 0 for index in position_to_token))

        best_path_probability = float(segmentation_probabilities[0])
        for token_index, edge in enumerate(best_path_edges):
            span_values = [
                float(decoded_probabilities[position, class_index[grapheme]])
                for position, grapheme in zip(
                    range(edge["start"], edge["end"]), edge["graphemes"]
                )
            ]
            evidence_confidence = float(np.exp(np.mean(np.log(np.maximum(span_values, cfg.epsilon)))))
            mean_attention = float(np.mean(attention_values[edge["start"]:edge["end"]]))
            relative_attention = min(1.0, mean_attention * patch_count)
            mean_precision = float(np.mean(precision_values[edge["start"]:edge["end"]]))
            precision_factor = float(np.clip(mean_precision / max(float(np.mean(precision_values)), cfg.epsilon), 0.5, 2.0))
            token_confidence = float(np.clip(
                evidence_confidence * (0.80 + 0.20 * relative_attention)
                * (0.90 + 0.10 * precision_factor), 0.0, 1.0
            ))
            tokens.append({
                "token_id": f"{sequence_id}:token:{token_index + 1}",
                "entry_id": edge["entry_id"],
                "orthography": edge["orthography"],
                "graphemes": list(edge["graphemes"]),
                "start": int(edge["start"]),
                "end": int(edge["end"]),
                "known_lexicon_entry": edge["kind"] == "lexicon",
                "verified_by": edge["verified_by"],
                "source_reference": edge["source_reference"],
                "language": edge["language"],
                "evidence_confidence": evidence_confidence,
                "confidence": token_confidence,
                "segmentation_probability": best_path_probability,
                "boundary_start_probability": float(boundary_probabilities[edge["start"]]),
                "boundary_end_probability": float(boundary_probabilities[edge["end"]]),
                "mean_attention": mean_attention,
                "mean_sensory_precision": mean_precision,
                "score": float(edge["score"]),
            })
        mark(55, mean_token_confidence=float(np.mean([item["confidence"] for item in tokens])))

        for token in tokens:
            span = slice(token["start"], token["end"])
            local_top = np.sort(decoded_probabilities[span], axis=1)[:, ::-1]
            mean_margin = float(np.mean(
                local_top[:, 0] - (local_top[:, 1] if class_count > 1 else 0.0)
            ))
            token["mean_grapheme_margin"] = mean_margin
            low_confidence = token["confidence"] < cfg.abstain_threshold
            ambiguous = mean_margin < cfg.ambiguity_margin
            token["abstain"] = bool(low_confidence or ambiguous)
            token["status"] = (
                "UNLEXICALIZED_GRAPHEME" if not token["known_lexicon_entry"]
                else "ABSTAIN" if token["abstain"]
                else "LEXICAL_MATCH"
            )
        mark(56, abstained_token_count=sum(bool(item["abstain"]) for item in tokens))

        segmentation_alternatives: list[dict[str, Any]] = []
        for rank, ((score, edges), probability) in enumerate(zip(k_paths, segmentation_probabilities), start=1):
            segmentation_alternatives.append({
                "rank": rank,
                "probability": float(probability),
                "score": float(score),
                "tokens": [
                    {
                        "entry_id": edge["entry_id"],
                        "orthography": edge["orthography"],
                        "graphemes": list(edge["graphemes"]),
                        "start": int(edge["start"]),
                        "end": int(edge["end"]),
                        "known_lexicon_entry": edge["kind"] == "lexicon",
                    }
                    for edge in edges
                ],
            })
        lexical_coverage = float(sum(
            token["end"] - token["start"] for token in tokens if token["known_lexicon_entry"]
        ) / patch_count)
        unknown_ratio = float(1.0 - lexical_coverage)
        mark(57, alternative_count=len(segmentation_alternatives), lexical_coverage=lexical_coverage, unknown_ratio=unknown_ratio)

        if any(position < 0 for position in position_to_token):
            raise RuntimeError("segmentation failed to cover every grapheme position")
        if n5_fingerprint(n5) != parent_fingerprint:
            raise RuntimeError("N6 modified the N5 model; read-only contract violated")
        mark(58, complete_position_coverage=True, n5_unchanged=True)

        diagnostics = {
            "valid": True,
            "stage_count": 60,
            "executed_operation_count": len(trace) + 2,
            "patch_count": patch_count,
            "class_count": class_count,
            "token_count": len(tokens),
            "lexicon_entry_count": len(lexicon),
            "lexicon_span_match_count": entry_matches,
            "lexical_coverage": lexical_coverage,
            "unknown_ratio": unknown_ratio,
            "sequence_transition_coherence": sequence_coherence,
            "mean_normalized_uncertainty": float(np.mean(normalized_uncertainty)),
            "best_segmentation_probability": best_path_probability,
            "all_probabilities_finite": bool(
                np.isfinite(input_probabilities).all()
                and np.isfinite(contextual_probabilities).all()
                and np.isfinite(decoded_probabilities).all()
                and np.isfinite(boundary_probabilities).all()
            ),
            "N5_parent_unchanged": True,
            "warnings": warnings,
        }
        metadata = {
            "schema": "ore.n6.hierarchical_orthographic_access.v1",
            "n5_parent_fingerprint": parent_fingerprint,
            "n5_model_version": int(n5.version),
            "boundary_source": boundary_source,
            "n5_prototype_evidence_used": prototype_evidence_used,
            "context_mode": "bidirectional" if cfg.use_bidirectional_context else "visual_only",
            "lexicon_policy": "caller_supplied_entries_with_verifier_and_source_provenance",
            "scientific_scope": (
                "Algorithmic approximation of sequence-level orthographic inference and lexical lookup; "
                "not a validated neural circuit model or semantic-language model."
            ),
            "configuration": cfg.to_dict(),
            "elapsed_seconds": float(perf_counter() - started),
        }
        mark(59, warnings=len(warnings), diagnostics_valid=True)
        mark(60, result_schema="ore.n6.hierarchical_orthographic_access.v1")

        diagnostics["executed_operation_count"] = len(trace)
        diagnostics["operation_trace_complete"] = len(trace) == 60
        return N6Result(
            sequence_id=sequence_id,
            class_names=class_names,
            input_probabilities=input_probabilities.astype(np.float32),
            contextual_probabilities=contextual_probabilities.astype(np.float32),
            decoded_probabilities=decoded_probabilities.astype(np.float32),
            decoded_graphemes=decoded_graphemes,
            boundary_probabilities=np.asarray(boundary_probabilities, dtype=np.float32),
            tokens=tokens,
            position_candidates=position_candidates,
            segmentation_alternatives=segmentation_alternatives,
            operation_trace=trace,
            diagnostics=diagnostics,
            metadata=metadata,
        )
