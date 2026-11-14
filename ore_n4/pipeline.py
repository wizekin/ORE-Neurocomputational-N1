"""The 40-operation ORE N4 attention/precision-gating algorithm."""
from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from .config import N4Config
from .contracts import N4Input, N4Result
from .stages import (
    damped_belief_update,
    normalize_distribution,
    normalize_signal,
    normalized_entropy,
    rank_hypotheses,
    smooth_spatial_attention,
    stable_softmax,
    symmetric_kl_rows,
)

N4_ALGORITHM_STEPS: tuple[str, ...] = (
    "01_validate_N4_input_contract",
    "02_validate_N3_result_interface",
    "03_validate_class_vocabulary",
    "04_validate_probability_and_error_tensors",
    "05_validate_probability_normalization",
    "06_extract_reliability_and_seed_cues",
    "07_measure_visual_entropy",
    "08_normalize_visual_uncertainty",
    "09_measure_classwise_prediction_error",
    "10_reduce_prediction_error_to_patch_signal",
    "11_measure_visual_prior_disagreement",
    "12_normalize_predictive_surprise",
    "13_derive_internal_bottom_up_salience",
    "14_integrate_external_bottom_up_salience",
    "15_bound_and_store_bottom_up_salience",
    "16_extract_task_relevance_cues",
    "17_project_goal_class_weights_onto_patch_beliefs",
    "18_combine_position_and_class_goal_relevance",
    "19_integrate_seed_attention_with_top_down_relevance",
    "20_compute_competitive_attention_logits",
    "21_allocate_attention_across_sequence_positions",
    "22_preserve_minimum_uniform_attention_mass",
    "23_apply_local_spatial_attention_smoothing",
    "24_renormalize_spatial_attention_allocation",
    "25_derive_classwise_attention_from_error_and_belief",
    "26_normalize_classwise_attention_gates",
    "27_estimate_patch_observation_reliability",
    "28_compute_base_sensory_precision",
    "29_modulate_sensory_precision_by_attention",
    "30_compute_uncertainty_sensitive_prior_precision",
    "31_modulate_prior_precision_by_top_down_relevance",
    "32_bound_and_validate_precision_values",
    "33_initialize_attended_belief_state_from_N3",
    "34_construct_precision_weighted_evidence_proposal",
    "35_normalize_attended_posterior_proposal",
    "36_iterate_damped_attended_belief_updates",
    "37_compute_final_attention_conditioned_prediction_error",
    "38_rank_candidates_and_apply_abstention_rules",
    "39_build_diagnostics_trace_and_provenance",
    "40_return_structured_N4_result",
)


class N4Processor:
    """Apply N4 attention allocation and precision gating to an N3 result.

    The processor is read-only with respect to the supplied N3 object. It keeps
    visual evidence and contextual priors as separate channels and returns a
    new result object rather than rewriting N3 state.
    """

    def __init__(self, config: N4Config | None = None) -> None:
        self.config = config or N4Config()

    def process(self, request: N4Input) -> N4Result:
        started = perf_counter()
        if not isinstance(request, N4Input):
            raise TypeError("request must be an N4Input")
        cfg = self.config
        operation_trace: list[dict[str, Any]] = []
        warnings: list[str] = []

        def mark(step: int, **summary: Any) -> None:
            operation_trace.append({
                "step": step,
                "operation": N4_ALGORITHM_STEPS[step - 1],
                **summary,
            })

        # 01. Validate the N4 request contract.
        if request.n3_result is None:
            raise ValueError("N4Input.n3_result is required")
        mark(1, status="valid")

        # 02. Validate the upstream N3 interface without importing or
        #     mutating the N3 implementation.
        n3 = request.n3_result
        required = (
            "sequence_id", "class_names", "visual_probabilities",
            "contextual_priors", "posterior_beliefs",
            "classwise_prediction_error", "candidates",
        )
        missing = [name for name in required if not hasattr(n3, name)]
        if missing:
            raise TypeError(f"N3 result is missing required fields: {missing}")
        if not isinstance(n3.sequence_id, str) or not n3.sequence_id.strip():
            raise ValueError("N3 sequence_id must be a non-empty string")
        mark(2, status="valid", sequence_id=str(n3.sequence_id))

        # 03. Validate a stable, unique, non-empty class vocabulary.
        class_names = tuple(n3.class_names)
        if not class_names or any(not isinstance(x, str) or not x for x in class_names):
            raise ValueError("N3 class_names must contain non-empty strings")
        if len(set(class_names)) != len(class_names):
            raise ValueError("N3 class_names must be unique")
        class_count = len(class_names)
        mark(3, class_count=class_count)

        # 04. Validate aligned N3 matrices and error tensor.
        visual = np.asarray(n3.visual_probabilities, dtype=np.float32)
        prior = np.asarray(n3.contextual_priors, dtype=np.float32)
        n3_beliefs = np.asarray(n3.posterior_beliefs, dtype=np.float32)
        n3_error = np.asarray(n3.classwise_prediction_error, dtype=np.float32)
        if visual.ndim != 2 or visual.shape[1] != class_count or visual.shape[0] < 1:
            raise ValueError("visual_probabilities must have shape [patches, classes]")
        expected_shape = visual.shape
        for name, array in (
            ("contextual_priors", prior),
            ("posterior_beliefs", n3_beliefs),
            ("classwise_prediction_error", n3_error),
        ):
            if array.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
        if not np.isfinite(visual).all() or np.any(visual < 0):
            raise ValueError("visual probabilities must be finite and non-negative")
        if np.any(prior < 0) or np.any(n3_beliefs < 0):
            raise ValueError("N3 prior and posterior probabilities must be non-negative")
        patch_count = int(visual.shape[0])
        mark(4, patch_count=patch_count, class_count=class_count)

        # 05. Verify row normalization; normalize tiny floating-point drift.
        for name, array in (("visual", visual), ("prior", prior), ("N3 posterior", n3_beliefs)):
            row_sums = array.sum(axis=1)
            if np.any(row_sums <= cfg.epsilon):
                raise ValueError(f"{name} contains an all-zero probability row")
            if not np.allclose(row_sums, 1.0, atol=1e-3):
                raise ValueError(f"{name} rows must sum to one")
        visual = (visual / visual.sum(axis=1, keepdims=True)).astype(np.float32)
        prior = (prior / prior.sum(axis=1, keepdims=True)).astype(np.float32)
        n3_beliefs = (n3_beliefs / n3_beliefs.sum(axis=1, keepdims=True)).astype(np.float32)
        mark(5, maximum_row_sum_error=float(np.max(np.abs(visual.sum(axis=1) - 1.0))))

        # 06. Read optional reliability and attention cues.
        candidates = list(n3.candidates or [])
        if candidates and len(candidates) != patch_count:
            raise ValueError("N3 candidate metadata count must match patch count")
        if any(not isinstance(item, dict) for item in candidates):
            raise TypeError("each N3 candidate metadata item must be a dictionary")
        patch_ids = [str(item.get("patch_id", f"patch-{i + 1}")) for i, item in enumerate(candidates)]
        if not candidates:
            patch_ids = [f"patch-{i + 1}" for i in range(patch_count)]
            candidates = [{"patch_id": patch_ids[i]} for i in range(patch_count)]
        if any(not patch_id.strip() for patch_id in patch_ids):
            raise ValueError("N3 patch IDs must be non-empty")
        if len(set(patch_ids)) != len(patch_ids):
            raise ValueError("N3 patch IDs must be unique")
        derived_confidence = np.asarray(
            [item.get("segmentation_confidence", 1.0) for item in candidates], dtype=np.float32
        )
        derived_seed = np.asarray([item.get("attention", 0.0) for item in candidates], dtype=np.float32)

        def cue_vector(name: str, supplied: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
            arr = np.asarray(fallback if supplied is None else supplied, dtype=np.float32)
            if arr.shape != (patch_count,):
                raise ValueError(f"{name} must have shape ({patch_count},)")
            if not np.isfinite(arr).all() or np.any(arr < 0) or np.any(arr > 1):
                raise ValueError(f"{name} values must be finite and lie in [0, 1]")
            return arr

        confidence = cue_vector("segmentation_confidence", request.segmentation_confidence, derived_confidence)
        seed_attention = cue_vector("seed_attention", request.seed_attention, derived_seed)
        external_salience = None if request.bottom_up_salience is None else cue_vector(
            "bottom_up_salience", request.bottom_up_salience, np.zeros(patch_count, dtype=np.float32)
        )
        task_relevance = cue_vector(
            "task_relevance", request.task_relevance, np.zeros(patch_count, dtype=np.float32)
        )
        upstream_valid = bool(getattr(n3, "diagnostics", {}).get("valid", True))
        if not upstream_valid:
            message = "N3 diagnostics report an invalid upstream result; N4 is proceeding with a warning."
            if cfg.strict_upstream_validity:
                raise ValueError(message)
            warnings.append(message)
        mark(6, cue_sources={
            "segmentation_confidence": "request_or_N3_candidates",
            "seed_attention": "request_or_N3_candidates",
            "bottom_up_salience": "request_or_internal",
            "task_relevance": "request_or_zero_default",
        })

        # 07-12. Compute uncertainty, classwise error and prior disagreement.
        entropy = normalized_entropy(visual, cfg.epsilon)
        mark(7, mean_entropy=float(np.mean(entropy)))
        uncertainty = np.clip(entropy, 0.0, 1.0)
        mark(8, mean_normalized_uncertainty=float(np.mean(uncertainty)))
        raw_error = np.mean(np.abs(n3_error), axis=1)
        mark(9, mean_absolute_class_error=float(np.mean(raw_error)))
        error_signal = normalize_signal(raw_error, cfg.epsilon)
        mark(10, maximum_normalized_error=float(np.max(error_signal)))
        disagreement = symmetric_kl_rows(visual, prior, cfg.epsilon)
        mark(11, mean_symmetric_visual_prior_KL=float(np.mean(disagreement)))
        surprise = normalize_signal(disagreement, cfg.epsilon)
        mark(12, maximum_predictive_surprise=float(np.max(surprise)))

        # 13-15. Build bounded bottom-up salience.
        internal_weights = np.asarray([
            cfg.entropy_salience_weight,
            cfg.error_salience_weight,
            cfg.surprise_salience_weight,
        ], dtype=np.float64)
        internal_weights /= internal_weights.sum()
        internal_salience = (
            internal_weights[0] * uncertainty
            + internal_weights[1] * error_signal
            + internal_weights[2] * surprise
        )
        mark(13, mean_internal_salience=float(np.mean(internal_salience)))
        if external_salience is None:
            salience = internal_salience
            external_used = False
        else:
            total_weight = cfg.entropy_salience_weight + cfg.error_salience_weight + cfg.surprise_salience_weight
            ext_weight = cfg.external_salience_weight
            salience = (internal_salience * total_weight + external_salience * ext_weight) / max(
                total_weight + ext_weight, cfg.epsilon
            )
            external_used = True
        mark(14, external_salience_used=external_used)
        salience = np.clip(salience, 0.0, 1.0).astype(np.float32)
        mark(15, min_salience=float(np.min(salience)), max_salience=float(np.max(salience)))

        # 16-19. Build top-down relevance from declared task and class goals.
        task = np.clip(task_relevance, 0.0, 1.0)
        mark(16, mean_task_relevance=float(np.mean(task)))
        if request.goal_class_weights is None:
            goal_weights = np.full(class_count, 1.0 / class_count, dtype=np.float32)
            has_goal_classes = False
        else:
            raw_goal = np.asarray(request.goal_class_weights, dtype=np.float32)
            if raw_goal.shape != (class_count,) or not np.isfinite(raw_goal).all() or np.any(raw_goal < 0):
                raise ValueError(f"goal_class_weights must be a finite non-negative vector of shape ({class_count},)")
            if float(raw_goal.sum()) <= cfg.epsilon:
                raise ValueError("goal_class_weights must contain positive mass")
            goal_weights = normalize_distribution(raw_goal, cfg.epsilon)
            has_goal_classes = True
        if has_goal_classes:
            class_goal = np.sum(n3_beliefs * goal_weights[None, :], axis=1)
            class_goal = normalize_signal(class_goal, cfg.epsilon)
        else:
            class_goal = np.zeros(patch_count, dtype=np.float32)
        mark(17, goal_class_vector_supplied=has_goal_classes)
        combined_goal = np.clip(
            cfg.task_relevance_weight * task + cfg.goal_class_weight * class_goal, 0.0, 1.0
        )
        mark(18, mean_combined_goal_relevance=float(np.mean(combined_goal)))
        top_down = combined_goal if cfg.enable_top_down_attention else np.zeros_like(combined_goal)
        top_down = np.clip(top_down + cfg.seed_attention_weight * seed_attention, 0.0, 1.0)
        mark(19, top_down_attention_enabled=cfg.enable_top_down_attention)

        # 20-24. Allocate competitive spatial attention while preserving a floor.
        attention_logits = salience + top_down
        if not cfg.enable_spatial_attention:
            attention_logits = np.zeros_like(attention_logits)
        mark(20, mean_attention_logit=float(np.mean(attention_logits)))
        allocation = stable_softmax(attention_logits / cfg.attention_temperature, axis=0).reshape(-1)
        mark(21, allocation_sum=float(np.sum(allocation)))
        uniform = np.full(patch_count, 1.0 / patch_count, dtype=np.float32)
        allocation = (
            (1.0 - cfg.uniform_attention_fraction) * allocation
            + cfg.uniform_attention_fraction * uniform
        ).astype(np.float32)
        mark(22, uniform_attention_fraction=cfg.uniform_attention_fraction)
        allocation = smooth_spatial_attention(allocation, cfg.spatial_smoothing)
        mark(23, spatial_smoothing=cfg.spatial_smoothing)
        allocation = np.maximum(allocation, cfg.epsilon)
        allocation /= allocation.sum()
        allocation = allocation.astype(np.float32)
        mark(24, allocation_sum=float(np.sum(allocation)))

        # 25-26. Construct class-selective attention gates.
        absolute_error = np.abs(n3_error)
        row_scale = np.max(absolute_error, axis=1, keepdims=True)
        scaled_error = np.divide(
            absolute_error, np.maximum(row_scale, cfg.epsilon),
            out=np.zeros_like(absolute_error), where=row_scale > cfg.epsilon,
        )
        class_attention_logits = np.log(np.maximum(n3_beliefs, cfg.epsilon)) + scaled_error
        if not cfg.enable_class_attention:
            class_attention_logits = np.zeros_like(class_attention_logits)
        mark(25, class_attention_enabled=cfg.enable_class_attention)
        class_attention = stable_softmax(class_attention_logits, axis=1)
        # Keep an interpretable class gate centred on one; row normalisation is
        # retained separately as the underlying class-attention distribution.
        class_attention_gate = np.clip(class_attention * class_count, 0.25, 4.0)
        mark(26, mean_class_gate=float(np.mean(class_attention_gate)))

        # 27-32. Convert reliability and attention into bounded precision.
        observation_quality = 0.5 + 0.5 * confidence
        mark(27, mean_observation_quality=float(np.mean(observation_quality)))
        sensory_base = cfg.sensory_precision_base * observation_quality
        mark(28, mean_base_sensory_precision=float(np.mean(sensory_base)))
        allocation_multiplier = allocation * patch_count
        sensory_precision = sensory_base * (1.0 + cfg.attention_gain * allocation_multiplier)
        if not cfg.enable_precision_gating:
            sensory_precision = np.full(patch_count, cfg.sensory_precision_base, dtype=np.float32)
        mark(29, mean_sensory_precision_before_clipping=float(np.mean(sensory_precision)))
        prior_precision = cfg.prediction_gain * (0.5 + uncertainty) * (2.0 - confidence)
        mark(30, mean_prior_precision_before_task=float(np.mean(prior_precision)))
        prior_precision *= 1.0 + cfg.task_prior_gain * top_down
        mark(31, mean_prior_precision_after_task=float(np.mean(prior_precision)))
        sensory_precision = np.clip(
            sensory_precision, cfg.sensory_precision_min, cfg.sensory_precision_max
        ).astype(np.float32)
        prior_precision = np.clip(
            prior_precision, cfg.prior_precision_min, cfg.prior_precision_max
        ).astype(np.float32)
        if not np.isfinite(sensory_precision).all() or not np.isfinite(prior_precision).all():
            raise ValueError("precision calculation produced non-finite values")
        mark(32, precision_bounds_valid=True)

        # 33-36. Reconcile sensory and contextual evidence under attended precision.
        beliefs = n3_beliefs.copy()
        mark(33, initialized_from="N3 posterior beliefs")
        # Operation 34 prepares the mathematical proposal; operation 35 is
        # executed at each inference cycle after the logits are formed.
        proposal_seed, _ = damped_belief_update(
            beliefs, visual, prior, sensory_precision, prior_precision,
            class_attention_gate, 1.0, cfg.epsilon,
        )
        mark(34, proposal_shape=list(proposal_seed.shape))
        proposal = proposal_seed
        mark(35, proposal_row_sum_error=float(np.max(np.abs(proposal.sum(axis=1) - 1.0))))

        iteration_trace: list[dict[str, Any]] = []
        converged = False
        mean_change = float("inf")
        for iteration in range(1, cfg.max_iterations + 1):
            updated, proposal = damped_belief_update(
                beliefs, visual, prior, sensory_precision, prior_precision,
                class_attention_gate, cfg.damping, cfg.epsilon,
            )
            mean_change = float(np.mean(np.abs(updated - beliefs)))
            iteration_trace.append({
                "iteration": iteration,
                "mean_belief_change": mean_change,
                "mean_max_probability": float(np.mean(np.max(updated, axis=1))),
                "posterior_entropy": float(np.mean(normalized_entropy(updated, cfg.epsilon))),
            })
            beliefs = updated
            if mean_change <= cfg.convergence_tolerance:
                converged = True
                break
        mark(36, iterations=len(iteration_trace), converged=converged)

        # 37. Compute attention-conditioned mismatch against the contextual prior.
        final_error = beliefs * (
            np.log(np.maximum(beliefs, cfg.epsilon))
            - np.log(np.maximum(prior, cfg.epsilon))
        )
        final_error = final_error.astype(np.float32)
        mark(37, mean_absolute_error=float(np.mean(np.abs(final_error))))

        # 38. Rank alternatives and explicitly abstain on weak/ambiguous results.
        ranked = rank_hypotheses(
            beliefs, class_names, patch_ids, confidence, allocation,
            cfg.top_k, cfg.abstain_threshold, cfg.ambiguity_margin,
        )
        mark(38, candidate_count=len(ranked))

        # 39. Validate diagnostics and record exact provenance.
        probability_error = float(np.max(np.abs(beliefs.sum(axis=1) - 1.0)))
        arrays = (
            visual, prior, n3_beliefs, beliefs, salience, allocation,
            class_attention, class_attention_gate, sensory_precision, prior_precision, final_error,
        )
        finite = all(np.isfinite(value).all() for value in arrays)
        diagnostics = {
            "valid": bool(finite and probability_error <= 1e-5 and len(operation_trace) == 38),
            "stage_count": 40,
            "step_names": list(N4_ALGORITHM_STEPS),
            "executed_operations_before_packaging": len(operation_trace),
            "iterations": len(iteration_trace),
            "converged": converged,
            "stop_reason": "converged" if converged else "maximum_iterations_reached",
            "mean_belief_change": mean_change,
            "probability_row_sum_error": probability_error,
            "finite_outputs": bool(finite),
            "attention_allocation_sum": float(allocation.sum()),
            "attention_entropy": float(-np.sum(allocation * np.log(np.maximum(allocation, cfg.epsilon)))),
            "mean_bottom_up_salience": float(np.mean(salience)),
            "mean_top_down_relevance": float(np.mean(top_down)),
            "mean_sensory_precision": float(np.mean(sensory_precision)),
            "mean_prior_precision": float(np.mean(prior_precision)),
            "abstention_count": sum(item["decision"] == "ABSTAIN" for item in ranked),
            "ambiguous_count": sum(item["decision"] == "AMBIGUOUS" for item in ranked),
            "upstream_n3_valid": upstream_valid,
            "warnings": warnings,
            "elapsed_seconds": perf_counter() - started,
            "scientific_status": (
                "algorithmic attention/precision-gating approximation; not a validated neural circuit model"
            ),
        }
        metadata = {
            "pipeline": "ORE N4 Attention and Precision Gating",
            "version": "0.1.0",
            "input_pipeline": "ORE N3 recurrent orthographic prediction and error",
            "sequence_id": str(n3.sequence_id),
            "patch_count": patch_count,
            "class_names": list(class_names),
            "config": cfg.to_dict(),
            "read_only_upstream": True,
            "visual_evidence_source": "N3 visual_probabilities",
            "contextual_prior_source": "N3 contextual_priors",
            "attention_scope": "sequence-position allocation and class-selective gates",
            "limitations": [
                "Attention cues are algorithmic proxies, not recorded neural attention.",
                "Precision values are configured computational gains, not measured neural precision.",
                "No eye movement, fixation, saccade or cortical microcircuit simulation is included.",
                "Software tests do not establish biological validity or real-world OCR performance.",
            ],
        }
        mark(39, valid=diagnostics["valid"], warning_count=len(warnings))

        # 40. Return a new result; the N3 object and its arrays remain untouched.
        result = N4Result(
            sequence_id=str(n3.sequence_id),
            class_names=class_names,
            visual_probabilities=visual.copy(),
            contextual_priors=prior.copy(),
            n3_posterior_beliefs=n3_beliefs.copy(),
            posterior_beliefs=beliefs.copy(),
            bottom_up_salience=salience.copy(),
            top_down_relevance=top_down.copy(),
            attention_allocation=allocation.copy(),
            class_attention=class_attention.copy(),
            class_attention_gate=class_attention_gate.copy(),
            sensory_precision=sensory_precision.copy(),
            prior_precision=prior_precision.copy(),
            classwise_prediction_error=final_error.copy(),
            iteration_trace=iteration_trace,
            operation_trace=operation_trace,
            candidates=ranked,
            diagnostics=diagnostics,
            metadata=metadata,
        )
        result.operation_trace.append({
            "step": 40,
            "operation": N4_ALGORITHM_STEPS[39],
            "status": "returned",
            "result_schema": "ore.n4.attention_precision_gating.v1",
        })
        result.diagnostics["executed_operation_count"] = len(result.operation_trace)
        result.diagnostics["valid"] = bool(
            result.diagnostics["valid"] and len(result.operation_trace) == 40
        )
        return result
