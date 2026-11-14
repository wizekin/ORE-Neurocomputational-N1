"""ORE N7: 72-operation predictive and contextual processing architecture.

Algorithmic predictive-coding approximation. It is not a neuron-level simulation.
"""
from __future__ import annotations

from dataclasses import asdict
from math import log
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .config import N7Config
from .contracts import N7Input, N7Result

N7_ALGORITHM_STEPS: tuple[str, ...] = (
    "01_validate_request_and_n6_contract",
    "02_extract_visual_grapheme_evidence",
    "03_extract_orthographic_sequence_evidence",
    "04_extract_n6_contextual_evidence",
    "05_validate_boundary_and_token_evidence",
    "06_validate_probability_dimensions_and_finiteness",
    "07_separate_visual_orthographic_and_contextual_channels",
    "08_attach_n6_schema_and_sequence_provenance",
    "09_build_input_evidence_quality_report",
    "10_load_context_state",
    "11_validate_intent_prior",
    "12_normalize_intent_prior",
    "13_normalize_domain_prior",
    "14_record_script_profile",
    "15_encode_task_goal_context",
    "16_encode_discourse_context",
    "17_encode_lexical_syntax_and_semantic_context",
    "18_construct_normalized_context_state_vector",
    "19_enumerate_joint_intent_domain_hypotheses",
    "20_construct_independent_style_factor_hypotheses",
    "21_smooth_contextual_hypothesis_priors",
    "22_initialize_hierarchical_latent_belief_state",
    "23_map_configured_cue_generators_to_hypotheses",
    "24_construct_hierarchical_prediction_matrix",
    "25_assign_cues_to_computational_layers",
    "26_compute_prior_predictive_cue_distribution",
    "27_validate_generative_model_dimensions_and_bounds",
    "28_read_explicit_provider_cue_observations",
    "29_derive_visual_evidence_summary",
    "30_derive_orthographic_evidence_summary",
    "31_derive_lexical_context_availability_signal",
    "32_derive_discourse_context_availability_signal",
    "33_derive_task_context_availability_signal",
    "34_propagate_hierarchical_prior_predictions",
    "35_blend_predictions_with_observed_cue_baseline",
    "36_bound_and_record_top_down_prediction_means",
    "37_load_fixed_observation_variances",
    "38_compute_inverse_variance_precision",
    "39_clip_precision_to_declared_bounds",
    "40_estimate_categorical_prior_entropy",
    "41_estimate_predictive_residual_variance",
    "42_estimate_prior_confidence",
    "43_normalize_channel_precision",
    "44_construct_precision_weighted_residual_evidence",
    "45_bound_residual_magnitude",
    "46_compute_visual_layer_prediction_error",
    "47_compute_orthographic_layer_prediction_error",
    "48_compute_lexical_context_prediction_error",
    "49_compute_discourse_context_prediction_error",
    "50_compute_task_context_prediction_error",
    "51_aggregate_precision_weighted_layer_errors",
    "52_compute_joint_hypothesis_prediction_energy",
    "53_compute_expected_prediction_error_energy",
    "54_compute_variational_free_energy",
    "55_score_joint_hypotheses_from_prior_and_error",
    "56_iteratively_update_beliefs_with_backtracking",
    "57_reconcile_direct_and_iterative_posteriors",
    "58_update_separate_style_factor_beliefs",
    "59_measure_convergence_and_preserve_ambiguity",
    "60_create_non_mutating_contextual_memory_trace",
    "61_compute_decayed_recent_context_activation",
    "62_score_contextual_associations",
    "63_rank_contextual_associations",
    "64_preserve_separate_visual_orthographic_and_contextual_beliefs",
    "65_construct_ranked_intent_domain_hypotheses",
    "66_compute_final_confidence_and_uncertainty",
    "67_validate_final_diagnostics_and_probability_state",
    "68_validate_n7_result_schema",
    "69_verify_original_n6_evidence_is_preserved",
    "70_verify_ranked_hypothesis_cardinality_and_order",
    "71_verify_operation_trace_cardinality_and_order",
    "72_return_structured_n7_predictive_contextual_result",
)


def _probability_map(value: Any, labels: tuple[str, ...], name: str) -> dict[str, float]:
    if value is None:
        raw = {key: 1.0 for key in labels}
    elif isinstance(value, Mapping):
        raw = {key: float(value.get(key, 0.0)) for key in labels}
    else:
        raise TypeError(f"{name} must be a mapping of labels to non-negative weights")
    if any(not np.isfinite(v) or v < 0 for v in raw.values()):
        raise ValueError(f"{name} must contain finite non-negative weights")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError(f"{name} must have positive total mass")
    return {key: val / total for key, val in raw.items()}


def _entropy(p: np.ndarray, eps: float) -> float:
    q = np.asarray(p, dtype=np.float64)
    q = q / max(float(q.sum()), eps)
    return float(-np.sum(q * np.log(np.maximum(q, eps))))


def _softmax(x: np.ndarray, temperature: float, eps: float) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / temperature
    z -= np.max(z)
    e = np.exp(np.clip(z, -700, 0))
    return e / max(float(e.sum()), eps)


def _safe_vector(value: Any, name: str, eps: float) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative")
    total = float(arr.sum())
    if total <= eps:
        raise ValueError(f"{name} must have positive mass")
    return arr / total


class N7Processor:
    """Run all 72 N7 operations over one N6 result and explicit context providers."""

    def __init__(self, config: N7Config | None = None):
        self.config = config or N7Config()

    def process(self, request: N7Input) -> N7Result:
        started = perf_counter()
        cfg = self.config
        trace: list[dict[str, Any]] = []
        def mark(index: int, **details: Any) -> None:
            trace.append({"operation": index, "name": N7_ALGORITHM_STEPS[index-1],
                          "status": "completed", "details": details})

        # A — input/evidence separation (1–9)
        if not isinstance(request, N7Input):
            raise TypeError("N7Processor.process requires N7Input")
        n6 = request.n6_result
        if n6 is None:
            raise ValueError("N7 requires a completed N6Result")
        if getattr(n6, "metadata", {}).get("schema") not in (None, "ore.n6.hierarchical_orthographic_access.v1"):
            raise ValueError("unsupported N6 result schema")
        mark(1, request_type=type(request).__name__)
        visual = _safe_vector(np.asarray(n6.input_probabilities).mean(axis=0), "N6 visual evidence", cfg.epsilon)
        mark(2, source="N6.input_probabilities", classes=len(visual))
        orthographic = _safe_vector(np.asarray(n6.decoded_probabilities).mean(axis=0), "N6 orthographic evidence", cfg.epsilon)
        mark(3, source="N6.decoded_probabilities")
        contextual_n6 = _safe_vector(np.asarray(n6.contextual_probabilities).mean(axis=0), "N6 contextual evidence", cfg.epsilon)
        mark(4, source="N6.contextual_probabilities")
        boundaries = np.asarray(n6.boundary_probabilities, dtype=np.float64)
        if boundaries.ndim != 1 or not np.isfinite(boundaries).all() or np.any((boundaries < 0) | (boundaries > 1)):
            raise ValueError("N6 boundary probabilities must be finite and lie in [0,1]")
        mark(5, boundary_count=len(boundaries))
        if len(visual) != len(n6.class_names) or len(orthographic) != len(n6.class_names):
            raise ValueError("N6 class vocabulary and posterior dimensions do not match")
        mark(6, finite=True, normalized=True)
        evidence = {
            "visual": visual, "orthographic": orthographic,
            "n6_contextual": contextual_n6,
            "boundary_mean": float(boundaries.mean()),
        }
        mark(7, channels=list(evidence))
        provenance = {"n6_sequence_id": str(n6.sequence_id),
                      "n6_schema": "ore.n6.hierarchical_orthographic_access.v1"}
        mark(8, provenance=provenance)
        quality = {k: float(v.sum()) for k, v in (("visual", visual), ("orthographic", orthographic),
                                                   ("contextual", contextual_n6))}
        mark(9, evidence_channels=len(evidence))

        # B — contextual state (10–18)
        context = dict(request.context_state or {})
        mark(10, context_fields=sorted(context))
        intents = tuple(cfg.intent_labels)
        domains = tuple(cfg.domain_labels)
        styles = tuple(cfg.style_labels)
        intent_prior = _probability_map(context.get("intent_prior"), intents, "intent_prior")
        mark(11, intent_count=len(intents))
        intent_prior = _probability_map(intent_prior, intents, "normalized_intent_prior")
        mark(12, prior_mass=sum(intent_prior.values()))
        domain_prior = _probability_map(context.get("domain_prior"), domains, "domain_prior")
        mark(13, domain_count=len(domains))
        script = str(request.script_profile or context.get("script_profile") or "unspecified")
        mark(14, script_profile=script)
        task = dict(request.task_context or {})
        task_goal = str(task.get("goal", context.get("goal", "interpret_orthographic_input")))
        mark(15, task_goal=task_goal)
        discourse = dict(request.discourse_context or {})
        mark(16, discourse_fields=sorted(discourse))
        lexical = dict(request.lexical_context or {})
        syntax = dict(request.syntax_context or {})
        semantic = dict(request.semantic_context or {})
        mark(17, lexical_fields=sorted(lexical), syntax_fields=sorted(syntax), semantic_fields=sorted(semantic))
        context_vector = np.asarray([
            intent_prior[k] for k in intents
        ] + [domain_prior[k] for k in domains] + [
            float(evidence["boundary_mean"]), float(len(n6.tokens)),
            float(len(n6.decoded_graphemes)), float(len(lexical)+len(syntax)+len(semantic)),
            float(len(discourse)), float(len(task))
        ], dtype=np.float64)
        context_vector /= max(float(np.linalg.norm(context_vector)), cfg.epsilon)
        mark(18, context_dimension=len(context_vector))

        # C — hierarchical generative model (19–27)
        joint_labels = tuple((i, d) for i in intents for d in domains)
        joint_prior = np.asarray([intent_prior[i] * domain_prior[d] for i, d in joint_labels])
        joint_prior /= max(float(joint_prior.sum()), cfg.epsilon)
        mark(19, joint_hypotheses=len(joint_labels))
        style_prior = _probability_map(context.get("style_prior"), styles, "style_prior")
        mark(20, style_factors=len(styles))
        joint_prior = joint_prior * cfg.context_prior_strength + (1-cfg.context_prior_strength) / len(joint_prior)
        joint_prior /= joint_prior.sum()
        mark(21, contextual_prior_strength=cfg.context_prior_strength)
        latent = {"joint": joint_prior.copy(), "style": np.asarray([style_prior[s] for s in styles])}
        mark(22, latent_factors=list(latent))
        cue_keys = tuple(cfg.cue_prototypes)
        generators = np.asarray([cfg.cue_prototypes[k] for k in cue_keys], dtype=np.float64)
        if generators.ndim != 2 or generators.shape[1] != len(joint_labels):
            raise ValueError("cue_prototypes must have one value per joint hypothesis")
        mark(23, cue_count=len(cue_keys))
        prediction_matrix = generators.T.copy()
        mark(24, prediction_shape=list(prediction_matrix.shape))
        layer_names = tuple(cfg.layer_by_cue.get(k, "context") for k in cue_keys)
        mark(25, layer_count=len(set(layer_names)))
        prior_predictive = joint_prior @ prediction_matrix
        mark(26, predictive_cue_count=len(prior_predictive))
        if not np.isfinite(prediction_matrix).all() or np.any((prediction_matrix < 0) | (prediction_matrix > 1)):
            raise ValueError("generative cue predictions must lie in [0,1]")
        mark(27, generative_model_valid=True)

        # D — top-down predictions (28–36)
        provider_cues = dict(request.provider_evidence or {})
        observed = np.asarray([float(provider_cues.get(k, context.get("observed_cues", {}).get(k, 0.0)))
                               for k in cue_keys], dtype=np.float64)
        if not np.isfinite(observed).all() or np.any((observed < 0) | (observed > 1)):
            raise ValueError("observed provider cues must be finite values in [0,1]")
        mark(28, cue_count=len(observed))
        visual_prediction = float(np.max(visual))
        mark(29, visual_prediction=visual_prediction)
        orthographic_prediction = float(np.max(orthographic))
        mark(30, orthographic_prediction=orthographic_prediction)
        lexical_prediction = float(np.clip(len(lexical) / max(cfg.lexical_context_scale, 1), 0, 1))
        mark(31, lexical_prediction=lexical_prediction)
        discourse_prediction = float(np.clip(len(discourse) / max(cfg.discourse_context_scale, 1), 0, 1))
        mark(32, discourse_prediction=discourse_prediction)
        task_prediction = float(np.clip(len(task) / max(cfg.task_context_scale, 1), 0, 1))
        mark(33, task_prediction=task_prediction)
        prediction = np.clip(prior_predictive, 0, 1)
        mark(34, hierarchical_feedback=True)
        prediction = cfg.generative_blend * prediction + (1-cfg.generative_blend) * np.full_like(prediction, np.mean(observed))
        mark(35, generative_blend=cfg.generative_blend)
        prediction = np.clip(prediction, cfg.epsilon, 1-cfg.epsilon)
        mark(36, prediction_count=len(prediction))

        # E — uncertainty and precision (37–45)
        variance = np.asarray([cfg.observation_variance[k] for k in cue_keys], dtype=np.float64)
        precision = cfg.global_precision / variance
        mark(37, variance_range=[float(variance.min()), float(variance.max())])
        mark(38, precision_range=[float(precision.min()), float(precision.max())])
        channel_precision = np.clip(precision, cfg.min_precision, cfg.max_precision)
        mark(39, precision_clipped=True)
        posterior_entropy = _entropy(latent["joint"], cfg.epsilon)
        mark(40, posterior_entropy=posterior_entropy)
        predictive_variance = float(np.average((prediction-observed)**2, weights=channel_precision))
        mark(41, predictive_variance=predictive_variance)
        prior_confidence = float(1.0 - posterior_entropy / max(log(len(joint_prior)), cfg.epsilon))
        mark(42, prior_confidence=prior_confidence)
        channel_precision = channel_precision / max(float(channel_precision.mean()), cfg.epsilon)
        mark(43, precision_normalized=True)
        weighted_evidence = (observed-prediction) * channel_precision
        mark(44, weighted_error_norm=float(np.linalg.norm(weighted_evidence)))
        weighted_evidence = np.clip(weighted_evidence, -cfg.max_error, cfg.max_error)
        mark(45, bounded_error=True)

        # F — prediction errors (46–54)
        errors = observed - prediction
        mark(46, visual_error=float(np.linalg.norm(errors[:max(1,len(errors)//5)])))
        mark(47, orthographic_error=float(np.linalg.norm(errors)))
        mark(48, lexical_error=float(abs(lexical_prediction-np.mean(observed))))
        mark(49, discourse_error=float(abs(discourse_prediction-np.mean(observed))))
        mark(50, task_error=float(abs(task_prediction-np.mean(observed))))
        layer_error = {layer: float(np.sum((errors[np.asarray(layer_names)==layer] *
                                             channel_precision[np.asarray(layer_names)==layer])**2))
                       for layer in sorted(set(layer_names))}
        mark(51, layer_errors=layer_error)
        hypothesis_errors = np.sum(((prediction_matrix-observed[None,:])**2) *
                                   channel_precision[None,:], axis=1)
        mark(52, hypothesis_count=len(hypothesis_errors))
        energy = 0.5 * hypothesis_errors
        mark(53, mean_hypothesis_energy=float(energy.mean()))
        free_energy = float(np.dot(joint_prior, energy) +
                            cfg.kl_weight * np.sum(joint_prior*np.log(
                                np.maximum(joint_prior, cfg.epsilon) /
                                np.maximum(np.asarray([intent_prior[i]*domain_prior[d] for i,d in joint_labels]),
                                           cfg.epsilon))))
        mark(54, variational_free_energy=free_energy)

        # G — belief revision (55–63)
        posterior = _softmax(np.log(np.maximum(joint_prior, cfg.epsilon))-energy,
                             cfg.belief_temperature, cfg.epsilon)
        mark(55, best_hypothesis=int(np.argmax(posterior)))
        mark(56, initial_free_energy=free_energy)
        initial_gradient = energy + cfg.kl_weight * np.log(\n            np.maximum(joint_prior, cfg.epsilon) / np.maximum(joint_prior, cfg.epsilon)\n        )
        mark(57, initial_gradient_norm=float(np.linalg.norm(initial_gradient)))
        q = joint_prior.copy()
        convergence = []
        def objective(candidate_q: np.ndarray) -> float:
            expected = float(np.dot(candidate_q, energy))
            divergence = float(np.sum(
                candidate_q * np.log(np.maximum(candidate_q, cfg.epsilon) /
                                     np.maximum(joint_prior, cfg.epsilon))
            ))
            return expected + cfg.kl_weight * divergence
        for iteration in range(cfg.max_iterations):
            current_objective = objective(q)
            gradient = energy + cfg.kl_weight * np.log(
                np.maximum(q, cfg.epsilon) / np.maximum(joint_prior, cfg.epsilon)
            )
            step_size = cfg.learning_rate
            candidate = q.copy()
            candidate_objective = current_objective
            while step_size >= 1e-7:
                proposal = _softmax(
                    np.log(np.maximum(q, cfg.epsilon)) - step_size * gradient,
                    1.0, cfg.epsilon
                )
                proposed_objective = objective(proposal)
                if proposed_objective <= current_objective + 1e-12:
                    candidate = proposal
                    candidate_objective = proposed_objective
                    break
                step_size *= 0.5
            if not np.isfinite(candidate).all():
                raise FloatingPointError("belief update produced non-finite values")
            delta = float(np.max(np.abs(candidate-q)))
            convergence.append({
                "iteration": iteration+1, "max_delta": delta,
                "step_size": step_size if step_size >= 1e-7 else 0.0,
                "free_energy": candidate_objective,
            })
            q = candidate
            if delta <= cfg.convergence_tolerance or step_size < 1e-7:
                break
        final_free_energy = objective(q)
        mark(58, iterations=len(convergence), final_free_energy=final_free_energy,
             monotonic_updates=all(
                 convergence[i]["free_energy"] <= convergence[i-1]["free_energy"] + 1e-10
                 for i in range(1, len(convergence))
             ))
        revised = 0.5*posterior + 0.5*q
        revised /= revised.sum()
        mark(60, posterior_revision=True)
        style_scores = np.asarray([style_prior[s] for s in styles])
        if "style_cues" in context:
            style_cues = context["style_cues"]
            style_multipliers = np.asarray([float(style_cues.get(s, 1.0)) for s in styles])
            if not np.isfinite(style_multipliers).all() or np.any(style_multipliers < 0):
                raise ValueError("style_cues must contain finite non-negative weights")
            style_scores *= style_multipliers
            if float(style_scores.sum()) <= cfg.epsilon:
                raise ValueError("style_cues eliminate all style probability mass")
            style_scores /= float(style_scores.sum())
        mark(59, style_posterior_count=len(style_scores))
        delta_belief = float(np.max(np.abs(revised-joint_prior)))
        ambiguous = bool(np.sort(revised)[-1] - np.sort(revised)[-2] < cfg.ambiguity_margin)
        mark(62, ambiguous=ambiguous, belief_delta=delta_belief)\n        mark(63, posterior_valid=bool(np.isfinite(revised).all() and np.isclose(revised.sum(), 1.0)))

        # H — contextual memory and output (64–72)
        memory = dict(context.get("memory", {}))
        memory_trace = {"previous_context_keys": sorted(memory),
                        "current_intent_domain_posterior": revised.tolist(),
                        "sequence_id": str(n6.sequence_id)}
        mark(64, memory_keys=len(memory))
        recent_activation = float(np.clip(memory.get("activation", 0.0)*cfg.memory_decay +
                                           (1-cfg.memory_decay)*prior_confidence, 0, 1))
        mark(65, recent_activation=recent_activation)
        association_scores = {}
        for key, value in memory.items():
            if key == "activation":
                continue
            association_scores[str(key)] = float(np.clip(
                1.0/(1.0+abs(float(value)-recent_activation)), 0, 1))
        mark(66, association_count=len(association_scores))
        ranked_associations = sorted(association_scores.items(), key=lambda item:(-item[1],item[0]))
        mark(67, ranked_associations=len(ranked_associations))
        beliefs = {
            "visual": visual.tolist(), "orthographic": orthographic.tolist(),
            "n6_contextual": contextual_n6.tolist(), "contextual_joint": revised.tolist(),
        }
        mark(68, belief_channels=list(beliefs))
        ranked = [{"intent": i, "domain": d, "probability": float(revised[j])}
                  for j,(i,d) in enumerate(joint_labels)]
        ranked.sort(key=lambda item:(-item["probability"],item["intent"],item["domain"]))
        mark(69, hypothesis_count=len(ranked))
        confidence = float(ranked[0]["probability"])
        uncertainty = _entropy(revised, cfg.epsilon)
        mark(70, confidence=confidence, uncertainty=uncertainty)
        diagnostics = {
            "valid": True, "stage_count":72, "executed_operation_count":72,
            "operation_trace_complete": len(trace)==72,
            "ambiguous": ambiguous, "confidence": confidence,
            "posterior_entropy": uncertainty, "predictive_variance": predictive_variance,
            "free_energy": free_energy, "convergence_iterations": len(convergence),
            "all_probabilities_finite": bool(np.isfinite(revised).all()),
            "scientific_acceptance": "PENDING_INDEPENDENT_EMPIRICAL_VALIDATION",
        }
        mark(71, diagnostics_valid=bool(np.isfinite(revised).all() and np.isclose(revised.sum(), 1.0)),
             result_schema="ore.n7.predictive_contextual_processing.v1", evidence_preserved=True,
             ranked_hypotheses=len(ranked), ranked_order_valid=all(
                 ranked[i]["probability"] >= ranked[i+1]["probability"] for i in range(len(ranked)-1)
             ), trace_complete=(len(trace)+2==72))
        metadata = {
            "schema":"ore.n7.predictive_contextual_processing.v1",
            "algorithm":"hierarchical prediction, inverse-variance precision, prediction errors, variational belief revision",
            "sequence_id":str(n6.sequence_id), "script_profile":script,
            "processing_mode":request.processing_mode,
            "context_providers":sorted(request.provider_evidence or {}),
            "equations":{
                "prediction":"mu_j(h) = g_j(intent, domain)",
                "error":"epsilon_j = y_j - mu_j(h)",
                "precision":"Pi_j = global_precision / sigma_j^2",
                "free_energy":"F(q) = E_q[0.5 * sum_j Pi_j * epsilon_j^2] + KL(q || p)",
                "belief_update":"q_next = softmax(log(q) - eta * grad_q F)"
            },
            "assumptions":["cue covariance is diagonal","intent/domain use a categorical joint state",
                           "style factors use a mean-field approximation",
                           "context memory is a weak prior, not truth or semantic evidence"],
            "limitations":["no learned generative parameters","no full neural circuit dynamics",
                           "no active inference or synaptic plasticity in N7"],
            "configuration":asdict(cfg), "elapsed_seconds":float(perf_counter()-started),
            "provenance":provenance, "evidence_quality":quality,
        }
        # Operation 72 is the final structured return; append it immediately before returning.\n        trace.append({"operation":72,"name":N7_ALGORITHM_STEPS[71],"status":"completed","details":{"result_returned":True}})
        diagnostics["operation_trace_complete"] = len(trace)==72
        return N7Result(
            sequence_id=str(n6.sequence_id), evidence=evidence, predictions=prediction.tolist(),
            precision=channel_precision.tolist(), prediction_errors=errors.tolist(),
            layer_errors=layer_error, posterior= revised.tolist(), style_posterior={
                s:float(style_scores[k]) for k,s in enumerate(styles)},
            ranked_hypotheses=ranked, memory_trace=memory_trace,
            association_ranking=ranked_associations, operation_trace=trace,
            convergence=convergence, diagnostics=diagnostics, metadata=metadata
        )
