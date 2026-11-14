"""N3 training and recurrent orthographic prediction pipeline (29 explicit steps)."""
from __future__ import annotations
from dataclasses import asdict
from typing import Any, Iterable
import time
import numpy as np

from .config import N3Config
from .contracts import LabeledN2Patch, N3Patch, N3Result, N3Sequence, UNKNOWN_LABEL
from .stages import (
    classwise_prediction_error, contextual_prior, energy_proxy, entropy_rows,
    feature_prediction_residual, normalize_rows, precision_from_prediction,
    precision_from_reliability, weighted_belief_update,
)
from .transitions import OrthographicTransitionModel
from .visual import VisualPopulation, extract_patch_features

N3_ALGORITHM_STEPS: tuple[str, ...] = (
    "01_validate_labelled_reference_patches",
    "02_extract_N2_visual_structure_descriptors",
    "03_validate_reference_feature_dimensions",
    "04_establish_grapheme_vocabulary",
    "05_estimate_feature_standardization",
    "06_normalize_reference_descriptors",
    "07_estimate_grapheme_population_prototypes",
    "08_normalize_population_prototypes",
    "09_count_unigram_exposures",
    "10_count_ordered_forward_bigrams",
    "11_count_ordered_reverse_bigrams",
    "12_smooth_and_normalize_unigram_prior",
    "13_normalize_forward_conditional_matrix",
    "14_normalize_reverse_conditional_matrix",
    "15_freeze_transition_model_and_fingerprint",
    "16_validate_recognition_sequence",
    "17_extract_query_N2_descriptors",
    "18_validate_query_contract_and_dimensions",
    "19_standardize_and_normalize_query_descriptors",
    "20_compute_grapheme_population_activations",
    "21_convert_activations_to_visual_beliefs_with_UNK",
    "22_measure_visual_entropy_and_uncertainty",
    "23_compute_sensory_precision",
    "24_compute_prior_precision",
    "25_initialize_recurrent_belief_state",
    "26_generate_bidirectional_contextual_predictions",
    "27_combine_visual_evidence_and_contextual_prior",
    "28_iterate_damped_belief_error_and_residual_updates",
    "29_finalize_structured_hypotheses_trace_and_diagnostics",
)

class N3Processor:
    """Train on labelled N2 patch outputs, then infer with recurrent context.

    Each N2Result must represent one already-segmented patch. The current N2
    pipeline produces visual-structure maps rather than grapheme distributions;
    N3 therefore learns a supervised prototype readout from labelled N2 outputs.
    It does not segment a complete word image or claim lexical/semantic access.
    """
    def __init__(self, config: N3Config | None = None) -> None:
        self.config = config or N3Config()
        self.visual_population = VisualPopulation(
            self.config.visual_temperature,
            self.config.unknown_similarity_threshold,
            self.config.unknown_temperature,
        )
        self.transition_model: OrthographicTransitionModel | None = None
        self.training_trace: list[dict[str, Any]] = []
        self._fitted = False

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.visual_population.output_classes

    @staticmethod
    def _feature_for_labeled(sample: LabeledN2Patch) -> np.ndarray:
        return extract_patch_features(n2_result=sample.n2_result, features=sample.features)

    @staticmethod
    def _feature_for_patch(patch: N3Patch) -> np.ndarray:
        return extract_patch_features(n2_result=patch.n2_result, features=patch.features)

    def fit(self, reference_patches: Iterable[LabeledN2Patch], *,
            grapheme_sequences: Iterable[Iterable[str]] = ()) -> "N3Processor":
        """Execute N3 learning operations 01-15 from labelled N2 patch observations."""
        started = time.perf_counter()
        samples = list(reference_patches)
        # 01. Validate the labelled reference set.
        if not samples:
            raise ValueError("N3 training requires at least one labelled reference patch")
        if any(not isinstance(item, LabeledN2Patch) for item in samples):
            raise TypeError("reference_patches must contain LabeledN2Patch objects")
        labels = [item.grapheme for item in samples]
        if any(not label.strip() for label in labels):
            raise ValueError("grapheme labels cannot be empty")
        self.training_trace = [{"step": 1, "operation": N3_ALGORITHM_STEPS[0],
                                "reference_count": len(samples)}]

        # 02. Extract N2 visual-structure descriptors.
        features = [self._feature_for_labeled(item) for item in samples]
        self.training_trace.append({"step": 2, "operation": N3_ALGORITHM_STEPS[1]})

        # 03. Validate common descriptor dimensions and finite values.
        dimensions = {int(row.size) for row in features}
        if len(dimensions) != 1:
            raise ValueError("all N2 reference descriptors must have the same dimension")
        matrix = np.stack(features).astype(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("reference descriptor matrix contains non-finite values")
        self.training_trace.append({
            "step": 3, "operation": N3_ALGORITHM_STEPS[2],
            "feature_dimension": int(matrix.shape[1]),
        })

        # 04. Establish the deterministic grapheme vocabulary.
        names = tuple(sorted(set(labels)))
        if len(names) < 2:
            raise ValueError("at least two distinct grapheme labels are required")
        self.training_trace.append({
            "step": 4, "operation": N3_ALGORITHM_STEPS[3], "class_count": len(names)
        })

        # 05-08. Estimate scaling statistics and construct normalized prototypes.
        self.visual_population.fit(labels, matrix)
        self.training_trace.extend([
            {"step": 5, "operation": N3_ALGORITHM_STEPS[4]},
            {"step": 6, "operation": N3_ALGORITHM_STEPS[5]},
            {"step": 7, "operation": N3_ALGORITHM_STEPS[6]},
            {"step": 8, "operation": N3_ALGORITHM_STEPS[7]},
        ])

        # 09-14. Learn unigram and bidirectional transition distributions.
        sequences = [tuple(sequence) for sequence in grapheme_sequences]
        self.training_trace.extend([
            {"step": 9, "operation": N3_ALGORITHM_STEPS[8]},
            {"step": 10, "operation": N3_ALGORITHM_STEPS[9]},
            {"step": 11, "operation": N3_ALGORITHM_STEPS[10]},
            {"step": 12, "operation": N3_ALGORITHM_STEPS[11]},
            {"step": 13, "operation": N3_ALGORITHM_STEPS[12]},
            {"step": 14, "operation": N3_ALGORITHM_STEPS[13]},
        ])
        self.transition_model = OrthographicTransitionModel.fit(
            (*names, UNKNOWN_LABEL), sequences, self.config.transition_alpha,
            reference_labels=labels,
        )

        # 15. Freeze a reproducible model identity; recognition is read-only.
        self._fitted = True
        self.training_trace.append({
            "step": 15, "operation": N3_ALGORITHM_STEPS[14],
            "visual_fingerprint": self.visual_population.fingerprint,
            "transition_fingerprint": self.transition_model.fingerprint,
            "elapsed_seconds": time.perf_counter() - started,
        })
        return self

    def recognize(self, sequence: N3Sequence) -> N3Result:
        """Execute N3 recognition operations 16-29 with bounded recurrent inference."""
        if not self._fitted or self.transition_model is None:
            raise RuntimeError("call fit() before recognize()")
        started = time.perf_counter()
        if not isinstance(sequence, N3Sequence):
            raise TypeError("sequence must be an N3Sequence")

        # 16. Validate sequence reliability metadata.
        patches = list(sequence.patches)
        confidence = np.asarray([p.segmentation_confidence for p in patches], dtype=np.float32)
        attention = np.asarray([p.attention for p in patches], dtype=np.float32)
        for name, values in (("segmentation_confidence", confidence), ("attention", attention)):
            if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError(f"{name} values must be finite and lie in [0, 1]")

        # 17. Extract one N2 descriptor per supplied patch.
        feature_rows = [self._feature_for_patch(patch) for patch in patches]
        # 18. Validate query dimensions and numerical integrity.
        expected_dimension = int(self.visual_population.mean.size)
        dimensions = {int(row.size) for row in feature_rows}
        if dimensions != {expected_dimension}:
            raise ValueError(
                f"query N2 descriptors must all have trained dimension {expected_dimension}"
            )
        raw_features = np.stack(feature_rows).astype(np.float32)
        if not np.isfinite(raw_features).all():
            raise ValueError("query descriptor matrix contains non-finite values")

        # 19. Standardize and normalize the query descriptor vectors.
        normalized_features = self.visual_population.transform(raw_features)
        # 20-21. Compute class activations and visual distributions including UNK.
        visual_probabilities, similarities = self.visual_population.probabilities(normalized_features)
        # 22. Quantify uncertainty before contextual feedback.
        visual_entropy = entropy_rows(visual_probabilities, self.config.epsilon)
        # 23-24. Calculate sensory and prior precision independently.
        sensory_precision = precision_from_reliability(
            confidence, attention, self.config.sensory_precision_base,
            self.config.attention_gain,
        )
        prior_precision = precision_from_prediction(
            visual_entropy, confidence, len(self.class_names),
            self.config.prediction_gain,
        )

        # 25. Initialize beliefs from bottom-up visual evidence.
        beliefs = normalize_rows(visual_probabilities, self.config.epsilon)
        initial_beliefs = beliefs.copy()
        final_prior = np.tile(self.transition_model.unigram, (len(patches), 1))
        final_error = np.zeros_like(beliefs)
        final_residual = np.zeros_like(normalized_features)
        trace: list[dict[str, Any]] = []
        converged = False
        mean_change = float("inf")
        mean_divergence = 0.0
        final_energy = float("nan")

        # 26-28. Predict bidirectionally, update, damp and retain prediction errors.
        for iteration in range(1, self.config.max_iterations + 1):
            prior = contextual_prior(
                beliefs, self.transition_model.unigram,
                self.transition_model.forward, self.transition_model.reverse,
                self.config.unigram_weight, self.config.neighbor_weight,
                self.config.epsilon,
            )
            proposed = weighted_belief_update(
                visual_probabilities, prior, sensory_precision,
                prior_precision, self.config.epsilon,
            )
            updated = normalize_rows(
                (1.0 - self.config.damping) * beliefs + self.config.damping * proposed,
                self.config.epsilon,
            )
            class_error, divergence = classwise_prediction_error(
                updated, prior, self.config.epsilon
            )
            residual = feature_prediction_residual(
                normalized_features, prior, self.visual_population.prototypes
            )
            mean_change = float(np.mean(np.abs(updated - beliefs)))
            mean_divergence = float(np.mean(divergence))
            final_energy = energy_proxy(
                updated, visual_probabilities, prior, sensory_precision,
                prior_precision, self.config.epsilon,
            )
            trace.append({
                "iteration": iteration,
                "mean_belief_change": mean_change,
                "mean_prediction_divergence": mean_divergence,
                "energy_proxy": final_energy,
                "mean_feature_residual_norm": float(
                    np.mean(np.linalg.norm(residual, axis=1))
                ),
            })
            beliefs, final_prior = updated, prior
            final_error, final_residual = class_error, residual
            if mean_change <= self.config.convergence_tolerance:
                converged = True
                break

        # 29. Package ranked alternatives, intermediate states and diagnostics.
        row_sums = beliefs.sum(axis=1)
        finite = all(np.isfinite(array).all() for array in (
            visual_probabilities, initial_beliefs, final_prior, beliefs,
            final_error, final_residual, similarities,
        ))
        candidates: list[dict[str, Any]] = []
        for row, patch in zip(beliefs, patches):
            order = np.argsort(row)[::-1][:self.config.top_k]
            candidates.append({
                "patch_id": patch.patch_id,
                "hypotheses": [
                    {"grapheme": self.class_names[int(i)], "probability": float(row[i])}
                    for i in order
                ],
                "top_candidate": self.class_names[int(order[0])],
                "top_probability": float(row[order[0])],
                "segmentation_confidence": float(patch.segmentation_confidence),
                "attention": float(patch.attention),
            })
        diagnostics = {
            "valid": bool(finite and np.allclose(row_sums, 1.0, atol=1e-5)),
            "stage_count": 29,
            "step_names": list(N3_ALGORITHM_STEPS),
            "iterations": len(trace),
            "converged": converged,
            "stop_reason": "converged" if converged else "maximum_iterations_reached",
            "mean_belief_change": mean_change,
            "mean_prediction_divergence": mean_divergence,
            "energy_proxy": final_energy,
            "mean_feature_residual_norm": float(np.mean(np.linalg.norm(final_residual, axis=1))),
            "probability_row_sum_error": float(np.max(np.abs(row_sums - 1.0))),
            "finite_outputs": bool(finite),
            "elapsed_seconds": time.perf_counter() - started,
            "scientific_status": (
                "algorithmic predictive-processing approximation; not a validated neural circuit model"
            ),
        }
        if not converged:
            diagnostics["warning"] = (
                "Inference reached max_iterations without satisfying convergence_tolerance."
            )
        metadata = {
            "pipeline": "ORE N3 Recurrent Orthographic Prediction and Error",
            "version": "0.1.0",
            "config": asdict(self.config),
            "training_trace": self.training_trace,
            "sequence_id": sequence.sequence_id,
            "position_count": len(patches),
            "feature_dimension": int(normalized_features.shape[1]),
            "class_names": list(self.class_names),
            "n2_feature_source": "labelled per-patch N2 visual-structure descriptors",
            "visual_model_fingerprint": self.visual_population.fingerprint,
            "transition_model_fingerprint": self.transition_model.fingerprint,
            "logical_reading_order": "input patch order; caller/script adapter is responsible for ordering",
            "semantic_scope": "grapheme-sequence hypotheses only; no lexical or semantic correction",
        }
        return N3Result(
            sequence_id=sequence.sequence_id,
            class_names=self.class_names,
            visual_probabilities=visual_probabilities,
            initial_beliefs=initial_beliefs,
            contextual_priors=final_prior,
            posterior_beliefs=beliefs,
            classwise_prediction_error=final_error,
            feature_residuals=final_residual,
            trace=trace,
            candidates=candidates,
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def process(self, sequence: N3Sequence) -> N3Result:
        """Alias for recognize(), matching N1/N2 processor naming."""
        return self.recognize(sequence)
