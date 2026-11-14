"""The complete 52-operation ORE N5 experience-dependent plasticity algorithm."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from time import perf_counter
from typing import Any

import numpy as np

from .config import N5Config
from .contracts import N5Experience, N5LearningRequest, N5ModelState, N5Result, UNKNOWN_LABEL
from .stages import (
    bounded_oja_delta,
    clip_prototype_drift,
    feature_digest,
    finite_state_arrays,
    fingerprint_model,
    normalize_rows,
    normalize_vector,
)

N5_ALGORITHM_STEPS: tuple[str, ...] = (
    "01_validate_N5_learning_request",
    "02_require_explicit_user_authorization",
    "03_require_independent_label_verification",
    "04_validate_parent_model_schema",
    "05_verify_parent_model_fingerprint",
    "06_validate_parent_model_arrays_and_dimensions",
    "07_validate_grapheme_vocabulary",
    "08_validate_nonempty_learning_batch",
    "09_validate_experience_identity_and_uniqueness",
    "10_validate_per_example_provenance",
    "11_validate_feature_vectors_and_dimensions",
    "12_verify_optional_feature_digests",
    "13_validate_target_labels_and_sequence_positions",
    "14_snapshot_parent_state_for_immutability",
    "15_create_detached_child_model",
    "16_validate_optional_N4_result_contract",
    "17_align_N4_patch_attention_to_experiences",
    "18_extract_N4_sensory_precision",
    "19_extract_N4_class_selective_gates",
    "20_compute_reliability_and_attention_learning_gates",
    "21_standardize_experience_features",
    "22_normalize_experience_activity_vectors",
    "23_measure_class_prototype_similarity",
    "24_estimate_feature_novelty",
    "25_estimate_target_prediction_error",
    "26_compute_per_example_plasticity_weights",
    "27_bound_and_validate_plasticity_weights",
    "28_append_verified_examples_to_replay_memory",
    "29_enforce_replay_memory_capacity",
    "30_select_deterministic_replay_examples",
    "31_apply_replay_recency_and_importance_weights",
    "32_measure_target_class_population_activity",
    "33_compute_fast_Oja_prototype_updates",
    "34_update_fast_eligibility_traces",
    "35_bound_fast_synaptic_update_magnitude",
    "36_apply_and_normalize_fast_prototypes",
    "37_compute_slow_Oja_prototype_updates",
    "38_update_slow_eligibility_traces",
    "39_apply_and_normalize_slow_prototypes",
    "40_construct_effective_fast_slow_population",
    "41_measure_post_update_class_activity",
    "42_apply_homeostatic_class_gain_control",
    "43_enforce_parent_relative_prototype_drift_bounds",
    "44_update_verified_exposure_counters",
    "45_update_fast_unigram_experience_memory",
    "46_update_fast_forward_transition_memory",
    "47_update_fast_reverse_transition_memory",
    "48_update_slow_consolidated_transition_memory",
    "49_consolidate_high_confidence_replay_traces",
    "50_measure_learning_drift_and_interference_proxies",
    "51_validate_child_integrity_and_parent_immutability",
    "52_fingerprint_and_return_immutable_child_result",
)


class N5Processor:
    """Learn a detached child from verified examples without mutating its parent.

    Recognition remains read-only. Learning is a separate operation and is
    rejected unless explicit user authorization and independent verification
    are both present in the request.
    """

    def __init__(self, config: N5Config | None = None) -> None:
        self.config = config or N5Config()

    @staticmethod
    def initialize(
        n3_processor: Any, config: N5Config | None = None
    ) -> N5ModelState:
        """Snapshot a fitted N3Processor into a new independent N5 model state."""
        cfg = config or N5Config()
        return N5ModelState.from_n3_processor(
            n3_processor, prior_pseudocount_mass=cfg.prior_pseudocount_mass
        )

    def _validate_model(self, model: N5ModelState) -> tuple[int, int, int]:
        if not isinstance(model, N5ModelState):
            raise TypeError("parent_model must be an N5ModelState")
        if model.schema_version != "1.0" or model.version < 1:
            raise ValueError("unsupported N5 parent model version")

        names = tuple(model.class_names)
        if not names or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("parent model needs non-empty class names")
        if UNKNOWN_LABEL in names or len(set(names)) != len(names):
            raise ValueError("parent model grapheme vocabulary is invalid")

        classes = len(names)
        outputs = classes + 1
        mean = np.asarray(model.feature_mean)
        scale = np.asarray(model.feature_scale)
        if mean.ndim != 1 or mean.size < 1 or scale.shape != mean.shape:
            raise ValueError("feature_mean and feature_scale must be aligned one-dimensional arrays")
        dimension = int(mean.size)

        for name in (
            "fast_prototypes", "slow_prototypes",
            "fast_eligibility", "slow_eligibility",
        ):
            if np.asarray(getattr(model, name)).shape != (classes, dimension):
                raise ValueError(f"{name} must have shape ({classes}, {dimension})")
        for name in ("exposure_counts", "class_gains"):
            if np.asarray(getattr(model, name)).shape != (classes,):
                raise ValueError(f"{name} must have shape ({classes},)")
        for name in ("fast_unigram_counts", "slow_unigram_counts"):
            if np.asarray(getattr(model, name)).shape != (outputs,):
                raise ValueError(f"{name} must have shape ({outputs},)")
        for name in (
            "fast_forward_counts", "slow_forward_counts",
            "fast_reverse_counts", "slow_reverse_counts",
        ):
            if np.asarray(getattr(model, name)).shape != (outputs, outputs):
                raise ValueError(f"{name} must have shape ({outputs}, {outputs})")

        if not finite_state_arrays(model):
            raise ValueError("parent model contains non-finite numerical state")
        if np.any(scale <= 0):
            raise ValueError("parent feature scales must be positive")
        if np.any(np.asarray(model.exposure_counts) < 0):
            raise ValueError("parent exposure counts cannot be negative")
        if np.any(np.asarray(model.class_gains) <= 0):
            raise ValueError("parent class gains must be positive")
        for name in (
            "fast_unigram_counts", "slow_unigram_counts",
            "fast_forward_counts", "slow_forward_counts",
            "fast_reverse_counts", "slow_reverse_counts",
        ):
            if np.any(np.asarray(getattr(model, name)) < 0):
                raise ValueError(f"{name} cannot contain negative counts")
        if np.any(np.linalg.norm(model.fast_prototypes, axis=1) < self.config.epsilon):
            raise ValueError("fast prototype rows cannot be zero")
        if np.any(np.linalg.norm(model.slow_prototypes, axis=1) < self.config.epsilon):
            raise ValueError("slow prototype rows cannot be zero")
        if not model.fingerprint or fingerprint_model(model) != model.fingerprint:
            raise ValueError("parent model fingerprint mismatch")
        return classes, outputs, dimension

    def learn(self, request: N5LearningRequest) -> N5Result:
        started = perf_counter()
        cfg = self.config
        trace: list[dict[str, Any]] = []
        warnings: list[str] = []

        def mark(step: int, **summary: Any) -> None:
            trace.append({
                "step": step,
                "operation": N5_ALGORITHM_STEPS[step - 1],
                **summary,
            })

        # 01-03. Validate the call and enforce the two hard learning gates.
        if not isinstance(request, N5LearningRequest):
            raise TypeError("request must be an N5LearningRequest")
        mark(1, status="valid")
        if request.user_authorized is not True:
            raise PermissionError("N5 learning requires user_authorized=True")
        mark(2, user_authorized=True)
        if request.independently_verified is not True:
            raise PermissionError("N5 learning requires independently_verified=True")
        mark(3, independently_verified=True)

        # 04-07. Validate the parent snapshot and its class vocabulary.
        parent = request.parent_model
        if not isinstance(parent, N5ModelState):
            raise TypeError("parent_model must be an N5ModelState")
        mark(4, schema_version=parent.schema_version)
        class_count, output_count, dimension = self._validate_model(parent)
        mark(5, parent_fingerprint_valid=True)
        mark(6, feature_dimension=dimension, class_count=class_count)
        class_index = {label: index for index, label in enumerate(parent.class_names)}
        output_names = parent.output_class_names
        mark(7, class_names=list(parent.class_names))

        # 08-13. Validate examples, provenance, descriptors and sequence labels.
        experiences = list(request.experiences)
        if not experiences:
            raise ValueError("N5 requires at least one verified experience")
        if len(experiences) > cfg.max_experiences:
            raise ValueError("learning batch exceeds max_experiences")
        mark(8, experience_count=len(experiences))
        if any(not isinstance(item, N5Experience) for item in experiences):
            raise TypeError("experiences must contain N5Experience objects")

        identifiers = [item.experience_id for item in experiences]
        patch_keys = [(item.sequence_id, item.patch_id) for item in experiences]
        if any(not isinstance(value, str) or not value.strip() for value in identifiers):
            raise ValueError("experience_id values must be non-empty")
        if len(set(identifiers)) != len(identifiers) or len(set(patch_keys)) != len(patch_keys):
            raise ValueError("experience IDs and sequence/patch pairs must be unique")
        mark(9, unique_experience_count=len(set(identifiers)))

        for item in experiences:
            for field_name in (
                "patch_id", "sequence_id", "grapheme",
                "verified_by", "source_reference", "verified_at",
            ):
                value = getattr(item, field_name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{field_name} is required for every verified experience")
            if not np.isfinite(item.confidence) or not 0.0 <= item.confidence <= 1.0:
                raise ValueError("experience confidence must lie in [0, 1]")
        mark(10, provenance_records_valid=True)

        raw_features: list[np.ndarray] = []
        for item in experiences:
            vector = np.asarray(item.features, dtype=np.float32)
            if vector.ndim != 1 or vector.shape != (dimension,):
                raise ValueError(
                    f"features must be a one-dimensional vector of length {dimension}"
                )
            if not np.isfinite(vector).all():
                raise ValueError("experience features contain non-finite values")
            raw_features.append(vector.copy())
        mark(11, feature_matrix_shape=[len(raw_features), dimension])

        for item, vector in zip(experiences, raw_features):
            if item.feature_sha256 is not None and item.feature_sha256 != feature_digest(vector):
                raise ValueError(f"feature digest mismatch for experience {item.experience_id}")
        mark(12, supplied_feature_digests_checked=True)

        for item in experiences:
            if item.grapheme not in class_index:
                raise ValueError(f"unknown or non-trainable grapheme label: {item.grapheme!r}")
            if item.sequence_position is not None and (
                isinstance(item.sequence_position, bool)
                or not isinstance(item.sequence_position, int)
                or item.sequence_position < 0
            ):
                raise ValueError("sequence_position must be a non-negative integer")

        sequences: list[tuple[tuple[str, ...], float]] = []
        explicit_sequences = [tuple(tokens) for tokens in request.verified_sequences]
        if explicit_sequences:
            for tokens in explicit_sequences:
                if len(tokens) < 2:
                    continue
                if any(token not in class_index for token in tokens):
                    raise ValueError("verified sequence contains a label outside the parent vocabulary")
                sequences.append((tokens, 1.0))
        else:
            grouped: dict[str, list[N5Experience]] = defaultdict(list)
            for item in experiences:
                if item.sequence_position is not None:
                    grouped[item.sequence_id].append(item)
            for grouped_examples in grouped.values():
                positions = [item.sequence_position for item in grouped_examples]
                if len(set(positions)) != len(positions):
                    raise ValueError("sequence positions must be unique within each sequence")
                ordered = sorted(grouped_examples, key=lambda item: int(item.sequence_position))
                tokens = tuple(item.grapheme for item in ordered)
                if len(tokens) >= 2:
                    sequences.append((
                        tokens,
                        float(np.mean([item.confidence for item in ordered])),
                    ))
        mark(13, sequence_count=len(sequences), sequence_positions_valid=True)

        # 14-15. Snapshot all parent state and create an independent child.
        parent_before = fingerprint_model(parent)
        parent_array_names = (
            "feature_mean", "feature_scale", "fast_prototypes", "slow_prototypes",
            "fast_eligibility", "slow_eligibility", "exposure_counts", "class_gains",
            "fast_unigram_counts", "slow_unigram_counts", "fast_forward_counts",
            "slow_forward_counts", "fast_reverse_counts", "slow_reverse_counts",
        )
        parent_arrays = {
            key: np.asarray(getattr(parent, key)).copy() for key in parent_array_names
        }
        mark(14, parent_fingerprint=parent_before)
        child = parent.clone()
        child.parent_fingerprint = parent_before
        child.version = parent.version + 1
        mark(15, child_version=child.version)

        # 16-20. Optionally use N4 attention and precision as learning gates.
        count = len(experiences)
        attention_factor = np.ones(count, dtype=np.float32)
        sensory_factor = np.ones(count, dtype=np.float32)
        class_factor = np.ones(count, dtype=np.float32)
        n4_source = "not_supplied"
        if request.n4_result is not None:
            n4 = request.n4_result
            required = (
                "sequence_id", "class_names", "attention_allocation",
                "sensory_precision", "class_attention_gate", "candidates",
            )
            missing = [name for name in required if not hasattr(n4, name)]
            if missing:
                raise TypeError(f"N4 result is missing required fields: {missing}")
            if tuple(n4.class_names) != tuple(output_names):
                raise ValueError("N4 class vocabulary does not match the N5 parent model")
            if getattr(n4, "diagnostics", {}).get("valid", True) is not True:
                raise ValueError("N5 refuses an N4 result marked invalid")
            mark(16, n4_sequence_id=str(n4.sequence_id), status="valid")

            n4_candidates = list(n4.candidates or [])
            attention = np.asarray(n4.attention_allocation, dtype=np.float32)
            sensory = np.asarray(n4.sensory_precision, dtype=np.float32)
            gates = np.asarray(n4.class_attention_gate, dtype=np.float32)
            patch_count = len(n4_candidates)
            if (
                patch_count < 1
                or attention.shape != (patch_count,)
                or sensory.shape != (patch_count,)
            ):
                raise ValueError("N4 attention and precision arrays must align with candidate metadata")
            if gates.shape != (patch_count, output_count):
                raise ValueError("N4 class_attention_gate has an incompatible shape")
            if not all(np.isfinite(array).all() for array in (attention, sensory, gates)):
                raise ValueError("N4 learning cues contain non-finite values")
            if np.any(attention < 0) or np.any(sensory <= 0) or np.any(gates <= 0):
                raise ValueError("N4 attention, precision and class gates must be positive/non-negative")
            if float(attention.sum()) <= cfg.epsilon:
                raise ValueError("N4 attention allocation must have positive mass")
            if any(item.sequence_id != str(n4.sequence_id) for item in experiences):
                raise ValueError("N4-guided experiences must belong to the N4 sequence")

            patch_map: dict[str, int] = {}
            for index, candidate in enumerate(n4_candidates):
                patch_id = str(candidate.get("patch_id", ""))
                if not patch_id or patch_id in patch_map:
                    raise ValueError("N4 candidate patch IDs must be present and unique")
                patch_map[patch_id] = index
            mark(17, aligned_patch_count=len(patch_map))

            mean_sensory = max(float(np.mean(sensory)), cfg.epsilon)
            mean_gate = max(float(np.mean(gates)), cfg.epsilon)
            for index, item in enumerate(experiences):
                if item.patch_id not in patch_map:
                    raise ValueError(f"experience patch {item.patch_id!r} is absent from N4 result")
                row = patch_map[item.patch_id]
                attention_factor[index] = np.clip(
                    attention[row] * patch_count, 0.25, 2.0
                )
                sensory_factor[index] = np.clip(
                    sensory[row] / mean_sensory, 0.5, 2.0
                )
                class_factor[index] = np.clip(
                    gates[row, class_index[item.grapheme]] / mean_gate, 0.5, 2.0
                )
            n4_source = (
                "N4 attention_allocation, sensory_precision and class_attention_gate"
            )
            mark(18, mean_sensory_precision=float(np.mean(sensory)))
            mark(19, mean_target_class_gate=float(np.mean(class_factor)))
        else:
            mark(16, status="not_supplied")
            mark(17, alignment="not_applicable")
            mark(18, sensory_precision="neutral_default")
            mark(19, class_gate="neutral_default")

        if not cfg.use_n4_attention:
            attention_factor.fill(1.0)
            sensory_factor.fill(1.0)
            class_factor.fill(1.0)
        reliability = np.asarray([item.confidence for item in experiences], dtype=np.float32)
        learning_gate = np.clip(
            reliability * attention_factor * sensory_factor * class_factor,
            0.0, 2.5,
        )
        mark(20, mean_learning_gate=float(np.mean(learning_gate)))

        # 21-27. Standardize experiences and derive bounded plasticity weights.
        standardized = (
            np.stack(raw_features).astype(np.float32)
            - np.asarray(child.feature_mean, dtype=np.float32)[None, :]
        ) / np.asarray(child.feature_scale, dtype=np.float32)[None, :]
        if not np.isfinite(standardized).all():
            raise ValueError("standardized experience matrix is non-finite")
        mark(21, standardized_shape=list(standardized.shape))
        normalized = normalize_rows(standardized, cfg.epsilon)
        row_norms = np.linalg.norm(normalized, axis=1)
        if np.any(row_norms < cfg.epsilon):
            raise ValueError("an experience has no feature activity after standardization")
        mark(22, normalized_row_count=int(normalized.shape[0]))

        parent_effective = normalize_rows(
            0.75 * np.asarray(parent.fast_prototypes)
            + 0.25 * np.asarray(parent.slow_prototypes),
            cfg.epsilon,
        )
        similarities = normalized @ parent_effective.T
        if not np.isfinite(similarities).all():
            raise ValueError("prototype similarity calculation is non-finite")
        mark(23, mean_max_similarity=float(np.mean(np.max(similarities, axis=1))))
        novelty = np.clip(1.0 - np.max(similarities, axis=1), 0.0, 1.0)
        mark(24, mean_feature_novelty=float(np.mean(novelty)))

        target_indices = np.asarray(
            [class_index[item.grapheme] for item in experiences], dtype=np.int64
        )
        target_similarity = similarities[np.arange(count), target_indices]
        prediction_error = np.clip(0.5 * (1.0 - target_similarity), 0.0, 1.0)
        mark(25, mean_target_prediction_error=float(np.mean(prediction_error)))
        novelty_term = cfg.novelty_floor + cfg.novelty_gain * novelty
        error_term = (1.0 - cfg.novelty_gain) * (0.25 + 0.75 * prediction_error)
        raw_weights = learning_gate * (novelty_term + error_term)
        mark(26, mean_unbounded_update_weight=float(np.mean(raw_weights)))
        update_weights = np.clip(raw_weights, 0.0, 2.5).astype(np.float32)
        if float(update_weights.sum()) <= cfg.epsilon:
            raise ValueError("all verified experiences have zero effective learning weight")
        mark(27, maximum_update_weight=float(np.max(update_weights)))

        # 28-31. Store new verified examples and construct a deterministic replay set.
        previous_replay = deepcopy(child.replay_buffer)
        for item, vector, weight in zip(experiences, normalized, update_weights):
            child.replay_buffer.append({
                "experience_id": item.experience_id,
                "sequence_id": item.sequence_id,
                "patch_id": item.patch_id,
                "grapheme": item.grapheme,
                "features": vector.tolist(),
                "importance": float(weight),
                "verified_by": item.verified_by,
                "source_reference": item.source_reference,
                "verified_at": item.verified_at,
                "consolidated": False,
            })
        mark(28, appended_count=count)
        if len(child.replay_buffer) > cfg.replay_capacity:
            child.replay_buffer = child.replay_buffer[-cfg.replay_capacity:]
        mark(29, replay_buffer_size=len(child.replay_buffer))

        selected_replay = sorted(
            previous_replay,
            key=lambda record: (
                -float(record.get("importance", 0.0)),
                str(record.get("experience_id", "")),
            ),
        )[:cfg.replay_max_samples] if cfg.use_replay else []
        replay_records: list[tuple[np.ndarray, int, float]] = []
        for record in selected_replay:
            label = record.get("grapheme")
            vector = np.asarray(record.get("features"), dtype=np.float32)
            if (
                label not in class_index
                or vector.shape != (dimension,)
                or not np.isfinite(vector).all()
            ):
                continue
            replay_records.append((
                vector, class_index[label], float(record.get("importance", 0.0))
            ))
        mark(30, replay_selected=len(replay_records))
        weighted_replay: list[tuple[np.ndarray, int, float]] = []
        for age, (vector, label_index, importance) in enumerate(replay_records, start=1):
            weighted_replay.append((
                vector, label_index, importance * (cfg.replay_decay ** age)
            ))
        current_records = [
            (normalized[index], int(target_indices[index]), float(update_weights[index]))
            for index in range(count)
        ]
        training_records = current_records + weighted_replay
        mark(31, current_count=len(current_records), replay_count=len(weighted_replay))

        # 32-36. Fast-timescale Oja updates with decaying eligibility traces.
        pre_activity = [
            max(0.0, float(np.dot(child.fast_prototypes[label_index], vector)))
            for vector, label_index, _weight in training_records
        ]
        mark(32, mean_target_activity=float(np.mean(pre_activity)) if pre_activity else 0.0)
        fast_delta = np.zeros_like(child.fast_prototypes, dtype=np.float32)
        fast_counts = np.zeros(class_count, dtype=np.float32)
        for vector, label_index, weight in training_records:
            if weight <= 0:
                continue
            fast_delta[label_index] += bounded_oja_delta(
                vector, child.fast_prototypes[label_index],
                learning_rate=cfg.fast_learning_rate,
                weight=weight,
                epsilon=cfg.epsilon,
            )
            fast_counts[label_index] += 1.0
        for row in range(class_count):
            if fast_counts[row] > 0:
                fast_delta[row] /= fast_counts[row]
        mark(33, updated_class_count=int(np.count_nonzero(fast_counts)))
        child.fast_eligibility = (
            cfg.fast_trace_decay * child.fast_eligibility + fast_delta
        ).astype(np.float32)
        mark(34, fast_eligibility_norm=float(np.linalg.norm(child.fast_eligibility)))

        bounded_fast = child.fast_eligibility.copy()
        for row in range(class_count):
            norm = float(np.linalg.norm(bounded_fast[row]))
            if norm > cfg.max_step_norm:
                bounded_fast[row] *= cfg.max_step_norm / norm
        mark(35, maximum_fast_step=float(np.max(np.linalg.norm(bounded_fast, axis=1))))
        child.fast_prototypes = normalize_rows(
            child.fast_prototypes + bounded_fast, cfg.epsilon
        )
        mark(36, fast_prototype_norm_error=float(np.max(np.abs(
            np.linalg.norm(child.fast_prototypes, axis=1) - 1.0
        ))))

        # 37-40. Apply slower plasticity and construct the effective population.
        slow_delta = np.zeros_like(child.slow_prototypes, dtype=np.float32)
        slow_counts = np.zeros(class_count, dtype=np.float32)
        for vector, label_index, weight in training_records:
            if weight <= 0:
                continue
            slow_delta[label_index] += bounded_oja_delta(
                vector, child.slow_prototypes[label_index],
                learning_rate=cfg.slow_learning_rate,
                weight=weight,
                epsilon=cfg.epsilon,
            )
            slow_counts[label_index] += 1.0
        for row in range(class_count):
            if slow_counts[row] > 0:
                slow_delta[row] /= slow_counts[row]
        mark(37, slow_learning_rate=cfg.slow_learning_rate)
        child.slow_eligibility = (
            cfg.slow_trace_decay * child.slow_eligibility + slow_delta
        ).astype(np.float32)
        mark(38, slow_eligibility_norm=float(np.linalg.norm(child.slow_eligibility)))

        bounded_slow = child.slow_eligibility.copy()
        slow_step_limit = cfg.max_step_norm * 0.25
        for row in range(class_count):
            norm = float(np.linalg.norm(bounded_slow[row]))
            if norm > slow_step_limit:
                bounded_slow[row] *= slow_step_limit / norm
        child.slow_prototypes = normalize_rows(
            child.slow_prototypes + bounded_slow, cfg.epsilon
        )
        mark(39, slow_prototype_norm_error=float(np.max(np.abs(
            np.linalg.norm(child.slow_prototypes, axis=1) - 1.0
        ))))
        effective = normalize_rows(
            0.75 * child.fast_prototypes + 0.25 * child.slow_prototypes,
            cfg.epsilon,
        )
        mark(40, effective_population_shape=list(effective.shape))

        # 41-43. Apply homeostatic gain control and protect against excessive drift.
        post_activity_sum = np.zeros(class_count, dtype=np.float64)
        post_activity_count = np.zeros(class_count, dtype=np.float64)
        for vector, label_index, _weight in training_records:
            post_activity_sum[label_index] += max(
                0.0, float(np.dot(effective[label_index], vector))
            )
            post_activity_count[label_index] += 1.0
        post_activity = np.divide(
            post_activity_sum,
            np.maximum(post_activity_count, 1.0),
        )
        mark(41, mean_post_update_activity=float(np.mean(post_activity)))
        if cfg.use_homeostasis:
            child.class_gains *= np.exp(
                cfg.homeostatic_gain * (cfg.homeostatic_target - post_activity)
            )
            geometric_mean = float(np.exp(np.mean(
                np.log(np.maximum(child.class_gains, cfg.epsilon))
            )))
            child.class_gains /= max(geometric_mean, cfg.epsilon)
            child.class_gains = np.clip(
                child.class_gains, cfg.class_gain_min, cfg.class_gain_max
            )
        mark(
            42,
            class_gain_min=float(np.min(child.class_gains)),
            class_gain_max=float(np.max(child.class_gains)),
        )

        maximum_drift = 0.0
        for row in range(class_count):
            parent_fast = parent.fast_prototypes[row]
            parent_slow = parent.slow_prototypes[row]
            child.fast_prototypes[row] = clip_prototype_drift(
                child.fast_prototypes[row], parent_fast,
                cfg.max_prototype_drift, cfg.epsilon,
            )
            child.slow_prototypes[row] = clip_prototype_drift(
                child.slow_prototypes[row], parent_slow,
                cfg.max_prototype_drift, cfg.epsilon,
            )
            maximum_drift = max(
                maximum_drift,
                float(np.linalg.norm(child.fast_prototypes[row] - parent_fast)),
                float(np.linalg.norm(child.slow_prototypes[row] - parent_slow)),
            )
        mark(43, maximum_parent_relative_prototype_drift=maximum_drift)

        # 44-49. Update verified exposure and directional transition memories.
        for label_index, weight in zip(target_indices, update_weights):
            child.exposure_counts[int(label_index)] += float(weight)
        mark(44, total_new_exposure=float(np.sum(update_weights)))
        for label_index, weight in zip(target_indices, update_weights):
            child.fast_unigram_counts[int(label_index)] += float(weight)
        mark(45, fast_unigram_mass=float(np.sum(child.fast_unigram_counts)))

        transition_additions = np.zeros_like(child.fast_forward_counts, dtype=np.float64)
        for tokens, sequence_weight in sequences:
            for left, right in zip(tokens, tokens[1:]):
                transition_additions[class_index[left], class_index[right]] += (
                    cfg.transition_learning_rate * sequence_weight
                )
        child.fast_forward_counts += transition_additions
        mark(46, forward_transition_updates=float(np.sum(transition_additions)))
        child.fast_reverse_counts += transition_additions
        mark(47, reverse_transition_updates=float(np.sum(transition_additions)))

        if cfg.use_slow_consolidation:
            slow_unigram_additions = np.zeros(output_count, dtype=np.float64)
            for label_index, weight in zip(target_indices, update_weights):
                slow_unigram_additions[int(label_index)] += (
                    cfg.slow_consolidation_rate * float(weight)
                )
            child.slow_unigram_counts += slow_unigram_additions
            slow_transition_additions = (
                transition_additions * cfg.slow_consolidation_rate
            )
            child.slow_forward_counts += slow_transition_additions
            child.slow_reverse_counts += slow_transition_additions
        mark(
            48,
            slow_memory_rate=(
                cfg.slow_consolidation_rate if cfg.use_slow_consolidation else 0.0
            ),
        )

        consolidated_ids: list[str] = []
        for record in child.replay_buffer:
            if float(record.get("importance", 0.0)) >= cfg.consolidation_weight:
                record["consolidated"] = True
                consolidated_ids.append(str(record.get("experience_id", "")))
        mark(49, consolidated_count=len(consolidated_ids))

        # 50-52. Measure drift, validate integrity, fingerprint and package results.
        fast_drift = np.linalg.norm(
            child.fast_prototypes - parent.fast_prototypes, axis=1
        )
        slow_drift = np.linalg.norm(
            child.slow_prototypes - parent.slow_prototypes, axis=1
        )
        gain_drift = np.abs(child.class_gains - parent.class_gains)
        parent_changed = (
            fingerprint_model(parent) != parent_before
            or any(
                not np.array_equal(np.asarray(getattr(parent, key)), value)
                for key, value in parent_arrays.items()
            )
        )
        metrics = {
            "experience_count": count,
            "replay_count": len(weighted_replay),
            "updated_class_count": int(np.count_nonzero(fast_counts)),
            "total_exposure_increment": float(np.sum(update_weights)),
            "mean_feature_novelty": float(np.mean(novelty)),
            "mean_target_prediction_error": float(np.mean(prediction_error)),
            "mean_plasticity_weight": float(np.mean(update_weights)),
            "max_fast_prototype_drift": float(np.max(fast_drift)),
            "max_slow_prototype_drift": float(np.max(slow_drift)),
            "max_class_gain_drift": float(np.max(gain_drift)),
            "mean_class_gain": float(np.mean(child.class_gains)),
            "fast_unigram_mass": float(np.sum(child.fast_unigram_counts)),
            "fast_forward_transition_mass": float(np.sum(child.fast_forward_counts)),
            "fast_reverse_transition_mass": float(np.sum(child.fast_reverse_counts)),
        }
        if maximum_drift >= cfg.max_prototype_drift * 0.98:
            warnings.append("At least one prototype approached its configured drift bound.")
        mark(
            50,
            maximum_prototype_drift=maximum_drift,
            maximum_class_gain_drift=float(np.max(gain_drift)),
            parent_unchanged=not parent_changed,
        )

        child.version = parent.version + 1
        child.parent_fingerprint = parent_before
        if not finite_state_arrays(child):
            raise ValueError("child model contains non-finite state")
        if np.any(child.feature_scale <= 0) or np.any(child.class_gains <= 0):
            raise ValueError("child model violates positive scale/gain constraints")
        if np.any(child.exposure_counts < 0):
            raise ValueError("child model has negative exposure counts")
        if not np.allclose(
            np.linalg.norm(child.fast_prototypes, axis=1), 1.0, atol=1e-5
        ):
            raise ValueError("fast prototypes are not normalized")
        if not np.allclose(
            np.linalg.norm(child.slow_prototypes, axis=1), 1.0, atol=1e-5
        ):
            raise ValueError("slow prototypes are not normalized")
        if parent_changed:
            raise RuntimeError("N5 detected mutation of the frozen parent model")
        child.fingerprint = fingerprint_model(child)
        if fingerprint_model(child) != child.fingerprint:
            raise RuntimeError("child model fingerprint validation failed")
        mark(51, child_integrity_valid=True, parent_immutable=True)

        sequence_ids = tuple(sorted({item.sequence_id for item in experiences}))
        diagnostics = {
            "valid": True,
            "stage_count": 52,
            "step_names": list(N5_ALGORITHM_STEPS),
            "scientific_status": (
                "bounded experience-dependent plasticity approximation; not a measured "
                "synaptic learning rule or validated neural circuit model"
            ),
            "warnings": warnings,
            "elapsed_seconds": perf_counter() - started,
            "consent_gate": "passed",
            "independent_verification_gate": "passed",
            "parent_model_immutable": True,
            "n4_modulation_source": n4_source,
        }
        metadata = {
            "pipeline": "ORE N5 Experience-Dependent Plasticity",
            "version": "0.1.0",
            "parent_fingerprint": parent_before,
            "child_fingerprint": child.fingerprint,
            "child_version": child.version,
            "input_stage": "optional N4Result and verified N2 feature vectors",
            "learning_is_separate_from_recognition": True,
            "config": cfg.to_dict(),
            "limitations": [
                "Oja-style prototype updates are computational approximations, not measured synaptic laws.",
                "Replay and homeostatic gain control are explicit engineering mechanisms.",
                "Labels must be independently verified and learning explicitly authorized.",
                "Software tests do not establish biological validity or real-world OCR accuracy.",
            ],
        }
        mark(
            52,
            child_fingerprint=child.fingerprint,
            result_schema="ore.n5.experience_dependent_plasticity.v1",
        )
        result = N5Result(
            sequence_ids=sequence_ids,
            parent_fingerprint=parent_before,
            child_model=child,
            operation_trace=trace,
            diagnostics=diagnostics,
            metrics=metrics,
            metadata=metadata,
        )
        result.diagnostics["executed_operation_count"] = len(result.operation_trace)
        result.diagnostics["valid"] = (
            result.diagnostics["valid"] and len(result.operation_trace) == 52
        )
        return result
