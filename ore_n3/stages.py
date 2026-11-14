"""Explicit N3 numerical operations corresponding to the 29-step algorithm."""
from __future__ import annotations
import numpy as np

def normalize_rows(values: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), epsilon)
    return (values / values.sum(axis=-1, keepdims=True)).astype(np.float32)

def entropy_rows(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    p = np.maximum(probabilities, epsilon)
    return (-np.sum(p * np.log(p), axis=1)).astype(np.float32)

def softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(np.clip(shifted, -80.0, 80.0))
    return (exponentials / exponentials.sum(axis=1, keepdims=True)).astype(np.float32)

def precision_from_reliability(segmentation_confidence: np.ndarray, attention: np.ndarray,
                               base: float, attention_gain: float) -> np.ndarray:
    """Sensory precision: base*(0.5+0.5*segmentation)*(1+gain*attention)."""
    return (base * (0.5 + 0.5 * segmentation_confidence)
            * (1.0 + attention_gain * attention)).astype(np.float32)

def precision_from_prediction(visual_entropy: np.ndarray, segmentation_confidence: np.ndarray,
                              class_count: int, prediction_gain: float) -> np.ndarray:
    """Prior precision rises with visual uncertainty and segmentation uncertainty."""
    normalized_entropy = visual_entropy / max(np.log(max(class_count, 2)), 1e-8)
    return (prediction_gain * (0.5 + normalized_entropy)
            * (2.0 - segmentation_confidence)).astype(np.float32)

def contextual_prior(beliefs: np.ndarray, unigram: np.ndarray, forward: np.ndarray,
                     reverse: np.ndarray, unigram_weight: float,
                     neighbor_weight: float, epsilon: float) -> np.ndarray:
    """Combine unigram, left-to-right and right-to-left messages per sequence slot."""
    length, class_count = beliefs.shape
    result = np.zeros_like(beliefs, dtype=np.float32)
    for position in range(length):
        total = np.zeros(class_count, dtype=np.float64)
        weight_sum = 0.0
        if unigram_weight > 0:
            total += unigram_weight * unigram
            weight_sum += unigram_weight
        if position > 0 and neighbor_weight > 0:
            total += neighbor_weight * (beliefs[position - 1] @ forward)
            weight_sum += neighbor_weight
        if position + 1 < length and neighbor_weight > 0:
            total += neighbor_weight * (reverse @ beliefs[position + 1])
            weight_sum += neighbor_weight
        if weight_sum <= 0:
            total += unigram
            weight_sum = 1.0
        result[position] = total / weight_sum
    return normalize_rows(np.maximum(result, epsilon), epsilon)

def weighted_belief_update(visual_probabilities: np.ndarray, prior: np.ndarray,
                            sensory_precision: np.ndarray, prior_precision: np.ndarray,
                            epsilon: float) -> np.ndarray:
    logits = (sensory_precision[:, None] * np.log(np.maximum(visual_probabilities, epsilon))
              + prior_precision[:, None] * np.log(np.maximum(prior, epsilon)))
    return softmax_rows(logits)

def classwise_prediction_error(posterior: np.ndarray, prior: np.ndarray,
                               epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    terms = posterior * (np.log(np.maximum(posterior, epsilon))
                         - np.log(np.maximum(prior, epsilon)))
    return terms.astype(np.float32), np.sum(terms, axis=1).astype(np.float32)

def feature_prediction_residual(normalized_features: np.ndarray, prior: np.ndarray,
                                known_prototypes: np.ndarray) -> np.ndarray:
    known_prior = prior[:, : known_prototypes.shape[0]]
    predicted = known_prior @ known_prototypes
    return (normalized_features - predicted).astype(np.float32)

def energy_proxy(posterior: np.ndarray, visual_probabilities: np.ndarray, prior: np.ndarray,
                 sensory_precision: np.ndarray, prior_precision: np.ndarray,
                 epsilon: float) -> float:
    sensory_error = -np.sum(posterior * np.log(np.maximum(visual_probabilities, epsilon)), axis=1)
    divergence = np.sum(posterior * (np.log(np.maximum(posterior, epsilon))
                         - np.log(np.maximum(prior, epsilon))), axis=1)
    return float(np.mean(sensory_precision * sensory_error + prior_precision * divergence))
