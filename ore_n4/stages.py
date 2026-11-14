"""Numerical primitives for ORE N4 attention and precision gating."""
from __future__ import annotations

import numpy as np


def stable_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exps = np.exp(np.clip(shifted, -80.0, 80.0))
    return (exps / np.maximum(exps.sum(axis=axis, keepdims=True), 1e-12)).astype(np.float32)


def normalized_entropy(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    p = np.maximum(np.asarray(probabilities, dtype=np.float64), epsilon)
    p /= p.sum(axis=1, keepdims=True)
    entropy = -np.sum(p * np.log(p), axis=1)
    denom = max(float(np.log(max(p.shape[1], 2))), epsilon)
    return np.clip(entropy / denom, 0.0, 1.0).astype(np.float32)


def normalize_signal(values: np.ndarray, epsilon: float) -> np.ndarray:
    """Robustly scale a non-negative vector to [0, 1]."""
    x = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    if x.size == 0:
        return x.astype(np.float32)
    scale = float(np.quantile(x, 0.95))
    if scale <= epsilon:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(x / scale, 0.0, 1.0).astype(np.float32)


def symmetric_kl_rows(left: np.ndarray, right: np.ndarray, epsilon: float) -> np.ndarray:
    p = np.maximum(np.asarray(left, dtype=np.float64), epsilon)
    q = np.maximum(np.asarray(right, dtype=np.float64), epsilon)
    p /= p.sum(axis=1, keepdims=True)
    q /= q.sum(axis=1, keepdims=True)
    kl_pq = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    kl_qp = np.sum(q * (np.log(q) - np.log(p)), axis=1)
    return (0.5 * (kl_pq + kl_qp)).astype(np.float32)


def normalize_distribution(values: np.ndarray, epsilon: float) -> np.ndarray:
    x = np.maximum(np.asarray(values, dtype=np.float64), epsilon)
    total = float(x.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("cannot normalize an empty or invalid distribution")
    return (x / total).astype(np.float32)


def smooth_spatial_attention(allocation: np.ndarray, strength: float) -> np.ndarray:
    """Mix each position with its immediate neighbors without wrapping edges."""
    x = np.asarray(allocation, dtype=np.float64)
    if x.size <= 1 or strength <= 0:
        return x.astype(np.float32)
    left = np.concatenate(([x[0]], x[:-1]))
    right = np.concatenate((x[1:], [x[-1]]))
    local = 0.25 * left + 0.5 * x + 0.25 * right
    mixed = (1.0 - strength) * x + strength * local
    mixed = np.maximum(mixed, 0.0)
    return (mixed / max(float(mixed.sum()), 1e-12)).astype(np.float32)


def damped_belief_update(
    current: np.ndarray,
    visual: np.ndarray,
    prior: np.ndarray,
    sensory_precision: np.ndarray,
    prior_precision: np.ndarray,
    class_attention: np.ndarray,
    damping: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    visual_logits = (
        sensory_precision[:, None]
        * class_attention
        * np.log(np.maximum(visual, epsilon))
    )
    prior_logits = prior_precision[:, None] * np.log(np.maximum(prior, epsilon))
    proposal = stable_softmax(visual_logits + prior_logits, axis=1)
    updated = (1.0 - damping) * current + damping * proposal
    updated = np.maximum(updated, epsilon)
    updated /= updated.sum(axis=1, keepdims=True)
    return updated.astype(np.float32), proposal.astype(np.float32)


def rank_hypotheses(
    beliefs: np.ndarray,
    class_names: tuple[str, ...],
    patch_ids: list[str],
    segmentation_confidence: np.ndarray,
    attention_allocation: np.ndarray,
    top_k: int,
    abstain_threshold: float,
    ambiguity_margin: float,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for i, row in enumerate(beliefs):
        order = np.argsort(row)[::-1][: min(top_k, len(class_names))]
        top_idx = int(order[0])
        top_prob = float(row[top_idx])
        second_prob = float(row[int(order[1])]) if len(order) > 1 else 0.0
        margin = top_prob - second_prob
        if top_prob < abstain_threshold:
            decision = "ABSTAIN"
        elif margin < ambiguity_margin:
            decision = "AMBIGUOUS"
        else:
            decision = "ATTENTION_SUPPORTED"
        results.append({
            "patch_id": patch_ids[i],
            "hypotheses": [
                {"class": class_names[int(j)], "probability": float(row[int(j)])}
                for j in order
            ],
            "top_candidate": class_names[top_idx],
            "top_probability": top_prob,
            "runner_up_probability": second_prob,
            "decision": decision,
            "segmentation_confidence": float(segmentation_confidence[i]),
            "attention_allocation": float(attention_allocation[i]),
        })
    return results
