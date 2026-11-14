"""Numerical helpers for ORE N5 plasticity and model integrity."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def normalize_vector(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm):
        raise ValueError("cannot normalize a non-finite vector")
    return (vector / max(norm, epsilon)).astype(np.float32)


def normalize_rows(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        raise ValueError("cannot normalize non-finite rows")
    return (matrix / np.maximum(norms, epsilon)).astype(np.float32)


def normalize_distribution(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("distribution must be a finite non-negative vector")
    total = float(array.sum())
    if total <= epsilon:
        raise ValueError("distribution must have positive mass")
    return (array / total).astype(np.float32)


def fingerprint_model(model: Any) -> str:
    """Hash all persisted model state, excluding its own fingerprint."""
    payload = model.to_dict(include_fingerprint=False)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_digest(features: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(features, dtype=np.float32))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def bounded_oja_delta(
    vector: np.ndarray,
    prototype: np.ndarray,
    *,
    learning_rate: float,
    weight: float,
    epsilon: float,
) -> np.ndarray:
    """A bounded Oja-style update for one target-selective prototype."""
    x = normalize_vector(vector, epsilon)
    w = normalize_vector(prototype, epsilon)
    activity = float(np.clip(0.5 + 0.5 * np.dot(w, x), 0.05, 1.0))
    delta = learning_rate * weight * activity * (x - activity * w)
    norm = float(np.linalg.norm(delta))
    return (delta / max(norm, epsilon) * min(norm, 1.0)).astype(np.float32)


def clip_prototype_drift(
    candidate: np.ndarray, parent: np.ndarray, maximum: float, epsilon: float
) -> np.ndarray:
    """Clip angular prototype movement so its final chord drift stays bounded."""
    base = normalize_vector(parent, epsilon)
    proposed = normalize_vector(candidate, epsilon)
    maximum_chord = min(max(float(maximum), 0.0), 2.0)
    cosine = float(np.clip(np.dot(base, proposed), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    maximum_angle = 2.0 * float(np.arcsin(0.5 * maximum_chord))
    if angle <= maximum_angle + 1e-12:
        return proposed
    if maximum_angle <= epsilon:
        return base

    sine = float(np.sin(angle))
    if abs(sine) <= epsilon:
        if base.size < 2:
            return base
        basis = np.zeros_like(base)
        basis[int(np.argmin(np.abs(base)))] = 1.0
        orthogonal = basis - float(np.dot(basis, base)) * base
        orthogonal_norm = float(np.linalg.norm(orthogonal))
        if orthogonal_norm <= epsilon:
            return base
        orthogonal = orthogonal / orthogonal_norm
        clipped = (
            np.cos(maximum_angle) * base
            + np.sin(maximum_angle) * orthogonal
        )
    else:
        fraction = maximum_angle / angle
        clipped = (
            np.sin((1.0 - fraction) * angle) * base
            + np.sin(fraction * angle) * proposed
        ) / sine
    return normalize_vector(clipped, epsilon)


def finite_state_arrays(model: Any) -> bool:
    names = (
        "feature_mean", "feature_scale", "fast_prototypes", "slow_prototypes",
        "fast_eligibility", "slow_eligibility", "exposure_counts", "class_gains",
        "fast_unigram_counts", "slow_unigram_counts", "fast_forward_counts",
        "slow_forward_counts", "fast_reverse_counts", "slow_reverse_counts",
    )
    return all(np.isfinite(np.asarray(getattr(model, name))).all() for name in names)


def transition_probabilities(
    forward_counts: np.ndarray,
    reverse_counts: np.ndarray,
    alpha: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    forward = np.asarray(forward_counts, dtype=np.float64) + alpha
    reverse = np.asarray(reverse_counts, dtype=np.float64) + alpha
    forward /= np.maximum(forward.sum(axis=1, keepdims=True), epsilon)
    reverse /= np.maximum(reverse.sum(axis=0, keepdims=True), epsilon)
    return forward.astype(np.float32), reverse.astype(np.float32)
