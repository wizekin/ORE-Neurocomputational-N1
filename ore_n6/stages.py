"""Numerical and graph helpers for ORE N6 sequence/lexical processing."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def normalize_rows(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("probability matrix must be finite, two-dimensional, and non-negative")
    totals = array.sum(axis=1, keepdims=True)
    if np.any(totals <= epsilon):
        raise ValueError("probability matrix contains a zero-mass row")
    return (array / totals).astype(np.float32)


def stable_softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64) / float(temperature)
    if not np.isfinite(logits).all():
        raise ValueError("softmax logits must be finite")
    logits -= np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(np.clip(logits, -745.0, 0.0))
    return (exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-300)).astype(np.float32)


def normalize_vector(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all() or np.any(vector < 0):
        raise ValueError("vector must be finite, one-dimensional, and non-negative")
    total = float(vector.sum())
    if total <= epsilon:
        raise ValueError("vector must have positive mass")
    return (vector / total).astype(np.float32)


def n5_fingerprint(model: Any) -> str:
    """Recompute the N5 state fingerprint using N5's canonical JSON contract."""
    payload = model.to_dict(include_fingerprint=False)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def logit(probability: float, epsilon: float = 1e-8) -> float:
    p = float(np.clip(probability, epsilon, 1.0 - epsilon))
    return float(np.log(p) - np.log1p(-p))


def path_key(path: list[dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (int(edge["start"]), int(edge["end"]),
         "" if edge["entry_id"] is None else str(edge["entry_id"]))
        for edge in path
    )


def k_best_paths(
    edge_lists: list[list[dict[str, Any]]],
    sequence_length: int,
    top_k: int,
) -> list[tuple[float, list[dict[str, Any]]]]:
    """Exact k-best DAG paths for a forward-only segmentation lattice."""
    paths: list[list[tuple[float, list[dict[str, Any]]]]] = [
        [] for _ in range(sequence_length + 1)
    ]
    paths[0] = [(0.0, [])]
    for start in range(sequence_length):
        if not paths[start]:
            continue
        for edge in edge_lists[start]:
            end = int(edge["end"])
            if end <= start or end > sequence_length:
                raise ValueError("segmentation edge has an invalid span")
            for base_score, base_path in paths[start]:
                candidate_path = [*base_path, edge]
                paths[end].append((base_score + float(edge["score"]), candidate_path))
            paths[end].sort(
                key=lambda item: (-item[0], path_key(item[1]))
            )
            unique: list[tuple[float, list[dict[str, Any]]]] = []
            seen: set[tuple[tuple[Any, ...], ...]] = set()
            for candidate in paths[end]:
                key = path_key(candidate[1])
                if key in seen:
                    continue
                seen.add(key)
                unique.append(candidate)
                if len(unique) >= top_k:
                    break
            paths[end] = unique
    if not paths[sequence_length]:
        raise RuntimeError("segmentation lattice contains no complete path")
    return paths[sequence_length]


def entropy_rows(probabilities: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    return (-np.sum(p * np.log(np.maximum(p, epsilon)), axis=1)).astype(np.float32)
