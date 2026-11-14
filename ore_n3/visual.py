"""N2-result feature extraction and a small N3 grapheme-population readout."""
from __future__ import annotations
import hashlib
import json
from collections.abc import Mapping
from typing import Any
import numpy as np
from .contracts import UNKNOWN_LABEL

_MAP_CHANNELS = (
    "contour", "boundary", "junction", "curvature",
    "crowding", "figure_probability", "invariant", "dorsal",
)

def _parts(value: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if hasattr(value, "representations"):
        return value.representations, getattr(value, "structured_output", {})
    if isinstance(value, Mapping):
        representations = value.get("representations", value)
        structured = value.get("structured_output", {})
        if not isinstance(representations, Mapping):
            raise TypeError("N2 representations must be a mapping")
        return representations, structured if isinstance(structured, Mapping) else {}
    raise TypeError("N2 patch input must be an N2Result or mapping")

def _summary(values: np.ndarray) -> list[float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise ValueError("N2 feature channels must be non-empty and finite")
    return [float(np.mean(flat)), float(np.std(flat)), float(np.min(flat)),
            float(np.max(flat)), float(np.quantile(flat, 0.90))]

def feature_vector_from_n2(value: Any) -> np.ndarray:
    """Extract a fixed 68-value descriptor from one current-repository N2 result.

    N2 produces visual-structure maps, not grapheme probabilities. N3 learns
    class prototypes from labelled N2 patch results and converts similarity
    scores into a visual candidate distribution.
    """
    representations, structured = _parts(value)
    if "ventral" not in representations:
        raise ValueError("N2 result is missing representations['ventral']")
    ventral = np.asarray(representations["ventral"], dtype=np.float32)
    if ventral.ndim != 3 or ventral.shape[0] < 1:
        raise ValueError("N2 ventral representation must have [channel, height, width] layout")
    descriptor: list[float] = []
    for channel in ventral:
        descriptor.extend(_summary(channel))
    for name in _MAP_CHANNELS:
        if name not in representations:
            raise ValueError(f"N2 result is missing representations[{name!r}]")
        values = np.asarray(representations[name], dtype=np.float32)
        if values.ndim < 1:
            raise ValueError(f"N2 representation {name!r} must be an array")
        descriptor.extend(_summary(values))
    if len(descriptor) != 20 + 5 * len(_MAP_CHANNELS):
        raise ValueError("unexpected N2 descriptor dimension")
    fragments = np.asarray(representations.get("shape_fragments", np.zeros((1, 1))))
    if fragments.size == 0:
        raise ValueError("shape_fragments cannot be empty")
    labels, counts = np.unique(fragments.astype(np.int64), return_counts=True)
    foreground = counts[labels != 0]
    descriptor.extend([
        float(foreground.size),
        float(foreground.max() / fragments.size) if foreground.size else 0.0,
        float(np.count_nonzero(fragments) / fragments.size),
    ])
    graph = structured.get("shape_graph", {}) if isinstance(structured, Mapping) else {}
    for key in ("node_count", "edge_count", "beta0", "beta1", "euler_characteristic"):
        descriptor.append(float(graph.get(key, 0.0)) if isinstance(graph, Mapping) else 0.0)
    result = np.asarray(descriptor, dtype=np.float32)
    if result.shape != (68,) or not np.isfinite(result).all():
        raise ValueError("N2 feature extraction did not produce a finite 68-value descriptor")
    return result

def extract_patch_features(*, n2_result: Any | None, features: np.ndarray | None) -> np.ndarray:
    if features is not None:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError("explicit features must be a non-empty finite one-dimensional vector")
        return array
    if n2_result is None:
        raise ValueError("patch requires N2 output or explicit features")
    return feature_vector_from_n2(n2_result)

class VisualPopulation:
    """Prototype population that maps N2 patch descriptors to candidate beliefs."""
    def __init__(self, temperature: float, unknown_threshold: float, unknown_temperature: float):
        self.temperature = float(temperature)
        self.unknown_threshold = float(unknown_threshold)
        self.unknown_temperature = float(unknown_temperature)
        self.class_names: tuple[str, ...] = ()
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.prototypes: np.ndarray | None = None
        self.fingerprint: str | None = None

    def fit(self, labels: list[str], features: np.ndarray) -> None:
        if len(labels) != features.shape[0] or not labels:
            raise ValueError("labels and reference feature rows must be non-empty and aligned")
        names = tuple(sorted(set(labels)))
        if UNKNOWN_LABEL in names:
            raise ValueError(f"{UNKNOWN_LABEL} is reserved")
        if len(names) < 2:
            raise ValueError("at least two grapheme classes are required for N3 training")
        self.class_names = names
        self.mean = features.mean(axis=0).astype(np.float32)
        raw_scale = features.std(axis=0).astype(np.float32)
        self.scale = np.where(raw_scale < 1e-6, 1.0, raw_scale).astype(np.float32)
        normalized = (features - self.mean) / self.scale
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        normalized = normalized / np.maximum(norms, 1e-8)
        prototypes = []
        for name in names:
            class_rows = normalized[np.asarray(labels) == name]
            prototype = class_rows.mean(axis=0)
            prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
            prototypes.append(prototype)
        self.prototypes = np.asarray(prototypes, dtype=np.float32)
        payload = {"classes": list(self.class_names), "mean": self.mean.tolist(),
                   "scale": self.scale.tolist(), "prototypes": self.prototypes.tolist()}
        self.fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def output_classes(self) -> tuple[str, ...]:
        if not self.class_names:
            raise RuntimeError("visual population has not been fitted")
        return (*self.class_names, UNKNOWN_LABEL)

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None or self.prototypes is None:
            raise RuntimeError("visual population has not been fitted")
        if features.ndim != 2 or features.shape[1] != self.mean.size:
            raise ValueError(
                f"query feature dimension {features.shape[-1] if features.ndim else 0} "
                f"does not match trained dimension {self.mean.size}"
            )
        normalized = (features - self.mean[None, :]) / self.scale[None, :]
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        normalized = normalized / np.maximum(norms, 1e-8)
        return normalized.astype(np.float32)

    def probabilities(self, normalized_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.prototypes is None:
            raise RuntimeError("visual population has not been fitted")
        similarities = normalized_features @ self.prototypes.T
        scaled = similarities / self.temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        known = np.exp(scaled)
        known /= known.sum(axis=1, keepdims=True)
        max_similarity = similarities.max(axis=1)
        confidence = 1.0 / (
            1.0 + np.exp(-np.clip(
                (max_similarity - self.unknown_threshold) / self.unknown_temperature,
                -60.0, 60.0,
            ))
        )
        output = np.zeros((known.shape[0], known.shape[1] + 1), dtype=np.float32)
        output[:, :-1] = known * confidence[:, None]
        output[:, -1] = 1.0 - confidence
        output /= np.maximum(output.sum(axis=1, keepdims=True), 1e-8)
        return output, similarities.astype(np.float32)
