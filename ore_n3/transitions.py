"""Smoothed unigram and bidirectional grapheme-transition model."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from typing import Iterable
import numpy as np

@dataclass(slots=True)
class OrthographicTransitionModel:
    class_names: tuple[str, ...]
    unigram: np.ndarray
    forward: np.ndarray
    reverse: np.ndarray
    fingerprint: str

    @classmethod
    def fit(cls, class_names: tuple[str, ...], sequences: Iterable[Iterable[str]],
            alpha: float, reference_labels: Iterable[str] = ()) -> "OrthographicTransitionModel":
        names = tuple(class_names)
        index = {name: i for i, name in enumerate(names)}
        size = len(names)
        unigram_counts = np.zeros(size, dtype=np.float64)
        bigram_counts = np.zeros((size, size), dtype=np.float64)
        for label in reference_labels:
            if label not in index:
                raise ValueError(f"reference label {label!r} is not in the visual vocabulary")
            unigram_counts[index[label]] += 1.0
        for sequence in sequences:
            tokens = list(sequence)
            for token in tokens:
                if token not in index:
                    raise ValueError(f"transition token {token!r} has no visual prototype")
                unigram_counts[index[token]] += 1.0
            for left, right in zip(tokens, tokens[1:]):
                bigram_counts[index[left], index[right]] += 1.0
        unigram = unigram_counts + alpha
        unigram /= unigram.sum()
        forward_counts = bigram_counts + alpha
        forward = forward_counts / forward_counts.sum(axis=1, keepdims=True)
        reverse_counts = bigram_counts + alpha
        reverse = reverse_counts / reverse_counts.sum(axis=0, keepdims=True)
        payload = {"classes": list(names), "unigram": unigram.tolist(),
                   "forward": forward.tolist(), "reverse": reverse.tolist(), "alpha": alpha}
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(names, unigram.astype(np.float32), forward.astype(np.float32),
                   reverse.astype(np.float32), fingerprint)

    def predict_from_left(self, left_belief: np.ndarray) -> np.ndarray:
        """P(current | uncertain left neighbor)."""
        return left_belief @ self.forward

    def predict_from_right(self, right_belief: np.ndarray) -> np.ndarray:
        """P(current | uncertain right neighbor)."""
        return self.reverse @ right_belief
