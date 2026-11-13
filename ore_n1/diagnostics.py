"""Stage-by-stage diagnostics and numerical validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(slots=True)
class StageRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    finite: bool
    elapsed_ms: float


class DiagnosticRecorder:
    """Collect compact numerical summaries for every named pipeline output."""

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict
        self.records: list[StageRecord] = []
        self.warnings: list[str] = []

    def record(self, name: str, values: np.ndarray, elapsed_ms: float = 0.0) -> None:
        arr = np.asarray(values)
        finite = bool(np.isfinite(arr).all())
        if not finite:
            message = f"{name} contains NaN or infinite values"
            if self.strict:
                raise FloatingPointError(message)
            self.warnings.append(message)
        if arr.size == 0:
            raise ValueError(f"{name} produced an empty array")
        safe = arr[np.isfinite(arr)]
        if safe.size == 0:
            minimum = maximum = mean = std = float("nan")
        else:
            minimum = float(safe.min())
            maximum = float(safe.max())
            mean = float(safe.mean())
            std = float(safe.std())
        self.records.append(StageRecord(
            name=name, shape=tuple(int(x) for x in arr.shape), dtype=str(arr.dtype),
            minimum=minimum, maximum=maximum, mean=mean,
            standard_deviation=std, finite=finite, elapsed_ms=float(elapsed_ms),
        ))

    def validate_range(self, name: str, values: np.ndarray, low: float, high: float) -> None:
        arr = np.asarray(values)
        if not np.isfinite(arr).all():
            raise FloatingPointError(f"{name} is not finite")
        if np.any(arr < low) or np.any(arr > high):
            raise ValueError(f"{name} must lie in [{low}, {high}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": all(record.finite for record in self.records),
            "stage_count": len(self.records),
            "warnings": list(self.warnings),
            "stages": [asdict(record) for record in self.records],
        }


def timed_call(function, *args, **kwargs):
    """Run a stage and return (result, elapsed milliseconds)."""
    start = perf_counter()
    result = function(*args, **kwargs)
    return result, (perf_counter() - start) * 1000.0
