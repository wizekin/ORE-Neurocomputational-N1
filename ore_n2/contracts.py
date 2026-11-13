"""N1-to-N2 contracts. N2 requires preserved spatial maps, never a reconstructed vector."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np

@dataclass(slots=True)
class N2Input:
    feature_maps: np.ndarray
    source_metadata: dict[str, Any]
    source_diagnostics: dict[str, Any]

@dataclass(slots=True)
class N2Result:
    structured_output: dict[str, Any]
    representations: dict[str, np.ndarray]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

def adapt_n1_result(value: Any) -> N2Input:
    """Accept N1Result or a mapping exposing N1 stage_outputs."""
    if hasattr(value, "stage_outputs"):
        outputs = value.stage_outputs
        metadata = dict(getattr(value, "metadata", {}) or {})
        diagnostics = dict(getattr(value, "diagnostics", {}) or {})
    elif isinstance(value, Mapping):
        outputs = value.get("stage_outputs", value)
        metadata = dict(value.get("metadata", {}) or {})
        diagnostics = dict(value.get("diagnostics", {}) or {})
    else:
        raise TypeError("N2 input must be an N1Result or mapping with stage_outputs")
    if "normalized_energy" not in outputs:
        raise ValueError("N2 requires N1 stage_outputs['normalized_energy'] spatial maps")
    maps = np.asarray(outputs["normalized_energy"], dtype=np.float32)
    if maps.ndim != 3 or min(maps.shape) < 1:
        raise ValueError("normalized_energy must have shape [orientation, height, width]")
    if not np.isfinite(maps).all():
        raise ValueError("N1 spatial maps contain non-finite values")
    return N2Input(np.ascontiguousarray(maps), metadata, diagnostics)
