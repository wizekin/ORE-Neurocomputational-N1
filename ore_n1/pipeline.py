"""Orchestrator for the complete 12-operation ORE N1 pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import N1Config
from .diagnostics import DiagnosticRecorder, timed_call
from .stages import (
    as_grayscale_float, center_surround, complex_cell_energy,
    construct_feature_vector, divisive_normalization, lgn_gain_control,
    photoreceptor_adaptation, retinal_on_off, simple_cell_responses,
    spatial_pooling, v1_orientation_population,
)
from .temporal import TemporalDynamics


@dataclass(slots=True)
class N1Result:
    """Complete processing result, retaining intermediate arrays for analysis."""
    feature_vector: np.ndarray
    pooled_features: np.ndarray
    stage_outputs: dict[str, np.ndarray]
    metadata: dict[str, Any]
    diagnostics: dict[str, Any]


class N1Processor:
    """Stateful processor; temporal state is retained only when enabled."""

    def __init__(self, config: N1Config | None = None) -> None:
        self.config = config or N1Config()
        self._temporal = TemporalDynamics(self.config.temporal_tau)

    def reset_temporal_state(self) -> None:
        self._temporal.reset()

    def process(
        self,
        image: np.ndarray,
        *,
        dt: float = 1.0 / 30.0,
        reset_temporal: bool = False,
    ) -> N1Result:
        """Run operations 1-12 on a single grayscale image.

        Floating inputs must be in [0, 1]; integer arrays are scaled by their
        dtype maximum. dt is used only when temporal dynamics are enabled.
        """
        if reset_temporal:
            self.reset_temporal_state()
        cfg = self.config
        frame = as_grayscale_float(image)
        if cfg.pooling_grid[0] > frame.shape[0] or cfg.pooling_grid[1] > frame.shape[1]:
            raise ValueError("pooling_grid cannot exceed the input image dimensions")
        diagnostics = DiagnosticRecorder(strict=cfg.strict_numerics)
        outputs: dict[str, np.ndarray] = {"input": frame}
        diagnostics.record("00.input", frame)

        # 1. Photoreceptor adaptation.
        (photo, adaptation), elapsed = timed_call(photoreceptor_adaptation, frame, cfg)
        outputs["photoreceptor_response"] = photo
        outputs["adaptation_field"] = adaptation
        diagnostics.record("01.photoreceptor_response", photo, elapsed)
        diagnostics.record("01.adaptation_field", adaptation)

        # 2. Retinal ON/OFF split.
        (retinal_on, retinal_off, signed_contrast), elapsed = timed_call(retinal_on_off, photo, cfg)
        outputs.update(retinal_on=retinal_on, retinal_off=retinal_off, retinal_signed_contrast=signed_contrast)
        diagnostics.record("02.retinal_on", retinal_on, elapsed)
        diagnostics.record("02.retinal_off", retinal_off)
        diagnostics.record("02.retinal_signed_contrast", signed_contrast)

        # 3. Center-surround ganglion-like responses.
        (ganglion_on, ganglion_off), elapsed = timed_call(center_surround, retinal_on, retinal_off, cfg)
        outputs.update(ganglion_on=ganglion_on, ganglion_off=ganglion_off)
        diagnostics.record("03.ganglion_on", ganglion_on, elapsed)
        diagnostics.record("03.ganglion_off", ganglion_off)

        # 4. LGN local contrast-gain control.
        (lgn_on, lgn_off, lgn_gain), elapsed = timed_call(lgn_gain_control, ganglion_on, ganglion_off, cfg)
        outputs.update(lgn_on=lgn_on, lgn_off=lgn_off, lgn_gain=lgn_gain)
        diagnostics.record("04.lgn_on", lgn_on, elapsed)
        diagnostics.record("04.lgn_off", lgn_off)
        diagnostics.record("04.lgn_gain", lgn_gain)

        # 5. V1 orientation-selective even/odd linear population.
        linear, elapsed = timed_call(v1_orientation_population, lgn_on, lgn_off, cfg)
        outputs["v1_linear_responses"] = linear
        diagnostics.record("05.v1_linear_responses", linear, elapsed)

        # 6. Phase- and polarity-specific simple-cell-like responses.
        simple, elapsed = timed_call(simple_cell_responses, linear)
        outputs["simple_cell_responses"] = simple
        diagnostics.record("06.simple_cell_responses", simple, elapsed)

        # 7. Quadrature energy, approximately phase tolerant.
        complex_energy, elapsed = timed_call(complex_cell_energy, simple)
        outputs["complex_cell_energy"] = complex_energy
        diagnostics.record("07.complex_cell_energy", complex_energy, elapsed)

        # 8. Orientation-population divisive normalization.
        (normalized, denominator), elapsed = timed_call(divisive_normalization, complex_energy, cfg)
        outputs.update(normalized_energy=normalized, normalization_denominator=denominator)
        diagnostics.record("08.normalized_energy", normalized, elapsed)
        diagnostics.record("08.normalization_denominator", denominator)

        # 11. Optional dynamics is applied to the normalized neural population
        # before spatial pooling. When disabled, this operation is an identity.
        if cfg.temporal_enabled:
            temporal = self._temporal.update(normalized, dt)
        else:
            temporal = normalized.copy()
        outputs["temporal_activity"] = temporal
        diagnostics.record("11.temporal_activity", temporal)

        # 9-10. Spatial readout and deterministic feature construction.
        pooled, elapsed = timed_call(spatial_pooling, temporal, cfg.pooling_grid)
        outputs["spatial_pooled"] = pooled
        diagnostics.record("09.spatial_pooled", pooled, elapsed)
        features, elapsed = timed_call(construct_feature_vector, pooled)
        outputs["feature_vector"] = features
        diagnostics.record("10.feature_vector", features, elapsed)

        # 12. Validate every output. The diagnostics object itself is not
        # self-recorded, avoiding recursive diagnostics.
        for key, value in outputs.items():
            if not np.isfinite(value).all():
                raise FloatingPointError(f"final validation failed for {key}")
        diagnostic_dict = diagnostics.to_dict()
        metadata = {
            "pipeline": "ORE Neurocomputational N1",
            "version": "0.1.0",
            "input_shape": tuple(int(v) for v in frame.shape),
            "orientation_angles_radians": tuple(float(a) for a in cfg.orientations),
            "orientation_count": cfg.orientation_count,
            "pooling_grid": tuple(int(v) for v in cfg.pooling_grid),
            "feature_dimension": int(features.size),
            "temporal_enabled": bool(cfg.temporal_enabled),
            "temporal_time_constant_seconds": float(cfg.temporal_tau),
            "biological_status": "biologically motivated approximation; not a validated physiological model",
        }
        return N1Result(
            feature_vector=features, pooled_features=pooled,
            stage_outputs=outputs, metadata=metadata, diagnostics=diagnostic_dict,
        )
