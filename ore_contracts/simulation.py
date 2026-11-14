"""Reference rate-based IF-18 simulator for canonical B-N1 foundations.

This is a deterministic computational substrate, not a claim of cellular or
physiological equivalence. Units and model class are explicit in every snapshot.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Sequence

import numpy as np

from .advanced import (
    NeuralInputEvent, NeuralSimulationConfig, NeuralSimulator, NeuralStateSnapshot,
)


class RateBasedNeuralSimulator:
    """Small deterministic leaky-population simulator with explicit state units.

    Membrane voltage uses mV. Population firing-rate output uses Hz. Recurrent
    weights are interpreted as mV/Hz. This rate model emits no spike events.
    """

    version = "0.1.0"

    def __init__(self, config: NeuralSimulationConfig) -> None:
        self.config = config
        self._connectivity_fingerprint = sha256(
            np.ascontiguousarray(config.recurrent_weights).tobytes()
            + repr(config.population_sizes).encode("utf-8")
        ).hexdigest()
        self._population_indices: dict[str, tuple[int, ...]] = {}
        offset = 0
        for population_id, size in config.population_sizes:
            self._population_indices[population_id] = tuple(range(offset, offset + size))
            offset += size

    def _rates(self, voltage: np.ndarray) -> np.ndarray:
        z = np.clip(
            self.config.sigmoid_slope_per_mv * (voltage - self.config.threshold_mv),
            -60.0, 60.0,
        )
        return (self.config.max_rate_hz / (1.0 + np.exp(-z))).astype(np.float32)

    def initialize(self, config: NeuralSimulationConfig | None = None) -> NeuralStateSnapshot:
        if config is not None and config is not self.config:
            raise ValueError("simulator configuration is immutable after construction")
        voltage = np.full(
            self.config.unit_count, self.config.resting_potential_mv, dtype=np.float32
        )
        return NeuralStateSnapshot(
            simulation_time_ms=0.0,
            step_index=0,
            dt_ms=self.config.dt_ms,
            population_unit_indices=self._population_indices,
            membrane_potential_mv=voltage,
            firing_rate_hz=self._rates(voltage),
            spike_events=(),
            synaptic_current=np.zeros(self.config.unit_count, dtype=np.float32),
            synaptic_current_units="mV",
            connectivity_fingerprint=self._connectivity_fingerprint,
            simulator_version=self.version,
            model_class=self.config.model_class,
            numerical_status="FINITE_INITIAL_STATE",
        )

    def step(
        self,
        state: NeuralStateSnapshot,
        inputs: Sequence[NeuralInputEvent],
        dt_ms: float | None = None,
    ) -> NeuralStateSnapshot:
        if not isinstance(state, NeuralStateSnapshot):
            raise TypeError("state must be a NeuralStateSnapshot")
        if state.connectivity_fingerprint != self._connectivity_fingerprint:
            raise ValueError("state connectivity fingerprint does not match this simulator")
        if state.model_class != self.config.model_class:
            raise ValueError("state model class is incompatible")
        if state.membrane_potential_mv is None or state.firing_rate_hz is None:
            raise ValueError("rate-based step requires membrane potential and firing rate")
        dt = self.config.dt_ms if dt_ms is None else float(dt_ms)
        if not np.isfinite(dt) or dt <= 0 or dt > self.config.membrane_tau_ms:
            raise ValueError("dt_ms must be finite, positive and no greater than membrane_tau_ms")
        voltage = np.asarray(state.membrane_potential_mv, dtype=np.float64)
        rate = np.asarray(state.firing_rate_hz, dtype=np.float64)
        if voltage.shape != (self.config.unit_count,) or rate.shape != voltage.shape:
            raise ValueError("state arrays do not match the configured unit count")

        external = np.zeros(self.config.unit_count, dtype=np.float64)
        next_time = state.simulation_time_ms + dt
        for event in inputs:
            if not isinstance(event, NeuralInputEvent):
                raise TypeError("inputs must contain NeuralInputEvent objects")
            if event.unit_index >= self.config.unit_count:
                raise ValueError("input event unit_index is outside the configured network")
            if not state.simulation_time_ms < event.event_time_ms <= next_time:
                raise ValueError("input event time must fall in the next simulation interval")
            if event.signal_kind != "current":
                raise ValueError("rate simulator supports current input events only")
            external[event.unit_index] += event.value

        recurrent = np.asarray(self.config.recurrent_weights, dtype=np.float64) @ rate
        synaptic = recurrent + external
        target = self.config.resting_potential_mv + synaptic
        decay = float(np.exp(-dt / self.config.membrane_tau_ms))
        next_voltage = decay * voltage + (1.0 - decay) * target
        next_rate = self._rates(next_voltage)
        if not np.isfinite(next_voltage).all() or not np.isfinite(next_rate).all():
            raise FloatingPointError("neural simulation produced non-finite state")
        return NeuralStateSnapshot(
            simulation_time_ms=next_time,
            step_index=state.step_index + 1,
            dt_ms=dt,
            population_unit_indices=self._population_indices,
            membrane_potential_mv=next_voltage.astype(np.float32),
            firing_rate_hz=next_rate,
            spike_events=(),
            synaptic_current=synaptic.astype(np.float32),
            synaptic_current_units="mV",
            connectivity_fingerprint=self._connectivity_fingerprint,
            simulator_version=self.version,
            model_class=self.config.model_class,
            numerical_status="FINITE_STEP",
            trace_ref=state.trace_ref,
        )

    def reset(self, state: NeuralStateSnapshot) -> NeuralStateSnapshot:
        if state.connectivity_fingerprint != self._connectivity_fingerprint:
            raise ValueError("cannot reset a state from a different network")
        return self.initialize()


__all__ = ["RateBasedNeuralSimulator", "NeuralSimulator", "NeuralSimulationConfig"]
