"""Optional first-order temporal neural dynamics."""
from __future__ import annotations

import numpy as np


class TemporalDynamics:
    """Leaky neural-state integrator with exponential time-constant semantics.

    x[t+dt] = x[t] + (1 - exp(-dt/tau)) * (u[t] - x[t])

    The first observation initializes the state (no artificial zero-frame
    transient). Call reset() at sequence boundaries.
    """

    def __init__(self, tau: float) -> None:
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = float(tau)
        self._state: np.ndarray | None = None

    @property
    def state(self) -> np.ndarray | None:
        return None if self._state is None else self._state.copy()

    def reset(self) -> None:
        self._state = None

    def update(self, activity: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 0:
            raise ValueError("dt must be positive")
        current = np.asarray(activity, dtype=np.float32)
        if not np.isfinite(current).all():
            raise FloatingPointError("temporal input contains NaN or infinite values")
        if self._state is None or self._state.shape != current.shape:
            self._state = current.copy()
        else:
            alpha = 1.0 - np.exp(-float(dt) / self.tau)
            self._state += np.float32(alpha) * (current - self._state)
        return self._state.copy()
