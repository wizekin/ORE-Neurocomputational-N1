"""ORE N4 attention and precision-gating stage."""
from .config import N4Config
from .contracts import N4Input, N4Result
from .pipeline import N4_ALGORITHM_STEPS, N4Processor

__all__ = ["N4Config", "N4Input", "N4Result", "N4_ALGORITHM_STEPS", "N4Processor"]
__version__ = "0.1.0"
