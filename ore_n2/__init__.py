"""ORE N2 biological visual-processing package."""
from .config import N2Config
from .contracts import N2Input, N2Result, adapt_n1_result
from .pipeline import N2Processor
__all__=["N2Config","N2Input","N2Result","N2Processor","adapt_n1_result"]
__version__="0.1.0"
