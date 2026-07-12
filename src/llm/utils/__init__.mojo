"""Foundation utilities: the seeded RNG and benchmark timing helpers."""

from .random import Rng, LCG_MULTIPLIER, LCG_INCREMENT
from .timing import median_ns, gflops_matmul
