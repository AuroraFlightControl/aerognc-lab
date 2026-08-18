from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True, slots=True)
class ForcesMoments:
    forces_body_lbs: NDArray[np.float64]
    moments_body_ftlbs: NDArray[np.float64]