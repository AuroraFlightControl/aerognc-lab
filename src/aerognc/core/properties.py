from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray



@dataclass(frozen=True, slots=True)
class MassProperties:
    mass_slug: float
    inertia_body_slug_ft2: NDArray[np.float64]
    cg_body_ft: NDArray[np.float64]

    def __post_init__(self) -> None:
        inertia = np.asarray(self.inertia_body_slug_ft2, dtype=np.float64)
        cg = np.asarray(self.cg_body_ft, dtype=np.float64)

        if inertia.shape != (3,3):
            raise ValueError("Inertia Matrix must have shape (3, 3).")

        if cg.shape != (3,):
            raise ValueError("Center of Gravity must have shaper (3,).")

        if self.mass_slug <= 0.0:
            raise ValueError("Mass must be positive.")

        object.__setattr__(self, "inertia_body_slug_ft2", inertia.copy())
        object.__setattr__(self, "cg_body_ft", cg.copy())