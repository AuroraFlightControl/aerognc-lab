from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Protocol
from src.aerognc.core.state import VehicleState
from src.aerognc.control.command_vector import CommandVector
from src.aerognc.core.ForcesMoments import ForcesMoments

@dataclass(frozen=True, slots=True)
class AeroForcesMoments:
    forces_body_lbs: NDArray[np.float64]
    moments_body_ftlbs: NDArray[np.float64]


class AerodynamicModel(Protocol):
    def evaluate(self, state: VehicleState, cmd: CommandVector) -> ForcesMoments:
        # TODO: Add enviroment and control inputs to the evaluate function
        ...