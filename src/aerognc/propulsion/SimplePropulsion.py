from src.aerognc.core.ForcesMoments import ForcesMoments
from src.aerognc.core.state import VehicleState
from src.aerognc.control.command_vector import CommandVector
from src.aerognc.enviroment.AtmosphereModel import EnvData
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numpy.typing import NDArray

@dataclass
class SimplePropulsion:
    Max_Thrust: float = 0.0
    Axis: NDArray[np.float64] = field(default_factory=lambda: np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    THR_Direct: Optional[float] = None

    def __pot_init__(self):

        norm = np.linalg.norm(self.Axis)
        if norm > 0:
            # Bypass and enforce normilization of the thrust axis
            object.__setattr__(self, 'Axis', self.Axis / norm)

    def update(self, time_s: float, state:VehicleState, cmd: CommandVector, AtmoData: EnvData) -> ForcesMoments:

        THR = 0.0
        if self.THR_Direct is not None:
            THR = self.THR_Direct
        elif "THR" in cmd.layout.slices:
            THR = cmd.scalar("THR")
        else:
            return ForcesMoments(forces_body_lbs=np.zeros(3, dtype=np.float64), moments_body_ftlbs=np.zeros(3, dtype=np.float64))

        THR_Clipped = float(np.clip(THR, 0.0, 1.0))

        Thrust = self.Max_Thrust * THR_Clipped

        forces_body_lbs = Thrust * self.Axis

        return ForcesMoments(forces_body_lbs=forces_body_lbs, moments_body_ftlbs=np.zeros(3, dtype=np.float64))