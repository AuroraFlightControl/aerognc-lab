from src.aerognc.core.ForcesMoments import ForcesMoments
from src.aerognc.core.state import VehicleState
from src.aerognc.control.command_vector import CommandVector
from src.aerognc.enviroment.AtmosphereModel import EnvData
from dataclasses import dataclass
import numpy as np

@dataclass
class ZeroThrust:

    def update(self, time_s: float, state:VehicleState, cmd: CommandVector, AtmoData: EnvData) -> ForcesMoments:
        return ForcesMoments(forces_body_lbs=np.zeros(3, dtype=np.float64), moments_body_ftlbs=np.zeros(3, dtype=np.float64))


