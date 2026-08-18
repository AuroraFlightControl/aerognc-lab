from typing import Protocol
from src.aerognc.core.ForcesMoments import ForcesMoments
from src.aerognc.core.state import VehicleState
from src.aerognc.control.command_vector import CommandVector
from src.aerognc.enviroment.AtmosphereModel import EnvData



class PropulsionModel(Protocol):
    def update(self, time_s: float, state:VehicleState, cmd: CommandVector, AtmoData: EnvData) -> ForcesMoments:
        ...