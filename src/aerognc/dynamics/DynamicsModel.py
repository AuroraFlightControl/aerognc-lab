from typing import Protocol
from src.aerognc.core.state import VehicleState, StateDerivative
from src.aerognc.control.command_vector import CommandVector


class DynamicsModel(Protocol):
    def derivatives(self, time_s: float, state: VehicleState, cmd: CommandVector) -> StateDerivative:
        ...

