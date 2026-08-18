from src.aerognc.control.command_vector import CommandVector
from src.aerognc.control.ControllerStatus import ControllerStatus
from src.aerognc.core.state import VehicleState
from typing import Protocol

class Controller(Protocol):
    def reset(self, time_s: float) -> None:
        # Method to reset controller integrators or other functions
        ...

    def update(self, time_s: float, state: VehicleState) -> tuple[CommandVector, ControllerStatus]:
        # Method to update the controller and produce the command vector
        ...

