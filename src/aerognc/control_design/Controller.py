from src.aerognc.control.command_vector import CommandVector
from src.aerognc.core.state import VehicleState
from typing import Protocol

class Controller(Protocol):
    def reset(self, time_s: float) -> None:
        # Method to reset controller integrators or other functions
        ...

    def update(self, time_s: float, state: VehicleState) -> CommandVector:
        # Method to update the controller and produce the command vector
        ...

