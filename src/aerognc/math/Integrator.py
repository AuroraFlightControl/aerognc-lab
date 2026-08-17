from typing import Callable, Protocol
from dataclasses import dataclass
from src.aerognc.core.state import VehicleState, StateDerivative
from src.aerognc.control.command_vector import CommandVector

StateEquation = Callable[[float, VehicleState, CommandVector], StateDerivative]

StateProjector = Callable[[VehicleState], VehicleState]

class Integrator(Protocol):
    def step(self, equation: StateEquation, time_s: float, state: VehicleState, cmd: CommandVector, step_size_s: float) -> VehicleState:
        ...