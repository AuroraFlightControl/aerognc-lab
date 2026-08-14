from typing import Callable, Protocol
from dataclasses import dataclass
from src.aerognc.core.state import VehicleState, StateDerivative

StateEquation = Callable[[float, VehicleState], StateDerivative]

StateProjector = Callable[[VehicleState], VehicleState]

class Integrator(Protocol):
    def step(self, equation: StateEquation, time_s: float, state: VehicleState, step_size_s: float) -> VehicleState:
        ...