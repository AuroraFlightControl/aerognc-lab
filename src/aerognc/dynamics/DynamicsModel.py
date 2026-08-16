from typing import Protocol
from src.aerognc.core.state import VehicleState, StateDerivative


class DynamicsModel(Protocol):
    def derivatives(self, time_s: float, state: VehicleState) -> StateDerivative:
        ...

