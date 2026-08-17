from dataclasses import dataclass
from src.aerognc.math.Integrator import StateEquation, StateProjector
from src.aerognc.core.state import VehicleState, StateDerivative
from src.aerognc.control.command_vector import CommandVector


@dataclass
class ForwardEulerIntegrator:
    projector: StateProjector | None = None

    def step(self, equation: StateEquation, time_s: float, state: VehicleState, cmd: CommandVector, step_size_s: float) -> VehicleState:

        if step_size_s <= 0.0:
            raise ValueError("Integration step must be positive.")

        derivitive = equation(time_s, state, cmd)

        self._verify_layout(state, derivitive)

        next_values = state.values + step_size_s * derivitive.values

        next_state = state.with_values(next_values)

        if self.projector is not None:
            next_state = self.projector(next_state)

        return next_state

    @staticmethod
    def _verify_layout(state: VehicleState, derivitive: StateDerivative) -> None:
        if derivitive.layout != state.layout:
            raise ValueError("StateDerivitive layout does not match VehicleState Layout.")