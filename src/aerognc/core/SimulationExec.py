from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from numpy.typing import NDArray

from src.aerognc.core.state import VehicleState, FloatVector, StateDerivative
from src.aerognc.control.command_vector import CommandVector
import src.aerognc.core.StopConditions as Stop 
from src.aerognc.math.Integrator import Integrator
from src.aerognc.dynamics.DynamicsModel import DynamicsModel
from src.aerognc.control_design.Controller import Controller

StopCondition = Callable[[float, VehicleState], str | None]

@dataclass
class SimulationConfig:
    duration_s: float
    integration_dt_s: float
    controller_dt_s: float
    logging_dt_s: float

    def __post_init__(self) -> None:

        if self.duration_s <= 0.0:
            raise ValueError("Simulation duration must be positve.")

        if self.integration_dt_s <= 0.0:
            raise ValueError("Integration time step must be positive.")

        if self.controller_dt_s <= 0.0:
            raise ValueError("Controller time step must be positive.")

        if self.logging_dt_s <= 0.0:
            raise ValueError("Logging rate must be positive.")

        self._validate_integer_multiple(
            larger=self.duration_s,
            smaller=self.integration_dt_s,
            names="duration_s / integration_dt_s"
        )

        self._validate_integer_multiple(
            larger=self.controller_dt_s,
            smaller=self.integration_dt_s,
            names="controller_dt_s / integration_dt_s"
        )

        self._validate_integer_multiple(
            larger=self.logging_dt_s,
            smaller=self.integration_dt_s,
            names="logging_dt_s / integration_dt_s"
        )

        if self.controller_dt_s < self.integration_dt_s:
            raise ValueError("Controller Rate Cannot be faster than Integratior Rate")

        if self.logging_dt_s < self.integration_dt_s:
            raise ValueError("Logging Rate Cannot be faster than Integrator Rate.")

    @staticmethod
    def _validate_integer_multiple(larger: float, smaller: float, names: str) -> None:
        ratio = larger / smaller
        nearest_integer = round(ratio)

        if not np.isclose(ratio, nearest_integer, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"{names} must be an integer ratio, recived {ratio}.")

    @property
    def total_integration_step(self) -> int:
        return round(self.duration_s / self.integration_dt_s)

    @property
    def controller_stride(self) -> int:
        return round(self.controller_dt_s / self.integration_dt_s)

    @property
    def logging_stride(self) -> int:
        return round(self.logging_dt_s / self.integration_dt_s)


@dataclass
class SimulationResult:
    time:               NDArray
    state_values:       NDArray
    command_values:     NDArray
    termination_reason: str




@dataclass
class SimulationExec:
    config:             SimulationConfig
    dynamics:           DynamicsModel
    integrator:         Integrator
    controller:         Controller
    stop_conditions:    tuple[StopCondition, ...] = field(default_factory=tuple)

    def run(self, initial_condition: VehicleState) -> SimulationResult:

        current_state = initial_condition
        current_command = np.asarray([0.0], dtype=np.float64) # TODO: Add controller interface

        time_history: list[float] = [0.0]
        state_history: list[FloatVector] = [current_state.values.copy()]
        command_history: list[FloatVector] = [current_command]

        termination_reason = "Completed Simulation Time."


        for step_index in range(self.config.total_integration_step):
            current_time_s = step_index * self.config.integration_dt_s

            # Update Controller
            if step_index == 1:
                self.controller.reset(time_s=current_time_s)
                current_command = self.controller.update(time_s=current_time_s, state=current_state)
            else:
                current_command = self.controller.update(time_s=current_time_s, state=current_state)

            # harness to ensure the dynamics are represented as a state Equation function
            def stateEquation(time_s: float, state: VehicleState, cmd: CommandVector) -> StateDerivative:
                return self.dynamics.derivatives(time_s=time_s, state=state, cmd=cmd)

            new_state = self.integrator.step(
                equation=stateEquation, 
                time_s=current_time_s, 
                state=current_state, 
                cmd=current_command,
                step_size_s=self.config.integration_dt_s
                )


            current_state = new_state             
            next_time_s = (step_index + 1) * self.config.integration_dt_s

            stop_reason = self.check_stop_conditions(time_s=next_time_s, state=current_state)

            should_log = (step_index + 1) % self.config.logging_stride == 0

            final_step = (step_index + 1) == self.config.total_integration_step

            if should_log | final_step | (stop_reason is not None):
                time_history.append(next_time_s)
                state_history.append(current_state.values.copy())
                command_history.append(current_command.values.copy()) # type: ignore


            if stop_reason is not None:
                termination_reason = stop_reason
                break


        return SimulationResult(
            time=np.asarray(time_history, dtype=np.float64), 
            state_values=np.stack(state_history, axis=0), 
            command_values=np.stack(command_history, axis=0), 
            termination_reason=termination_reason)

    def check_stop_conditions(self, time_s: float, state: VehicleState) -> str | None:

        for condition in self.stop_conditions:
            reason = condition(time_s, state)

            if reason is not None:
                return reason

        return None







    