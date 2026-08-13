from dataclasses import dataclass, field
import numpy as np

@dataclass
class SimulationConfig:
    duration_s: float
    integration_dt_s: float
    logging_dt_s: float

    def __post_init__(self) -> None:

        if self.duration_s <= 0.0:
            raise ValueError("Simulation duration must be positve.")

        if self.integration_dt_s <= 0.0:
            raise ValueError("Integration time step must be positive.")

        if self.logging_dt_s <= 0.0:
            raise ValueError("Logging rate must be positive.")

        self._validate_integer_multiple(
            larger=self.duration_s,
            smaller=self.integration_dt_s,
            names="duration_s / integration_dt_s"
        )

        self._validate_integer_multiple(
            larger=self.integration_dt_s,
            smaller=self.logging_dt_s,
            names="logging_dt_s / integration_dt_s"
        )

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
    def logging_stride(self) -> int:
        return round(self.logging_dt_s / self.integration_dt_s)



@dataclass
class SimulationExec:
    config: SimulationConfig






# @dataclass
# class SimulationExecutor:
#     dynamics: DynamicsModel
#     integrator: Integrator
#     controller: Controller
#     config: SimulationConfig
#     stop_conditions: tuple[StopCondition, ...] = field(
#         default_factory=tuple
#     )

#     def run(
#         self,
#         initial_state: VehicleState,
#     ) -> SimulationResult:
#         self.controller.reset(initial_state)

#         current_state = initial_state
#         current_command = self.controller.update(
#             time_s=0.0,
#             state=current_state,
#         )

#         time_history: list[float] = [0.0]
#         state_history: list[FloatVector] = [
#             current_state.values.copy()
#         ]
#         command_history: list[FloatVector] = [
#             current_command.values.copy()
#         ]

#         termination_reason = "Completed requested duration."

#         for step_index in range(
#             self.config.total_integration_steps
#         ):
#             current_time_s = (
#                 step_index
#                 * self.config.integration_dt_s
#             )

#             # Do not repeat the t=0 update.
#             if (
#                 step_index > 0
#                 and step_index
#                 % self.config.control_stride
#                 == 0
#             ):
#                 current_command = self.controller.update(
#                     time_s=current_time_s,
#                     state=current_state,
#                 )

#             held_command = current_command

#             def state_equation(
#                 stage_time_s: float,
#                 stage_state: VehicleState,
#             ) -> StateDerivative:
#                 return self.dynamics.derivatives(
#                     time_s=stage_time_s,
#                     state=stage_state,
#                     command=held_command,
#                 )

#             next_state = self.integrator.step(
#                 equation=state_equation,
#                 time_s=current_time_s,
#                 state=current_state,
#                 step_size_s=(
#                     self.config.integration_dt_s
#                 ),
#             )

#             next_time_s = (
#                 (step_index + 1)
#                 * self.config.integration_dt_s
#             )

#             current_state = next_state

#             stop_reason = self._check_stop_conditions(
#                 time_s=next_time_s,
#                 state=current_state,
#             )

#             should_log = (
#                 (step_index + 1)
#                 % self.config.logging_stride
#                 == 0
#             )

#             is_final_step = (
#                 step_index + 1
#                 == self.config.total_integration_steps
#             )

#             if should_log or is_final_step or stop_reason:
#                 time_history.append(next_time_s)
#                 state_history.append(
#                     current_state.values.copy()
#                 )
#                 command_history.append(
#                     current_command.values.copy()
#                 )

#             if stop_reason is not None:
#                 termination_reason = stop_reason
#                 break

#         return SimulationResult(
#             time_s=np.asarray(
#                 time_history,
#                 dtype=np.float64,
#             ),
#             state_values=np.stack(
#                 state_history,
#                 axis=0,
#             ),
#             command_values=np.stack(
#                 command_history,
#                 axis=0,
#             ),
#             termination_reason=termination_reason,
#             state_layout=initial_state.layout,
#         )

#     def _check_stop_conditions(
#         self,
#         time_s: float,
#         state: VehicleState,
#     ) -> str | None:
#         for condition in self.stop_conditions:
#             reason = condition(time_s, state)

#             if reason is not None:
#                 return reason

#         return None



if __name__=="__main__":
    Test_duration_s = 10.0
    Test_Sim_dt_s = 0.01
    Test_Logging_dt_s = 0.01

    Test_Sim_Config = SimulationConfig(Test_duration_s, Test_Sim_dt_s, Test_Logging_dt_s)

    