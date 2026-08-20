from dataclasses import dataclass
from src.aerognc.core.state import VehicleState
from src.aerognc.control.command_vector import CommandVector
from src.aerognc.Test_Harness.TestSignal import TestSignal
from src.aerognc.control_design.Controller import Controller


@dataclass
class TestHarnessController:
    fcc: Controller

    target_variable: str
    signal_generator: TestSignal

    def reset(self, time_s: float) -> None:
        self.fcc.reset(time_s=time_s)

    def update(self, time_s: float, state: VehicleState) -> CommandVector:

        cmd_signal = self.signal_generator(time_s=time_s)

        setattr(self.fcc, self.target_variable, cmd_signal)

        return self.fcc.update(time_s=time_s, state=state)