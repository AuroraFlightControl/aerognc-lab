from src.aerognc.control.command_vector import CommandVector, CmdLayout
from src.aerognc.core.state import VehicleState
import numpy as np

class DummyController:
    def __init__(self, layout: CmdLayout, trim_cmd: CommandVector):
        self.layout = layout
        self.trim_cmd = trim_cmd

    def reset(self, time_s: float) -> None:
        pass

    def update(self, time_s: float, state: VehicleState) -> CommandVector:
        return self.trim_cmd