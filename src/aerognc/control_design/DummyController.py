from src.aerognc.control.command_vector import CommandVector, CmdLayout
from src.aerognc.core.state import VehicleState
import numpy as np

class DummyController:
    def __init__(self, layout: CmdLayout):
        self.layout = layout

    def reset(self, time_s: float) -> None:
        pass

    def update(self, time_s: float, state: VehicleState) -> CommandVector:
        return CommandVector(layout=self.layout, values=np.zeros(shape=self.layout.size, dtype=np.float64))