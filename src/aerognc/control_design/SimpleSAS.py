from src.aerognc.control.command_vector import CommandVector, CmdLayout
from src.aerognc.core.state import VehicleState
from src.aerognc.core.Parameters import Parameters
import numpy as np
from dataclasses import dataclass
from typing import Optional
import math


class SimpleSAS:
    REQUIRED_PARAMS = {"pitch_Kp"}
    def __init__(self, layout: CmdLayout, params: Parameters):
        self.layout = layout

        Missing_Params = self.REQUIRED_PARAMS - params.params.keys()
        if Missing_Params:
            raise ValueError(f"Missing Prameters: {Missing_Params}")

        self.params = params.get_values()

        self.constant_throttle = 0.0

    def reset(self, time_s: float) -> None:
        pass

    def update(self, time_s: float, state: VehicleState) -> CommandVector:

        error = state.section("body_rate_rad_s")[0]

        elv_cmd = error * self.params["pitch_Kp"]

        elv_limit = 25 * math.pi/180 # +/- 25 degrees

        elv_cmd = [max(min(elv_cmd, elv_limit), -elv_limit)]


        cmd_values = np.zeros(self.layout.size, dtype=np.float64)
        

        if "de" in self.layout.slices:
            cmd_values[self.layout.get_slice("de")] = elv_cmd
            
        if "THR" in self.layout.slices:
            cmd_values[self.layout.get_slice("THR")] = self.constant_throttle

        return CommandVector(layout=self.layout, values=cmd_values)



