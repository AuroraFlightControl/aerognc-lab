from src.aerognc.control.command_vector import CommandVector, CmdLayout
from src.aerognc.core.state import VehicleState
import numpy as np

class DummyController:
    def __init__(self, layout: CmdLayout, trim_cmd: CommandVector):
        self.layout = layout
        self.trim_cmd = trim_cmd

        if "de" in self.layout.slices:
            self.elev_cmd = self.trim_cmd.scalar("de")
        if "da" in self.layout.slices:
            self.ail_cmd = self.trim_cmd.scalar("da")
        if "dr" in self.layout.slices:
            self.rud_cmd = self.trim_cmd.scalar("dr")
        if "THR" in self.layout.slices:
            self.thr_cmd = self.trim_cmd.scalar("THR")

    def reset(self, time_s: float) -> None:
        pass

    def update(self, time_s: float, state: VehicleState) -> CommandVector:

        new_cmds = self.trim_cmd.values.copy()

        if "de" in self.layout.slices:
           new_cmds[self.layout.get_slice("de")] = self.elev_cmd
        if "da" in self.layout.slices:
            new_cmds[self.layout.get_slice("da")] = self.ail_cmd
        if "dr" in self.layout.slices:
            new_cmds[self.layout.get_slice("dr")] = self.rud_cmd
        if "THR" in self.layout.slices:
            new_cmds[self.layout.get_slice("THR")] = self.thr_cmd

        return CommandVector(layout=self.layout, values=new_cmds)