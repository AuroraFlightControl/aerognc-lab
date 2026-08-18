from src.aerognc.core.state import VehicleState
from src.aerognc.control.command_vector import CommandVector, CmdLayout
from src.aerognc.control.PIDController import PIDControl
from src.aerognc.core.Parameters import Parameters
from src.aerognc.control.CommandGenerator import Cmd_Step
from src.aerognc.control.ControllerStatus import ControllerStatus
import numpy as np
import math

class PIDPitchControl:
    def __init__(self, layout: CmdLayout, parameters: Parameters) -> None:
        self.layout = layout
        params = parameters.get_values()

        P_Kp = params["P_Kp"]
        P_Ki = params["P_Ki"]
        P_Kd = params["P_Kd"]
        P_Imax = params["P_IMAX"]

        self.P_PID = PIDControl(
            Kp=P_Kp,
            Ki=P_Ki,
            Kd=P_Kd,
            Imax=P_Imax
            )
        
        self.prev_time_s = 0.0
        self.trim_throttle = 0.0
        self.trim_theta_rad = 0.0

    def reset(self, time_s: float) -> None:
        self.prev_time_s = time_s
        self.P_PID.reset(time_s=time_s)

    def update(self, time_s: float, state: VehicleState) -> tuple[CommandVector, ControllerStatus]:


        if time_s < 10.0:
            theta_rad_cmd = self.trim_theta_rad
        elif 30.0 > time_s >= 10.0:
            theta_rad_cmd = self.trim_theta_rad + (3.0 * math.pi / 180)
        else:
            theta_rad_cmd = self.trim_theta_rad


        elv_cmd, P_PID_status = self.P_PID.update(time_s=time_s, target=theta_rad_cmd, actual=state.section("attitude_angle_rad")[0])

        cmd_values = np.zeros(self.layout.size, dtype=np.float64)
        if "de" in self.layout.slices:
            cmd_values[self.layout.get_slice("de")] = elv_cmd
            
        if "THR" in self.layout.slices:
            cmd_values[self.layout.get_slice("THR")] = self.trim_throttle

        merged_telemetry = {}

        for key, value in P_PID_status.items():
            merged_telemetry[f"P_PID_{key}"] = value

        
        status = ControllerStatus(
            timestamp_s=time_s,
            name= "Pitch PID",
            is_healthy=True,
            active_mode="Pitch",
            telemetry=merged_telemetry
        )

        return CommandVector(layout=self.layout, values=cmd_values), status
    