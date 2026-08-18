import numpy as np
from scipy.optimize import minimize # type: ignore[import-not-found]
import math
from src.aerognc.dynamics.DynamicsModel import DynamicsModel
from src.aerognc.control.command_vector import CommandVector, CmdLayout
from src.aerognc.core.state import VehicleState, StateLayout

def Longitudinal_trim_cost(
        free_vars: np.ndarray,
        target_V_fps: float,
        target_alt_ft: float,
        dynamics_model: DynamicsModel,
        state_layout: StateLayout,
        cmd_layout: CmdLayout
    ) -> float:

    # AoA = Pitch for Steady State flight
    theta_rad, elev_cmd, thr_cmd = free_vars

    u = target_V_fps * math.cos(theta_rad)
    w = target_V_fps * math.sin(theta_rad)

    state_values = np.asarray([
        0.0,            # Position Downrange [ft]
        -target_alt_ft, # Position Down [ft]
        u,              # Forward Body Velocity [ft/s]
        w,              # Vertical Body Velocity [ft/s]
        0.0,            # Pitch Rate [rad/s], 0.0 in steady State Flight
        theta_rad,      # Pitch Angle [rad]
    ], dtype=np.float64)

    state_vector = VehicleState(layout=state_layout, values=state_values)

    cmd_values = np.asarray([
        elev_cmd,
        thr_cmd,
    ])

    cmd_vector = CommandVector(layout=cmd_layout, values=cmd_values)

    state_dot = dynamics_model.derivatives(time_s=0.0, state=state_vector, cmd=cmd_vector)

    u_dot = state_dot.values[2]
    w_dot = state_dot.values[3]
    q_dot = state_dot.values[4]

    cost = (u_dot ** 2) + (w_dot ** 2) + (q_dot ** 2)*100.0

    return float(cost)

def find_trim_state(
        target_V_fps: float,
        target_alt_ft: float,
        dynamics_model: DynamicsModel,
        state_layout: StateLayout,
        cmd_layout: CmdLayout):

    initial_guess = np.array([0.0, 0.0, 0.0])

    bounds = (
        (-deg2rad(15), deg2rad(15)),
        (-deg2rad(15), deg2rad(15)),
        (0.0, 1.0)
    )

    print(f"Trimming for V_fps = {target_V_fps}, and alt_ft = {target_alt_ft}...")

    result = minimize(
        Longitudinal_trim_cost,
        initial_guess,
        args=(target_V_fps, target_alt_ft, dynamics_model, state_layout, cmd_layout),
        method='SLSQP',
        bounds=bounds,
        options={'ftol': 1e-9, 'disp': True}
        )

    if result.success:
        theta_trim, elv_trim, thr_trim = result.x 
        print(f"Trim Successful!")
        print(f"Alpha: {math.degrees(theta_trim):.2f} deg")
        print(f"Elevator: {elv_trim:.4f}")
        print(f"Throttle: {thr_trim:.4f}")

        return theta_trim, elv_trim, thr_trim
    
    else:
        raise RuntimeError("Trim routine failed to converge. Check aircraft limits or speeds.")

def deg2rad(value: float) -> float:
    return value * (math.pi / 180)