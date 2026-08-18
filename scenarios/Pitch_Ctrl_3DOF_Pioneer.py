import sys, math
from pathlib import Path
import numpy as np  # type: ignore[import-not-found]

# Get the path to 'aerognc-lab' (up two levels from this script) and add it to the system path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.aerognc.core.SimulationExec import SimulationExec, SimulationConfig
from src.aerognc.core.state import VehicleState, StateLayout
from src.aerognc.dynamics.LongitudinalAircraftDynamics import LongitudinalAircraftDynamics
from src.aerognc.aerodynamics.LinearModel import build_linear_aero_from_json, AeroVariableLayout
from src.aerognc.math.ForwardEulerIntegrator import ForwardEulerIntegrator
from src.aerognc.enviroment.AtmosphereModel import AtmosphereModel
from src.aerognc.core.properties import build_mass_properties_from_json
from src.aerognc.control.command_vector import CmdLayout, CommandVector
from src.aerognc.control_design.PIDPitchControl import PIDPitchControl
from src.aerognc.core.Parameters import Parameters
from src.aerognc.propulsion.SimplePropulsion import SimplePropulsion
import src.aerognc.core.StopConditions as Stop 
from src.aerognc.visualization.basic_3DOF_Viz import visualize
from tools.AutoTrim3DOF import find_trim_state




def SimpleSAS_3DOF_Pioneer_Autotrim(target_V_fps: float, target_alt_ft: float):

    print("Starting Pioneer Londgitudinal 3-DoF Simulation: Pitch Control....\n")

    LONGITUDINAL_LAYOUT = StateLayout.from_fields(
    [
        ("position_ned_ft", 2),     # north, down
        ("velocity_body_fps", 2),   # u, w
        ("body_rate_rad_s", 1),     # q
        ("attitude_angle_rad", 1)   # theta
        ])


    PIONEER_CONTROL_LAYOUT = CmdLayout.from_fields([
        ("de", 1),
        ("THR", 1)
        
    ])

    PIONEER_AERO_LAYOUT = AeroVariableLayout.from_variables([
        "a",    # Angle of Attack
        "q",    # Pitch Rate
        "de",   # Elevator Command
        ])

    CONTROLLER_PARAMS = {
        "P_Kp": {
            "value": 5.0,
            "min_value": 0.0,
            "max_value": 10.0,
            "increment": 0.01,
            "description": 'PitchProportional Gain'
        },
        "P_Ki": {
            "value": 1.0,
            "min_value": 0.0,
            "max_value": 10.0,
            "increment": 0.01,
            "description": 'Pitch Integral Gain'
        },
       "P_Kd": {
            "value": 1.0,
            "min_value": 0.0,
            "max_value": 10.0,
            "increment": 0.01,
            "description": 'Pitch Derivitive Gain'
        },
       "P_IMAX": {
            "value": 1.0,
            "min_value": 0.0,
            "max_value": 10.0,
            "increment": 0.01,
            "description": 'Pitch Integrator Max'
        },

    }

    ctrl_params = Parameters.from_dic(CONTROLLER_PARAMS)

    base_dir = Path(__file__).resolve().parent.parent
    pioneer_path = base_dir / "aircraft" / "Linear_Airlib" / "IAI_Pioneer.json"

    PioneerAeroModel = build_linear_aero_from_json(str(pioneer_path), layout=PIONEER_AERO_LAYOUT)

    PioneerMassModel = build_mass_properties_from_json(str(pioneer_path))

    sim_config = SimulationConfig(
        duration_s=50.0,
        integration_dt_s=0.001,
        controller_dt_s=0.005,
        logging_dt_s=0.001
        )

    longitudinal_dynamics = LongitudinalAircraftDynamics(
        mass_properties=PioneerMassModel,
        atmosphere_model=AtmosphereModel(),
        aerodynamic_model=PioneerAeroModel,
        propulsion_model=SimplePropulsion(Max_Thrust=100.0),
        )

    theta_rad_trim, elv_cmd_trim, thr_cmd_trim = find_trim_state(
                                                    target_V_fps=target_V_fps, 
                                                    target_alt_ft=target_alt_ft, 
                                                    dynamics_model=longitudinal_dynamics, 
                                                    state_layout=LONGITUDINAL_LAYOUT,
                                                    cmd_layout=PIONEER_CONTROL_LAYOUT
                                                    ) 
    u_fps_trim = target_V_fps * math.cos(theta_rad_trim)
    w_fps_trim = target_V_fps * math.sin(theta_rad_trim)

    initial_conditions = np.asarray([
        0.0,                # X Global Position [ft]
        -target_alt_ft,     # Z Global Position [ft]
        u_fps_trim,         # u Body Velocity [ft/s]
        w_fps_trim,         # w Body Velocity [ft/s]
        0.0,                # q Body Angular Rate [rad/s]
        theta_rad_trim      # Theta Pitch Angle [rad]
    ])

    trim_cmd = np.asarray([
        elv_cmd_trim,
        thr_cmd_trim,
    ])

    controller = PIDPitchControl(layout=PIONEER_CONTROL_LAYOUT, parameters=ctrl_params)
    controller.trim_throttle = thr_cmd_trim
    controller.trim_theta_rad = theta_rad_trim

    sim_progress_bar = Stop.SimulationProgressBar(sim_config.total_integration_step)

    simExec = SimulationExec(
        config=sim_config,
        dynamics=longitudinal_dynamics,
        integrator=ForwardEulerIntegrator(),
        controller=controller,
        stop_conditions=(
            sim_progress_bar,
            Stop.check_ground_impact,
            )
        )  

    initial_state = VehicleState(layout=LONGITUDINAL_LAYOUT, values=initial_conditions)
    initial_cmd = CommandVector(layout=PIONEER_CONTROL_LAYOUT, values=trim_cmd)

    result = simExec.run(initial_condition=initial_state, initial_cmd=initial_cmd)

    sim_progress_bar.close()
    print(f"\nSimulation Complete")
    print(f"Termination Reason: {result.termination_reason}")
    print(f"Total Data Points Logged: {len(result.time)}")


    visualize(
        result=result, 
        state_layout=LONGITUDINAL_LAYOUT, 
        cmd_layout=PIONEER_CONTROL_LAYOUT
        )


if __name__ == "__main__":

    SimpleSAS_3DOF_Pioneer_Autotrim(
        target_V_fps=120.0,
        target_alt_ft=1000.0
    )