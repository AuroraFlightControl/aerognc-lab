import sys
from pathlib import Path

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
import src.aerognc.core.StopConditions as Stop 

import numpy as np

def StickFixed_3DOF_Pioneer():

    print("Starting Pioneer Londgitudinal 3-DoF Simulation: Stick Fixed....\n")

    LONGITUDINAL_LAYOUT = StateLayout.from_fields(
    [
        ("position_ned_ft", 2),     # north, down
        ("velocity_body_fps", 2),   # u, w
        ("body_rate_rad_s", 1),     # q
        ("attitude_angle_rad", 1)   # theta
        ])

    initial_conditions = np.asarray([
        0.0,        # X Global Position [ft]
        -100.0,     # Z Global Position [ft]
        50.0,       # u Body Velocity [ft/s]
        0.0,        # w Body Velocity [ft/s]
        0.0,        # q Body Angular Rate [rad/s]
        0.0         # Theta Pitch Angle [rad]
    ])

    initial_state = VehicleState(layout=LONGITUDINAL_LAYOUT, values=initial_conditions)

    PIONEER_AERO_LAYOUT = AeroVariableLayout.from_variables([
        "a",    # Angle of Attack
        "q",    # Pitch Rate
        ])

    base_dir = Path(__file__).resolve().parent.parent
    pioneer_path = base_dir / "aircraft" / "Linear_Airlib" / "IAI_Pioneer.json"

    PioneerAeroModel = build_linear_aero_from_json(str(pioneer_path), layout=PIONEER_AERO_LAYOUT)

    PioneerMassModel = build_mass_properties_from_json(str(pioneer_path))


    sim_config = SimulationConfig(
        duration_s=10.0,
        integration_dt_s=0.01,
        logging_dt_s=0.1
        )

    longitudinal_dynamics = LongitudinalAircraftDynamics(
        mass_properties=PioneerMassModel,
        atmosphere_model=AtmosphereModel(),
        aerodynamic_model=PioneerAeroModel
        )

    sim_progress_bar = Stop.SimulationProgressBar(sim_config.total_integration_step)

    simExec = SimulationExec(
        config=sim_config,
        dynamics=longitudinal_dynamics,
        integrator=ForwardEulerIntegrator(),
        stop_conditions=(
            sim_progress_bar,
        )
    )

    result = simExec.run(initial_condition=initial_state)

    sim_progress_bar.close()
    print(f"\nSimulation Complete")
    print(f"Termination Reason: {result.termination_reason}")
    print(f"Total Data Points Logged: {len(result.time)}")


if __name__ == "__main__":

    StickFixed_3DOF_Pioneer()