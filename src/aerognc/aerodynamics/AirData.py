from dataclasses import dataclass
from src.aerognc.core.state import VehicleState
import math
import src.aerognc.Constants as CST

@dataclass(frozen=True, slots=True)
class AirData:
    dynamic_pressure_psf: float
    alpha_rad: float
    beta_rad: float
    true_airspeed_fps: float
    mach: float
    ambient_density_slug_ft3: float
    

def calculate_air_data(state: VehicleState, ambient_density: float) -> AirData:
    # Ambiend Density is in slug_ft3

    vel_body = state.section("velocity_body_fps")
    
    if vel_body.size == 3:
        # 6DOF Model: [u, v, w]
        u, v, w = vel_body[0], vel_body[1], vel_body[2]
    elif vel_body.size == 2:
        # Longitudinal 3DOF Model: [u, w]
        u, w = vel_body[0], vel_body[1]
        v = 0.0  # Explicitly force sideslip velocity to zero
    elif vel_body.size == 1:
        # 1DOF Model (e.g., rocket straight up, or car on a track)
        u = vel_body[0]
        v = 0.0
        w = 0.0
    else:
        raise ValueError(f"velocity_body_fps must have 1, 2, or 3 elements, got {vel_body.size}")

    true_airspeed_fps = (u**2 + v**2 + w**2)**0.5

    # Angle of Attack (alpha) and Sideslip Angle (beta) calculation
    if true_airspeed_fps > 0:
        alpha_rad = math.atan2(w, u)
        beta_rad = math.asin(v / true_airspeed_fps)
    else:
        alpha_rad = 0.0
        beta_rad = 0.0

    # Dynamic Pressure (Q) calculation
    dynamic_pressure_psf = 0.5 * ambient_density * (true_airspeed_fps ** 2)

    # Speed of Sound
    # TODO: Currently Assumes Speed of Sound at Sea Level
    speed_of_spound_fps = CST.SPEED_OF_SOUND_MSL
    mach = true_airspeed_fps / speed_of_spound_fps

    return AirData(
        dynamic_pressure_psf=dynamic_pressure_psf,
        alpha_rad=alpha_rad,
        beta_rad=beta_rad,
        true_airspeed_fps=true_airspeed_fps,
        mach=mach,
        ambient_density_slug_ft3=ambient_density
    )
 
