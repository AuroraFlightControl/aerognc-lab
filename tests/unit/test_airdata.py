import pytest
import math
import numpy as np
from src.aerognc.core.state import VehicleState, StateLayout
from src.aerognc.aerodynamics.AirData import calculate_air_data


def build_mock_state(layout: StateLayout, **kwargs) -> VehicleState:
    raw_values = np.zeros(layout.size, dtype=np.float64)
    for field_name, value in kwargs.items():
        raw_values[layout.get_slice(field_name)] = value
    return VehicleState(layout=layout, values=raw_values)

def test_airdata_6dof_pure_forward():
    # 6DOF layout (size 3 velocity)
    layout = StateLayout.from_fields([("velocity_body_fps", 3)])
    
    # 100 fps straight ahead. No vertical, no side slip.
    state = build_mock_state(layout, velocity_body_fps=[100.0, 0.0, 0.0])
    
    rho = 0.0023769 # Standard Sea Level Density
    air_data = calculate_air_data(state, ambient_density=rho)
    
    # Assertions
    assert np.isclose(air_data.true_airspeed_fps, 100.0)
    assert np.isclose(air_data.alpha_rad, 0.0)
    assert np.isclose(air_data.beta_rad, 0.0)
    # Q = 0.5 * 0.0023769 * 100^2 = 11.8845
    assert np.isclose(air_data.dynamic_pressure_psf, 11.8845)

def test_airdata_6dof_sideslip():
    layout = StateLayout.from_fields([("velocity_body_fps", 3)])
    
    # Flying 100 fps forward, while sliding 100 fps to the right
    state = build_mock_state(layout, velocity_body_fps=[100.0, 100.0, 0.0])
    
    air_data = calculate_air_data(state, ambient_density=0.002)
    
    # TAS should be sqrt(100^2 + 100^2) = 141.42
    assert np.isclose(air_data.true_airspeed_fps, math.sqrt(20000))
    # Alpha should be 0 (no w velocity)
    assert np.isclose(air_data.alpha_rad, 0.0)
    # Beta should be exactly 45 degrees (pi/4 radians)
    assert np.isclose(air_data.beta_rad, math.pi / 4)

def test_airdata_3dof_longitudinal():
    # 3DOF layout (size 2 velocity)[cite: 1]
    layout = StateLayout.from_fields([("velocity_body_fps", 2)])
    
    # Flying 100 fps forward, 100 fps down (w)
    state = build_mock_state(layout, velocity_body_fps=[100.0, 100.0])
    
    air_data = calculate_air_data(state, ambient_density=0.002)
    
    # Alpha should be 45 degrees (pi/4 radians)
    assert np.isclose(air_data.alpha_rad, math.pi / 4)
    # Beta MUST be forced to 0.0 by the 3DOF size check logic[cite: 1]
    assert air_data.beta_rad == 0.0

def test_airdata_zero_airspeed():
    layout = StateLayout.from_fields([("velocity_body_fps", 3)])
    state = build_mock_state(layout, velocity_body_fps=[0.0, 0.0, 0.0])
    
    air_data = calculate_air_data(state, ambient_density=0.002)
    
    # Should safely handle divide-by-zero checks and return 0[cite: 1]
    assert air_data.true_airspeed_fps == 0.0
    assert air_data.alpha_rad == 0.0
    assert air_data.beta_rad == 0.0
    assert air_data.dynamic_pressure_psf == 0.0