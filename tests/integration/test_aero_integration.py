import pytest
import numpy as np
import math
from src.aerognc.core.state import VehicleState, StateLayout
from src.aerognc.aerodynamics.LinearModel import AeroVariableLayout, LinearAeroModel
import src.aerognc.Constants as CST

def build_mock_state(layout: StateLayout, **kwargs) -> VehicleState:
    raw_values = np.zeros(layout.size, dtype=np.float64)
    for field_name, value in kwargs.items():
        raw_values[layout.get_slice(field_name)] = value
    return VehicleState(layout=layout, values=raw_values)

def test_aero_dimensionalization_and_rotation():
    # 1. Setup a simple Aero Model
    # We will only use Base Drag (CD0 = 0.1) and Base Lift (CL0 = 0.5) to test the math
    layout = AeroVariableLayout.from_variables([]) # No stability derivatives needed for this test
    
    C0 = np.array([0.1, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64)
    Jacobian = np.zeros((6, 0), dtype=np.float64) # Empty jacobian
    
    model = LinearAeroModel(
        layout=layout, 
        S_ref_ft2=10.0,  # 10 sq ft wing
        c_bar_ft=1.0, 
        b_span_ft=1.0, 
        C0=C0, 
        Jacobian=Jacobian
    )
    
    # 2. Setup a Vehicle State in pure forward flight
    state_layout = StateLayout.from_fields([("velocity_body_fps", 3)])
    
    # Fly forward at exactly 100 fps. Q = 0.5 * rho * 100^2
    state_forward = build_mock_state(state_layout, velocity_body_fps=[100.0, 0.0, 0.0])
    
    # 3. Evaluate Pure Forward Flight
    forces = model.evaluate(state_forward)
    
    # Math expectations for forward flight (Alpha = 0, Beta = 0):
    rho = CST.DENSITY_MSL
    expected_Q = 0.5 * rho * (100.0 ** 2) 

    # Drag = Q * S * CD0. Lift = Q * S * CL0
    expected_drag = expected_Q * 10.0 * 0.1
    expected_lift = expected_Q * 10.0 * 0.5

    # Since alpha = 0: Fx = -Drag, Fz = -Lift
    assert np.isclose(forces.forces_body_lbs[0], -expected_drag) # Fx
    assert np.isclose(forces.forces_body_lbs[1], 0.0)   # Fy
    assert np.isclose(forces.forces_body_lbs[2], -expected_lift) # Fz
    
    # 4. Evaluate with 90-degree Angle of Attack
    # Flying straight down (w = 100.0, u = 0.0). Alpha = 90 degrees.
    state_falling = build_mock_state(state_layout, velocity_body_fps=[0.0, 0.0, 100.0])
    forces_falling = model.evaluate(state_falling)
    
    assert np.isclose(forces_falling.forces_body_lbs[0], expected_lift)
    assert np.isclose(forces_falling.forces_body_lbs[2], -expected_drag)