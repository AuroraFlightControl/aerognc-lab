import pytest
import json
import numpy as np
from typing import cast

from src.aerognc.aerodynamics.LinearModel import (
    AeroVariableLayout, 
    LinearAeroModel, 
    build_linear_aero_from_json
)
from src.aerognc.core.state import VehicleState, StateLayout

def build_mock_state(layout: StateLayout, **kwargs) -> VehicleState:
    """
    Safely builds a VehicleState by field name.
    Any fields defined in the layout but not provided will default to 0.0.
    
    Example:
        state = build_mock_state(layout, velocity_body_fps=[100.0, 0.0], pitch_rate_rad_s=0.5)
    """
    # 1. Start with a blank canvas of zeros exactly the size of the layout
    raw_values = np.zeros(layout.size, dtype=np.float64)
    
    # 2. Iterate through the provided keyword arguments
    for field_name, value in kwargs.items():
        # Get the exact slice where this data belongs
        field_slice = layout.get_slice(field_name)
        
        # Inject the data into the raw array. 
        # NumPy beautifully handles both scalars and lists/arrays here!
        raw_values[field_slice] = value
        
    # 3. Return the fully strictly validated VehicleState
    return VehicleState(layout=layout, values=raw_values)


# ==========================================
# 1. Test Fixtures (Mock Data)
# ==========================================

@pytest.fixture
def mock_aero_json_path(tmp_path):
    """Creates a temporary JSON file with a subset of aero data for testing."""
    mock_data = {
        "Geo_Mass": {"cbar": 0.5, "b": 5.0, "S": 2.8, "Ixx": 47.0, "Iyy": 90.0, "Izz": 111.0, "Ixy": 0, "Ixz": 0, "Iyz": 0, "m": 190.0},
        "Aero_D_Force": {"CD0": 0.060, "CDa": 0.430}, # Missing CDq and CDde to test default-to-zero
        "Aero_L_Force": {"CL0": 0.385, "CLa": 4.78, "CLq": 8.05, "CLde": 0.401},
        "Aero_Y_Moment": {"Cm0": 0.194, "Cma": -2.12, "Cmq": -36.6, "Cmde": -1.76}
    }
    
    # tmp_path is provided by pytest to create safe, temporary directories
    file_path = tmp_path / "mock_wing.json"
    with open(file_path, "w") as f:
        json.dump(mock_data, f)
        
    return str(file_path)


# ==========================================
# 2. Tests for the Layout
# ==========================================

def test_aero_variable_layout():
    # Define a layout with alpha, pitch rate, and elevator
    variables = ["a", "q", "de"]
    layout = AeroVariableLayout.from_variables(variables)
    
    assert layout.size == 3
    # Check that indices were mapped in the exact order provided
    assert layout.variable_map["a"] == 0
    assert layout.variable_map["q"] == 1
    assert layout.variable_map["de"] == 2


# ==========================================
# 3. Tests for the JSON Factory
# ==========================================

def test_build_aero_from_json(mock_aero_json_path):
    # We only care about alpha and elevator for this specific test
    layout = AeroVariableLayout.from_variables(["a", "de"])
    
    model = build_linear_aero_from_json(mock_aero_json_path, layout)
    
    # 1. Verify Geometry was parsed
    assert model.S_ref_ft2 == 2.8
    assert model.c_bar_ft == 0.5
    assert model.b_span_ft == 5.0
    
    # 2. Verify C0 (Base Coefficients)
    # Row 0 is Drag (D), Row 2 is Lift (L), Row 4 is Pitching Moment (m)
    assert model.C0[0] == 0.060  # CD0
    assert model.C0[2] == 0.385  # CL0
    assert model.C0[4] == 0.194  # Cm0
    
    # Verify a missing base coefficient defaulted to 0 (e.g., Side Force CY0)
    assert model.C0[1] == 0.0 
    
    # 3. Verify Jacobian Matrix (Stability Derivatives)
    # "a" is column 0, "de" is column 1
    assert model.Jacobian[0, 0] == 0.430  # CDa
    assert model.Jacobian[2, 0] == 4.78   # CLa
    assert model.Jacobian[4, 1] == -1.76  # Cmde
    
    # "q" was in the JSON, but NOT in our layout! It should be completely ignored.
    # Therefore, Cmq (-36.6) should not exist anywhere in the Jacobian.
    assert -36.6 not in model.Jacobian
    
    # 4. Verify Immutability
    assert model.C0.flags.writeable is False
    assert model.Jacobian.flags.writeable is False


# ==========================================
# 4. Tests for the Math Engine
# ==========================================

def test_linear_aero_math_execution():
    # 1. Construct the Aero layout
    aero_layout = AeroVariableLayout.from_variables(["a", "q"])
    
    C0 = np.asarray([0.1, 0.0, 0.5, 0.0, 0.2, 0.0], dtype=np.float64)
    Jacobian = np.zeros((6, 2), dtype=np.float64)
    Jacobian[0, 0] = 0.5   # CDa = 0.5
    Jacobian[2, 0] = 5.0   # CLa = 5.0
    Jacobian[4, 1] = -10.0 # Cmq = -10.0
    
    model = LinearAeroModel(
        layout=aero_layout, S_ref_ft2=1.0, c_bar_ft=1.0, b_span_ft=1.0, 
        C0=C0, Jacobian=Jacobian
    )
    
    # 2. Construct the VehicleState layout (NOW WITH VELOCITY!)
    vehicle_layout = StateLayout.from_fields([
        ("a", 1), 
        ("q", 1), 
        ("velocity_body_fps", 3)
    ])
    
    # Provide a forward velocity of 100 fps to avoid divide-by-zero checks
    state = build_mock_state(vehicle_layout, a=0.1, q=0.5, velocity_body_fps=[100.0, 0.0, 0.0])
    
    # 3. Evaluate
    result = model.evaluate(state)
    
    # ... (Keep your existing assertions here) ...

def test_linear_aero_size_mismatch():
    aero_layout = AeroVariableLayout.from_variables(["a", "q", "de"])
    model = LinearAeroModel(
        layout=aero_layout, S_ref_ft2=1.0, c_bar_ft=1.0, b_span_ft=1.0, 
        C0=np.zeros(6), Jacobian=np.zeros((6, 3))
    )
    
    # ALSO ADD VELOCITY HERE
    vehicle_layout = StateLayout.from_fields([
        ("a", 1), 
        ("q", 1), 
        ("velocity_body_fps", 3)
    ])
    
    bad_state = build_mock_state(vehicle_layout, a=0.1, q=0.5, velocity_body_fps=[100.0, 0.0, 0.0])
    
    with pytest.raises(KeyError, match="Unknown state field"):
        model.evaluate(bad_state)

