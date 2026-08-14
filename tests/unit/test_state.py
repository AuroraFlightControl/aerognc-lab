import pytest
import numpy as np
from src.aerognc.core.state import (
    validate_vector,
    StateLayout,
    VehicleState,
    StateDerivative
)

# ==========================================
# 1. Tests for validate_vector
# ==========================================

def test_validate_vector_success():
    # It should accept standard python lists and convert to read-only numpy arrays
    raw_values = [1.0, 2.0, 3.0]
    result = validate_vector(raw_values, expected_size=3, vector_name="TestVec")
    
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))
    # FIX: Use .writeable instead of .write
    assert result.flags.writeable is False

def test_validate_vector_invalid_dimensions():
    # 2D arrays should be rejected
    with pytest.raises(ValueError, match="must be 1-D"):
        invalid_values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        validate_vector(invalid_values, expected_size=4, vector_name="TestVec")

def test_validate_vector_invalid_size():
    # Passing 4 values when 3 are expected
    with pytest.raises(ValueError, match="must contain 3 values"):
        validate_vector([1.0, 2.0, 3.0, 4.0], expected_size=3, vector_name="TestVec")

def test_validate_vector_non_finite_values():
    # NaNs and Infs should be caught immediately to prevent physics explosions
    with pytest.raises(ValueError, match="contains NaN or Infinite"):
        validate_vector([1.0, np.nan, 3.0], expected_size=3, vector_name="TestVec")
        
    with pytest.raises(ValueError, match="contains NaN or Infinite"):
        validate_vector([1.0, np.inf, 3.0], expected_size=3, vector_name="TestVec")


# ==========================================
# 2. Tests for StateLayout
# ==========================================

@pytest.fixture
def sample_layout() -> StateLayout:
    """A pytest fixture to provide a standard layout for multiple tests"""
    fields = [
        ("position", 3),
        ("velocity", 3),
        ("mass", 1)
    ]
    return StateLayout.from_fields(fields)

def test_state_layout_from_fields(sample_layout):
    # Total size should be 3 + 3 + 1 = 7
    assert sample_layout.size == 7
    
    # Check that slices are mapped correctly
    assert sample_layout.get_slice("position") == slice(0, 3)
    assert sample_layout.get_slice("velocity") == slice(3, 6)
    assert sample_layout.get_slice("mass") == slice(6, 7)

def test_state_layout_duplicate_fields():
    fields = [("position", 3), ("position", 3)]
    with pytest.raises(ValueError, match="Duplicate State Field: position"):
        StateLayout.from_fields(fields)

def test_state_layout_invalid_width():
    fields = [("position", 0)]
    with pytest.raises(ValueError, match="must have positive width"):
        StateLayout.from_fields(fields)

def test_state_layout_unknown_field(sample_layout):
    with pytest.raises(KeyError, match="Unknown state field: orientation"):
        sample_layout.get_slice("orientation")


# ==========================================
# 3. Tests for VehicleState & StateDerivative
# ==========================================

def test_vehicle_state_accessors(sample_layout):
    values = np.asarray([10.0, 20.0, 30.0, 1.5, 2.5, 3.5, 500.0], dtype=float)
    state = VehicleState(layout=sample_layout, values=values)
    
    # Test section() returns the correct numpy sub-array
    pos = state.section("position")
    assert np.array_equal(pos, np.array([10.0, 20.0, 30.0]))
    
    # Test scalar() returns a standard python float
    mass = state.scalar("mass")
    assert isinstance(mass, float)
    assert mass == 500.0

def test_vehicle_state_scalar_error(sample_layout):
    values = np.asarray([10.0, 20.0, 30.0, 1.5, 2.5, 3.5, 500.0], dtype=float)
    state = VehicleState(layout=sample_layout, values=values)
    
    # Trying to call scalar() on a 3-value section should fail
    with pytest.raises(ValueError, match="constains 3 values, not one"):
        state.scalar("position")

def test_vehicle_state_with_values(sample_layout):
    original_values = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0], dtype=float)
    state1 = VehicleState(layout=sample_layout, values=original_values)
    
    new_values = np.asarray([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 99.0], dtype=float)
    state2 = state1.with_values(new_values)
    
    # Prove that state2 is a completely new object
    assert state1 is not state2
    
    # Prove state2 has the new values and the same layout
    assert np.array_equal(state2.values, np.array(new_values))
    assert state2.layout is sample_layout
    
    # Prove state1 was not modified
    assert np.array_equal(state1.values, np.array(original_values))

def test_state_derivative_initialization(sample_layout):
    # Just verify StateDerivative runs the same validations via __post_init__
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
    derivative = StateDerivative(layout=sample_layout, values=values)
    
    assert derivative.layout is sample_layout
    assert derivative.values.flags.writeable is False