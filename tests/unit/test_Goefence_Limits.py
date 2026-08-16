import pytest
from unittest.mock import Mock
import src.aerognc.core.StopConditions as Stop 
from src.aerognc.core.state import VehicleState, StateLayout
import numpy as np



@pytest.fixture
def make_dummy_state():
    # Factory Fixture to generate test vehicle states

    layout = StateLayout.from_fields([
        ("position_ned_ft", 3),
        ("velocity_fps", 3)
    ])

    def _make(north: float = 0.0, east: float = 0.0, down: float = 0.0) -> VehicleState:
        values = np.asarray([north, east, down, 0.0, 0.0, 0.0], dtype=np.float64)
        return VehicleState(layout=layout, values=values)

    return _make



def test_dummy_state(make_dummy_state):

    test_state = make_dummy_state(north=100.0, east=50, down=-10)


    ned_position = test_state.section("position_ned_ft")

    assert len(ned_position) == 3
    assert ned_position[0] == 100.0
    assert ned_position[1] == 50.0
    assert ned_position[2] == -10.0


def test_circle_keep_in_geofence(make_dummy_state):
    shape = Stop.Circle(center_x=0.0, center_y=0.0, radius=20.0)
    geofence = Stop.GeofenceLimit(shape=shape, fence_type=Stop.GeofenceType.KEEP_IN)

    safe_state = make_dummy_state(north=10.0, east=10.0)
    assert geofence(time_s=0.0, state=safe_state) is None

    breach_state = make_dummy_state(north=50.0, east=50.0)
    result = geofence(time_s=1.0, state=breach_state)

    assert result is not None
    assert "Breach" in result
    assert "Safe Zone" in result

def test_circle_keep_out_geofence(make_dummy_state):
    shape = Stop.Circle(center_x=0.0, center_y=0.0, radius=20.0)
    geofence = Stop.GeofenceLimit(shape=shape, fence_type=Stop.GeofenceType.KEEP_OUT)

    safe_state = make_dummy_state(north=25.0, east=25.0)
    assert geofence(time_s=0.0, state=safe_state) is None

    breach_state = make_dummy_state(north=10.0, east=10.0)
    result = geofence(time_s=1.0, state=breach_state)

    assert result is not None
    assert "Breach" in result
    assert "Restricted Zone" in result 

def test_rectangle_keep_in_geofence(make_dummy_state):
    shape = Stop.Rectangle(min_x=0.0, max_x=20.0, min_y=0.0, max_y=20.0)
    geofence = Stop.GeofenceLimit(shape=shape, fence_type=Stop.GeofenceType.KEEP_IN)

    safe_state = make_dummy_state(north=10.0, east=10.0)
    assert geofence(time_s=0.0, state=safe_state) is None

    breach_state = make_dummy_state(north=50.0, east=50.0)
    result = geofence(time_s=1.0, state=breach_state)

    assert result is not None
    assert "Breach" in result
    assert "Safe Zone" in result

def test_rectangle_keep_out_geofence(make_dummy_state):
    shape = Stop.Rectangle(min_x=0.0, max_x=20.0, min_y=0.0, max_y=20.0)
    geofence = Stop.GeofenceLimit(shape=shape, fence_type=Stop.GeofenceType.KEEP_OUT)

    safe_state = make_dummy_state(north=25.0, east=25.0)
    assert geofence(time_s=0.0, state=safe_state) is None

    breach_state = make_dummy_state(north=10.0, east=10.0)
    result = geofence(time_s=1.0, state=breach_state)

    assert result is not None
    assert "Breach" in result
    assert "Restricted Zone" in result 





