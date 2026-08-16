import pytest

from dataclasses import dataclass
from unittest.mock import Mock
import numpy as np

from src.aerognc.core.SimulationExec import SimulationConfig, SimulationExec
import src.aerognc.core.StopConditions as Stop 
from src.aerognc.core.state import StateLayout, VehicleState, StateDerivative
from src.aerognc.math.ForwardEulerIntegrator import ForwardEulerIntegrator


# Dummy Classes for test

@dataclass
class Simple_Dynamics:
    def derivatives(self, time_s: float, state: VehicleState) -> StateDerivative:
        x_dot = state.section("velocity_fps")[0]
        y_dot = state.section("velocity_fps")[1]
        return StateDerivative(layout=state.layout, values=np.asarray([x_dot, y_dot, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))

def test_simulation_geofence_abort():

    # Set Up Stop Conditions: Geofence Violation

    box = Stop.Rectangle(min_x=-10.0, max_x=10.0, min_y=-10.0, max_y=10.0)
    TestBox = Stop.GeofenceLimit(shape=box, fence_type=Stop.GeofenceType.KEEP_IN, name="Test Box")

    sim_config = SimulationConfig(
        duration_s = 20.0,
        integration_dt_s=0.1,
        logging_dt_s=0.1
        )

    simExec = SimulationExec(
        config=sim_config,
        dynamics=Simple_Dynamics(),
        integrator=ForwardEulerIntegrator(),
        stop_conditions=(TestBox,)
        )

    test_state_layout = StateLayout.from_fields([
        ("position_ned_ft", 3),
        ("velocity_fps", 3)
        ])

    test_initial_values = np.asarray([0.0, 0.0, -10.0, 1.0, 0.0, 0.0], dtype=np.float64)

    initial_conditions = VehicleState(layout=test_state_layout, values=test_initial_values)

    result = simExec.run(initial_condition=initial_conditions)

    assert "Breach" in result.termination_reason
    assert "Safe Zone" in result.termination_reason
    assert "10" in result.termination_reason # Geofence violation should be at time 10.[dt] ie 10.1 s


def test_logging_stride():
    
    sim_config = SimulationConfig(
        duration_s = 1.0,
        integration_dt_s=0.1,
        logging_dt_s=0.5
        )

    simExec = SimulationExec(
        config=sim_config,
        dynamics=Simple_Dynamics(),
        integrator=ForwardEulerIntegrator(),
        stop_conditions=()
        )

    test_state_layout = StateLayout.from_fields([
        ("position_ned_ft", 3),
        ("velocity_fps", 3)
        ])

    test_initial_values = np.asarray([0.0, 0.0, -10.0, 1.0, 0.0, 0.0], dtype=np.float64)

    initial_conditions = VehicleState(layout=test_state_layout, values=test_initial_values)

    result = simExec.run(initial_condition=initial_conditions)

    assert len(result.time) == 3
    assert len(result.state_values) == 3
    assert len(result.command_values) == 3

    assert result.time[0] == 0.0
    assert result.time[1] == 0.5
    assert result.time[2] == 1.0

def test_logging_final_step():

    sim_config = SimulationConfig(
        duration_s = 1.0,
        integration_dt_s=0.1,
        logging_dt_s=0.3
        )

    simExec = SimulationExec(
        config=sim_config,
        dynamics=Simple_Dynamics(),
        integrator=ForwardEulerIntegrator(),
        stop_conditions=()
        )

    test_state_layout = StateLayout.from_fields([
        ("position_ned_ft", 3),
        ("velocity_fps", 3)
        ])

    test_initial_values = np.asarray([0.0, 0.0, -10.0, 1.0, 0.0, 0.0], dtype=np.float64)

    initial_conditions = VehicleState(layout=test_state_layout, values=test_initial_values)

    result = simExec.run(initial_condition=initial_conditions)

    assert len(result.time) == 5
    assert result.time[4] == 1.0

def test_logging_early_termination():

    # Dummy Stop Condition to terminate at t = 0.6 sec
    def early_Kill(time_s: float, state: VehicleState) -> str | None:
        if time_s >= 0.6:
            return f"Early Termination at time 0.6 sec, Kill {time_s}"
        return None

    sim_config = SimulationConfig(
        duration_s = 1.0,
        integration_dt_s=0.1,
        logging_dt_s=0.5
        )

    simExec = SimulationExec(
        config=sim_config,
        dynamics=Simple_Dynamics(),
        integrator=ForwardEulerIntegrator(),
        stop_conditions=(early_Kill,)
        )

    test_state_layout = StateLayout.from_fields([
        ("position_ned_ft", 3),
        ("velocity_fps", 3)
        ])

    test_initial_values = np.asarray([0.0, 0.0, -10.0, 1.0, 0.0, 0.0], dtype=np.float64)

    initial_conditions = VehicleState(layout=test_state_layout, values=test_initial_values)

    result = simExec.run(initial_condition=initial_conditions)

    assert len(result.time) == 3
    assert len(result.state_values) == 3
    assert len(result.command_values) == 3

    assert np.isclose(result.time[2], 0.6, rtol=1e-4)

    assert "Early Termination" in result.termination_reason

