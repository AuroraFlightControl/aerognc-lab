import pytest
import numpy as np

from src.aerognc.core.state import VehicleState, StateDerivative, StateLayout
from src.aerognc.math.ForwardEulerIntegrator import ForwardEulerIntegrator

def test_1d_freefall_integration():
    # Layout
    Layout = StateLayout.from_fields([("altitude", 1), ("velocity", 1)])

    initial_values = np.asarray([1000.0, 0.0], dtype=float) # Starting at 1000 ft, 0 ft/s
    initial_state = VehicleState(layout=Layout, values=initial_values)

    # Define gravity Dynamics
    def gravity_dynamics(time: float, state: VehicleState) -> StateDerivative:
        g = 32.174  # ft/s^2
        altitude_derivative = state.values[1]  # velocity
        velocity_derivative = -g  # acceleration due to gravity
        return StateDerivative(layout=Layout, values=np.array([altitude_derivative, velocity_derivative]))

    integrator = ForwardEulerIntegrator()
    dt = 1.0  # 1 second time step
    time = 0.0

    next_state = integrator.step(gravity_dynamics, time, initial_state, dt)

    # Assert the Subsystems Communicated Correctly
    # Step 1 Expected: 
    # Alt = 1000 + (0 * 1) = 1000
    # Vel = 0 + (-32.174 * 1) = -32.174
    assert np.isclose(next_state.scalar("altitude"), 1000.0)
    assert np.isclose(next_state.scalar("velocity"), -32.174)
    # Ensure layout was preserved through the integration cycle
    assert next_state.layout is Layout

    # Perform a SECOND Integration Step (Proves the loop is stable)
    time += dt
    final_state = integrator.step(gravity_dynamics, time, next_state, dt)
    
    # Step 2 Expected: 
    # Alt = 1000 + (-32.174 * 1) = 967.826
    # Vel = -32.174 + (-32.174 * 1) = -64.348
    assert np.isclose(final_state.scalar("altitude"), 967.826)
    assert np.isclose(final_state.scalar("velocity"), -64.348)


