import pytest
import numpy as np
import math

from src.aerognc.core.state import VehicleState, StateLayout
from src.aerognc.core.properties import MassProperties
from src.aerognc.math.ForwardEulerIntegrator import ForwardEulerIntegrator
from src.aerognc.dynamics.LongitudinalAircraftDynamics import LongitudinalAircraftDynamics

def test_longitudinal_dynamics_gravity_integration():
    # 1. Setup Mass Properties
    mass_props = MassProperties(
        mass_slug=100.0,
        inertia_body_slug_ft2=np.array([
            [1000.0, 0.0, 0.0],
            [0.0, 2000.0, 0.0],
            [0.0, 0.0, 3000.0]
        ]),
        cg_body_ft=np.array([0.0, 0.0, 0.0])
    )

    # 2. Setup the State Layout
    # Must perfectly match the order in the derivative_values array:
    # [x_dot, z_dot, u_dot, w_dot, q_dot, pitch_dot]
    layout = StateLayout.from_fields([
        ("position_ned_ft", 2),    # indices 0, 1 (x, z)
        ("velocity_body_fps", 2),  # indices 2, 3 (u, w)
        ("pitch_rate_rad_s", 1),   # index 4 (q)
        ("pitch_angle_rad", 1)     # index 5 (pitch)
    ])
    
    # Initial State: 
    # x=0, z=-5000 (5000 ft altitude)
    # u=100 fps (forward speed), w=0
    # q=0, pitch=0 (level flight)
    initial_values = np.asarray([0.0, -5000.0, 100.0, 0.0, 0.0, 0.0], dtype=float)
    state = VehicleState(layout=layout, values=initial_values)

    # 3. Initialize Dynamics Model
    dynamics = LongitudinalAircraftDynamics(mass_properties=mass_props)

    # 4. Setup Integrator
    integrator = ForwardEulerIntegrator()
    dt = 0.1  # 100ms timestep
    time_s = 0.0
    
    # 5. Perform the Step
    next_state = integrator.step(dynamics.derivatives, time_s, state, dt)

    # 6. Assertions on Physics Math
    # Math expectations for a flat, level aircraft experiencing ONLY gravity:
    # Fx = 0  --> u_dot = 0
    # Fz = weight --> w_dot = g (32.17)
    # q_dot = 0, pitch_dot = 0
    # x_dot = u = 100, z_dot = w = 0
    
    # New X Position = 0 + (100 * 0.1) = 10.0 ft
    assert np.isclose(next_state.section("position_ned_ft")[0], 10.0)
    
    # New Z Position = -5000 + (0 * 0.1) = -5000.0 ft
    assert np.isclose(next_state.section("position_ned_ft")[1], -5000.0)
    
    # New U Velocity = 100 + (0 * 0.1) = 100.0 fps
    assert np.isclose(next_state.section("velocity_body_fps")[0], 100.0)
    
    # New W Velocity = 0 + (32.17405 * 0.1) = 3.217405 fps (falling!)
    assert np.isclose(next_state.section("velocity_body_fps")[1], 3.217405)
    
    # Pitch and Pitch Rate remain completely undisturbed (no moments)
    assert np.isclose(next_state.scalar("pitch_rate_rad_s"), 0.0)
    assert np.isclose(next_state.scalar("pitch_angle_rad"), 0.0)