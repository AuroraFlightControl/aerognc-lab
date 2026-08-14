from dataclasses import dataclass
import numpy as np
import math
from src.aerognc.core.state import VehicleState, StateDerivative
from src.aerognc.core.properties import MassProperties
import src.aerognc.Constants as CST


@dataclass
class LongitudinalAircraftDynamics:
    mass_properties: MassProperties
    #atmosphere_model: object
    #aerodynamic_model: object
    #propulsion_model: object
    #actuator_model: object

    def derivatives(
        self,
        time_s: float,
        state: VehicleState,
    ) -> StateDerivative:
        altitude_ft = -state.section("position_ned_ft")[1]

        x       = state.section("position_ned_ft")[0]
        z       = state.section("position_ned_ft")[1]
        u       = state.section("velocity_body_fps")[0]
        w       = state.section("velocity_body_fps")[1]
        q       = state.scalar("pitch_rate_rad_s")
        pitch   = state.scalar("pitch_angle_rad")

        mass = self.mass_properties.mass_slug
        Iyy = self.mass_properties.inertia_body_slug_ft2[1,1]



        #atmosphere = self.atmosphere_model.evaluate(altitude_ft)

        # Calculate wind and air data.

        # Calculate actuator outputs.

        # Calculate propulsion forces.

        # Calculate aerodynamic forces and moments.
        Fx = -mass * CST.ONE_G * math.sin(pitch)
        Fz = mass * CST.ONE_G * math.cos(pitch)
        M  = 0.0

        # Calculate rigid-body derivatives.

        u_dot = -q * w + Fx / mass
        w_dot = q * u + Fz / mass
        q_dot = M / Iyy
        x_dot = u * math.cos(pitch) + w * math.sin(pitch)
        z_dot = -u * math.sin(pitch) + w * math.cos(pitch)
        pitch_dot = q

        derivative_values = np.array(
            [x_dot, z_dot, u_dot, w_dot, q_dot, pitch_dot],
            dtype=np.float64,
        )

        return StateDerivative(
            layout=state.layout,
            values=derivative_values,
        )