from dataclasses import dataclass
import math
from typing import Dict, Sequence, Optional
import numpy as np
from numpy.typing import NDArray
from src.aerognc.aerodynamics.AeroForcesMoments import AeroForcesMoments
from src.aerognc.core.state import VehicleState
from src.aerognc.aerodynamics.AirData import AirData, calculate_air_data
import json, math
import src.aerognc.Constants as CST
from src.aerognc.enviroment.AtmosphereModel import EnvData
from src.aerognc.control.command_vector import CommandVector

@dataclass(frozen=True)
class AeroVariableLayout:
    variable_map: Dict[str, int]
    size: int

    @classmethod
    def from_variables(cls, variables: Sequence[str]) -> "AeroVariableLayout":
        mapping = {name: idx for idx, name in enumerate(variables)}
        return cls(variable_map=mapping, size=len(variables))

@dataclass(frozen=True)
class LinearAeroModel:
    layout: AeroVariableLayout
    S_ref_ft2: float
    c_bar_ft: float
    b_span_ft: float
    C0: NDArray[np.float64] # Shape: (6,)
    Jacobian: NDArray[np.float64] # Shape: (6,N) where N is the number of variables in the layout

    def evaluate(self, state: VehicleState, cmd: CommandVector, AtmoData: Optional[EnvData] = None) -> AeroForcesMoments:

        if AtmoData is not None:
            density_slug_ft3 = AtmoData.density_slug_ft3
        else:
            density_slug_ft3 = CST.DENSITY_MSL

        airdata_packet = calculate_air_data(state=state, ambient_density=density_slug_ft3)


        # Extract the relevant variables from the state based on the layout
        variable_values = []
        for var in self.layout.variable_map.keys():
            if var == "a":
                variable_values.append(airdata_packet.alpha_rad)
            elif var == "b":
                variable_values.append(airdata_packet.beta_rad)
            elif var == "q":
                variable_values.append(state.section("body_rate_rad_s")[0])
            elif var == "p":
                variable_values.append(state.section("body_rate_rad_s")[1])
            elif var == "r":
                variable_values.append(state.section("body_rate_rad_s")[3])
            elif var == "de":
                variable_values.append(cmd.scalar("de"))
            elif var == "da":
                variable_values.append(cmd.scalar("da"))
            elif var == "dr":
                variable_values.append(cmd.scalar("dr"))
            elif var == "df":
                variable_values.append(cmd.scalar("df"))
            else:
                variable_values.append(state.scalar(var))
        

        variables = np.asarray(variable_values, dtype=np.float64)

        Q = airdata_packet.dynamic_pressure_psf
        alpha = airdata_packet.alpha_rad
        beta = airdata_packet.beta_rad

        # Compute the aerodynamic forces and moments using the linear model
        aero_effects = self.C0 + self.Jacobian @ variables
        
        CD, CY, CL, Cl, Cm, Cn = aero_effects

        
        # Dimensionalize the Forces (F = Q * S * C)
        Drag = Q * self.S_ref_ft2 * CD
        Side = Q * self.S_ref_ft2 * CY
        Lift = Q * self.S_ref_ft2 * CL

        # Pre-compute trig functions to save CPU cycles
        ca = math.cos(alpha)
        sa = math.sin(alpha)
        cb = math.cos(beta)
        sb = math.sin(beta)
        
        # Rotate Wind Frame to Body Frame
        Fx = -Drag * ca * cb - Side * ca * sb + Lift * sa
        Fy = -Drag * sb      + Side * cb
        Fz = -Drag * sa * cb - Side * sa * sb - Lift * ca
        
        # Dimensionalize the Moments (M = Q * S * Reference_Length * C)
        Roll_moment  = Q * self.S_ref_ft2 * self.b_span_ft * Cl
        Pitch_moment = Q * self.S_ref_ft2 * self.c_bar_ft * Cm
        Yaw_moment   = Q * self.S_ref_ft2 * self.b_span_ft * Cn
        
        forces_body_lbs = np.array([Fx, Fy, Fz], dtype=np.float64)
        moments_body_ftlbs = np.array([Roll_moment, Pitch_moment, Yaw_moment], dtype=np.float64)
        
        return AeroForcesMoments(forces_body_lbs=forces_body_lbs, moments_body_ftlbs=moments_body_ftlbs)

def build_linear_aero_from_json(json_filepath: str, layout: AeroVariableLayout) -> LinearAeroModel:

    with open(json_filepath, 'r') as f:
        data = json.load(f)

    geo = data["Geo_Mass"]
    S = geo["S"]
    cbar = geo["cbar"]
    b = geo["b"]

    C0 = np.zeros(6, dtype=np.float64)
    Jacobian = np.zeros((6, layout.size), dtype=np.float64)

    axis_map = {
        "Aero_D_Force": ("D", 0),
        "Aero_Y_Force": ("Y", 1),
        "Aero_L_Force": ("L", 2),
        "Aero_X_Moment": ("L", 3),
        "Aero_Y_Moment": ("M", 4),
        "Aero_Z_Moment": ("N", 5)
    }

    for aero_key, (var_prefix, row_idx) in axis_map.items():
        group_data = data.get(aero_key, {})

        for key, value in group_data.items():
            suffix = key[2:] # Strip C and axis from the key to get the variable name

            if suffix == "0":
                C0[row_idx] = value
            elif suffix in layout.variable_map:
                col_idx = layout.variable_map[suffix]
                Jacobian[row_idx, col_idx] = value

    found_variables = set()
    for group_name, (axis_letter, row_idx) in axis_map.items():
        for key in data.get(group_name, {}).keys():
            suffix = key[2:]
            if suffix in layout.variable_map:
                found_variables.add(suffix)

    # Check if we mapped a variable that the JSON completely ignored
    missing_vars = set(layout.variable_map.keys()) - found_variables
    if missing_vars:
        # You can use the standard logging module here, or raise a ValueError
        print(f"WARNING: Layout requested {missing_vars}, but they were missing from the JSON.")


    C0.setflags(write=False)
    Jacobian.setflags(write=False)

    return LinearAeroModel(
        layout=layout,
        S_ref_ft2=S,
        c_bar_ft=cbar,
        b_span_ft=b,
        C0=C0,
        Jacobian=Jacobian
    )
