from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import json
from pathlib import Path



@dataclass(frozen=True, slots=True)
class MassProperties:
    mass_lbs: float
    inertia_body_slug_ft2: NDArray[np.float64]
    cg_body_ft: NDArray[np.float64]

    def __post_init__(self) -> None:
        inertia = np.asarray(self.inertia_body_slug_ft2, dtype=np.float64)
        cg = np.asarray(self.cg_body_ft, dtype=np.float64)

        if inertia.shape != (3,3):
            raise ValueError("Inertia Matrix must have shape (3, 3).")

        if cg.shape != (3,):
            raise ValueError("Center of Gravity must have shaper (3,).")

        if self.mass_lbs <= 0.0:
            raise ValueError("Mass must be positive.")

        object.__setattr__(self, "inertia_body_slug_ft2", inertia.copy())
        object.__setattr__(self, "cg_body_ft", cg.copy())




def build_mass_properties_from_json(json_filepath: str | Path) -> MassProperties:
    """Loads mass and inertia properties from an aircraft JSON configuration file."""
    
    with open(json_filepath, 'r') as f:
        data = json.load(f)

    # Extract the Geo_Mass dictionary[cite: 5]
    geo_mass = data.get("Geo_Mass", {})
    if not geo_mass:
        raise KeyError("JSON file is missing the 'Geo_Mass' block.")

    # Extract mass[cite: 5]
    mass_lbs = float(geo_mass["m"])

    # Extract inertia components[cite: 5]
    Ixx = float(geo_mass["Ixx"])
    Iyy = float(geo_mass["Iyy"])
    Izz = float(geo_mass["Izz"])
    Ixy = float(geo_mass["Ixy"])
    Ixz = float(geo_mass["Ixz"])
    Iyz = float(geo_mass["Iyz"])

    # Construct the 3x3 symmetric inertia tensor
    # Note: Standard flight dynamics convention negates the products of inertia in the off-diagonals.
    inertia_tensor = np.array([
        [ Ixx, -Ixy, -Ixz],
        [-Ixy,  Iyy, -Iyz],
        [-Ixz, -Iyz,  Izz]
    ], dtype=np.float64)

    # The provided JSON does not contain CG offsets, so we default to [0, 0, 0]
    cg_body = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Return the validated, frozen dataclass[cite: 6]
    return MassProperties(
        mass_lbs=mass_lbs,
        inertia_body_slug_ft2=inertia_tensor,
        cg_body_ft=cg_body
    )