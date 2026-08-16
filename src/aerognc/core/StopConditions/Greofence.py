from src.aerognc.core.state import VehicleState

from enum import Enum, auto
from dataclasses import dataclass
from typing import Protocol

class GeofenceType(Enum):
    KEEP_IN = auto()
    KEEP_OUT = auto()

class Shape(Protocol):
    def contains(self, x: float, y: float) -> bool:
        ...

@dataclass
class Circle:
    center_x: float
    center_y: float
    radius: float

    def contains(self, x: float, y: float) -> bool:
        return (x - self.center_x) ** 2 + (y - self.center_y) ** 2 <= self.radius ** 2

@dataclass
class Rectangle:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y




@dataclass
class GeofenceLimit:
    shape: Shape
    fence_type: GeofenceType
    name: str = "Geofence"

    def __call__(self, time_s: float, state: VehicleState) -> str | None:
        # Assumes state has "position_ned_ft" and is in 3D

        vehicle_x = state.section("position_ned_ft")[0]
        vehicle_y = state.section("position_ned_ft")[1]


        is_inside = self.shape.contains(x=vehicle_x, y=vehicle_y)

        if self.fence_type == GeofenceType.KEEP_IN and not is_inside:
            return f"{self.name} Breach: Vehicle Left the Safe Zone at time: {time_s}."
        elif self.fence_type == GeofenceType.KEEP_OUT and is_inside:
            return f"{self.name} Breach: Vehicle Entered Restricted Zone at time: {time_s}."

        return None