from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence, cast

import numpy as np  
from numpy.typing import NDArray

FloatVector = NDArray[np.float64]

def validate_vector(values: FloatVector | Sequence[float], expected_size: int, vector_name: str) -> FloatVector:
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError( f"{vector_name} must be 1-D, recived {array.shape}.")

    if array.size != expected_size:
        raise ValueError(f"{vector_name} must contain {expected_size} values, recived {array.size}")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{vector_name} contains NaN or Infinite values.")

    result = array.copy()
    result.setflags(write=False)

    return result

@dataclass(frozen=True, slots=True)
class CmdLayout:
    slices: Mapping[str, slice]
    size: int

    @classmethod
    def from_fields(cls, fields: Sequence[tuple[str, int]]) -> "CmdLayout":

        slices: dict[str, slice] = {}
        start = 0

        for name, width in fields:
            if name in slices:
                raise ValueError(f"Duplicate Cmd Field: {name}")

            if width <= 0:
                raise ValueError(f"Cmd Field '{name}' must have positive width.")

            slices[name] = slice(start, start + width)
            start += width

        return cls(slices=MappingProxyType(slices), size=start)

    def get_slice(self, name: str) -> slice:
        try:
            return self.slices[name]
        except KeyError as exc:
            raise KeyError(f"Unknown Cmd field: {name}") from exc

@dataclass(frozen=True, slots=True)
class CommandVector:
    layout: CmdLayout
    values: FloatVector

    def __post_init__(self) -> None:
        validated = validate_vector(self.values, expected_size=self.layout.size, vector_name="VehicleState")
        object.__setattr__(self, "values", validated)

    def section(self, name: str) -> FloatVector:
        # Cast to FloatVector to satisfy numpy typing for sliced arrays
        return cast(FloatVector, self.values[self.layout.get_slice(name)])

    def scalar(self, name: str) -> float:
        values = self.section(name)

        if values.size != 1:
            raise ValueError(f" State Field '{name}' constains {values.size} values, not one.")

        return float(values[0])

    def with_values(self, values: FloatVector | Sequence[float]) -> "CommandVector":
        return CommandVector(layout=self.layout, values = np.asarray(values, dtype=np.float64))


