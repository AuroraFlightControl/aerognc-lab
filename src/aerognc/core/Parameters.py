from dataclasses import dataclass
from typing import Optional
import math, json

@dataclass(frozen=True)
class Param:
    value: float
    min_value: float
    max_value: float
    increment: float
    description: Optional[str] = None

    def __post_init__(self) -> None:

        if self.value < self.min_value:
            raise ValueError(f"Parameter Value Invalid. Must be Greater then {self.min_value}")

        if self.value > self.max_value:
            raise ValueError(f"Parameter Value Invalid. Must be less then {self.max_value}")

        steps = (self.value - self.min_value) / self.increment
        if not math.isclose(steps, round(steps), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Parameter Value Invalid. Must be of increment {self.increment}")

@dataclass(frozen=True)
class Parameters:
    params: dict[str, Param]

    def get_values(self):
        return {name: p.value for name, p in self.params.items()}

    @staticmethod
    def from_dic(param_file: dict) -> "Parameters":
        loaded_param = {}
        for name, data in param_file.items():
            loaded_param[name] = Param(**data)
        return Parameters(loaded_param)

    @staticmethod
    def from_json(file_path: str) -> "Parameters":
        with open(file_path, 'r') as f:
            file_data = json.load(f)

        loaded_param = {}
        for name, data in file_data.items():
            loaded_param[name] = Param(**data)
        return Parameters(loaded_param)
