from typing import Protocol

class TestSignal(Protocol):
    def __call__(self, time_s: float) -> float:
        ...

class StepInput:
    def __init__(self, start_time_s: float, amplitude: float, base_value: float = 0.0) -> None:
        self.start_time_s = start_time_s
        self.amplitude = amplitude
        self.base_value = base_value

    def __call__(self, time_s: float) -> float:
        if time_s >= self.start_time_s:
            return self.base_value + self.amplitude
        
        return self.base_value


class DoubletInput:
    def __init__(self, start_time_s: float,  duration: float, amplitude: float, base_value: float = 0.0) -> None:
        self.start_time_s = start_time_s
        self.duration = duration
        self.amplitude = amplitude
        self.base_value = base_value

    def __call__(self, time_s: float) -> float:
        if self.start_time_s <= time_s < self.start_time_s + self.duration:
            return self.base_value + self.amplitude
        elif self.start_time_s + self.duration <= time_s < self.start_time_s + (2 *self.duration):
            return self.base_value - self.amplitude
        
        return self.base_value