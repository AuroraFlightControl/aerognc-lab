from dataclasses import dataclass


@dataclass
class Cmd_Step:
    initial_Value:  float
    amplitude:      float
    start_time_s:   float
    end_time_s:     float

    def update(self, time_s: float) -> float:

        if self.end_time_s >= time_s >= self.start_time_s:
            return self.initial_Value + self.amplitude
        else:
            return self.initial_Value