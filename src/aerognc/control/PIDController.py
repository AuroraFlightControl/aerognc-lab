

class PIDControl:
    def __init__(self, Kp: float, Ki: float, Kd: float, Imax: float) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Imax = Imax

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = 0.0

    def reset(self, time_s: float) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time_s

    def update(self, time_s: float, target: float, actual: float) -> tuple[float, dict]:

        dt = time_s - self.prev_time
        if dt <= 0.0:

            status = {
                "Target": float(target),
                "Actual": float(actual),
                "Cmd": float(0.0)
                }


            return 0.0, status

        error = actual - target

        self.integral += error * dt
        self.integral = min(max(self.integral, -self.Imax), self.Imax)

        derivitive = (error - self.prev_error) / dt

        P = self.Kp * error
        I = self.Ki * self.integral
        D = self.Kd * derivitive

        cmd = P + I + D

        self.prev_error = error
        self.prev_time = time_s

        status = {
            "Target": float(target),
            "Actual": float(actual),
            "P": float(P),
            "I": float(I),
            "D": float(D),
            "error": float(error),
            "Integrator": float(self.integral),
            "Derivitive": float(derivitive),
            "Cmd": float(cmd),
        }

        return cmd, status