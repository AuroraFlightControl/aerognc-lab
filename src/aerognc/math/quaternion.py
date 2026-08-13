import math
from dataclasses import dataclass

# Refactored Quaternion class to Quat class for consistency with the rest of the codebase.

from dataclasses import dataclass
import src.aerognc.Constants as CST

import math

@dataclass
class Quat:
    w: float
    x: float
    y: float
    z: float

    def __post_init__(self):
        # Ensure that the quaternion is normalized
        norm = (self.w**2 + self.x**2 + self.y**2 + self.z**2)

        if not math.isclose(norm, 1.0, rel_tol=1e-5):
            raise ValueError("Quaternion must be normalized.")

    def multiply(self, other: "Quat") -> "Quat":
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.z * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quat(w, x, y, z)

    def norm(self) -> float:
        return (self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def derivative(self, p: float, q: float, r: float) -> tuple[float, float, float, float]:
        correction_gain = 1 - self.norm()

        w_dot = -0.5 * (self.x * p + self.y * q + self.z * r) + self.w * correction_gain
        x_dot = 0.5 * (self.w * p + self.y * r - self.z * q) + self.x * correction_gain
        y_dot = 0.5 * (self.w * q + self.z * p - self.x * r) + self.y * correction_gain
        z_dot = 0.5 * (self.w * r + self.x * q - self.y * p) + self.z * correction_gain

        return (w_dot, x_dot, y_dot, z_dot)

    def conjugate(self) -> "Quat":
        return Quat(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> "Quat":
        return self.conjugate()  # For unit quaternions, the inverse is the conjugate

    def rotate_vector(self, vx: float, vy: float, vz: float) -> tuple[float, float, float]:
        # Extract the vector part of the quaternion
        qx, qy, qz = self.x, self.y, self.z
        qw = self.w

        # Cross product: q_vec x v
        cx = qy * vz - qz * vy
        cy = qz * vx - qx * vz
        cz = qx * vy - qy * vx

        # Cross product: q_vec x (q_vec x v)
        ccx = qy * cz - qz * cy
        ccy = qz * cx - qx * cz
        ccz = qx * cy - qy * cx

        # v_rotated = v + 2w(q_vec x v) + 2(q_vec x (q_vec x v))
        rx = vx + 2.0 * qw * cx + 2.0 * ccx
        ry = vy + 2.0 * qw * cy + 2.0 * ccy
        rz = vz + 2.0 * qw * cz + 2.0 * ccz

        return (rx, ry, rz)

    @staticmethod
    def from_axis_angle(n1: float, n2: float, n3: float, angle: float) -> "Quat":
        w = math.cos(angle/2)
        x = n1 * math.sin(angle/2)
        y = n2 * math.sin(angle/2)
        z = n3 * math.sin(angle/2)
        return Quat(w, x, y, z)

    @staticmethod
    def from_euler_angles(roll: float, pitch: float, yaw: float) -> "Quat":
        cy = math.cos(yaw / 2)
        sy = math.sin(yaw / 2)
        cp = math.cos(pitch / 2)
        sp = math.sin(pitch / 2)
        cr = math.cos(roll / 2)
        sr = math.sin(roll / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return Quat(w, x, y, z)



