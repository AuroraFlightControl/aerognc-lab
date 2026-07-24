import math
from dataclasses import dataclass
import numpy as np


@dataclass
class Quaternion:
    q0: float = 1
    q1: float = 0
    q2: float = 0
    q3: float = 0

    def getVector(self) -> np.ndarray:
        return np.array([self.q0, self.q1, self.q2, self.q3]).reshape((4,1))

    def getConjugate(self) -> "Quaternion":
        return Quaternion(self.q0, -self.q1, -self.q2, -self.q3)

    def multiply(self, Quat2: "Quaternion"):
        a0 = self.q0
        b0 = Quat2.q0
        a = np.array([self.q1, self.q2, self.q3]).reshape((3,1))
        b = np.array([Quat2.q1, Quat2.q2, Quat2.q3]).reshape((3,1))

        a0b0 = a0 * b0
        ab = np.vdot(a, b)

        c0 = a0b0 - ab

        a0b = a0 * b
        b0a = b0 * a
        axb = cross(a, b)

        c = a0b + b0a + axb

        return Quaternion(c0, c[0], c[1], c[2])

    @classmethod
    def fromEuler(cls, Roll_rad: float, Pitch_rad: float, Yaw_rad: float) -> "Quaternion":
        cosR = math.cos(Roll_rad/2)
        sinR = math.sin(Roll_rad/2)
        cosP = math.cos(Pitch_rad/2)
        sinP = math.sin(Pitch_rad/2)
        cosY = math.cos(Yaw_rad/2)
        sinY = math.sin(Yaw_rad/2)

        q0 = cosR*cosP*cosY + sinR*sinP*sinY
        q1 = sinR*cosP*cosY - cosR*sinP*sinY
        q2 = cosR*sinP*cosY + sinR*cosP*sinY
        q3 = cosR*cosP*sinY - sinR*sinP*cosY

        # Pass the calculated values directly into the constructor (cls)
        return cls(q0, q1, q2, q3)


def cross(a, b):
    i = (a[1] * b[2] - a[2] * b[1])
    j = -(a[0] * b[2] - a[2] * b[0])
    k = (a[0] * b[1] - a[1] * b[0])
    return np.array([i, j, k]).reshape((3, 1))





