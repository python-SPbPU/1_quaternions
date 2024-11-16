from __future__ import annotations

import math
from doctest import testmod


class Quaternion:
    def __init__(self, a: float, b: float, c: float, d: float):
        """
        Initialize Quaternion object with given coefficients a, b, c, d.

        Quaternion - number system extends the complex numbers. Q = a + bi + cj + dk

        :param a: coefficient a
        :param b: coefficient b
        :param c: coefficient c
        :param d: coefficient d

        Example:
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q
        Quaternion(1, 2, 3, 4)
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __add__(self, other: Quaternion) -> Quaternion:
        """
        Implements addition operator '+'. Equivalent to <self + other>

        :param other: Quaternion - second summand
        :return: result of adding self to other

        Example:
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> q1 + q2
        Quaternion(6, 8, 10, 12)
        """
        return Quaternion(
            self.a + other.a,
            self.b + other.b,
            self.c + other.c,
            self.d + other.d
        )

    def __sub__(self, other: Quaternion) -> Quaternion:
        """
        Implements subtraction operator '-'. Equivalent to <self - other>

        :param other: Quaternion - subtrahend
        :return: result of subtraction other from self

        Example:
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> q1 - q2
        Quaternion(-4, -4, -4, -4)
        """
        return Quaternion(
            self.a - other.a,
            self.b - other.b,
            self.c - other.c,
            self.d - other.d
        )

    def __mul__(self, other: Quaternion) -> Quaternion:
        """
        Implements multiplication operator '*'. Equivalent to <self * other>

        :param other: Quaternion - second multiplier
        :return: result of multiplication self and other

        Example:
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> q1 * q2
        Quaternion(-60, 12, 30, 24)
        """
        return Quaternion(
            self.a * other.a - self.b * other.b - self.c * other.c - self.d * other.d,
            self.a * other.b + self.b * other.a + self.c * other.d - self.d * other.c,
            self.a * other.c - self.b * other.d + self.c * other.a + self.d * other.b,
            self.a * other.d + self.b * other.c - self.c * other.b + self.d * other.a
        )

    def norm(self) -> float:
        """
        Calculate the norm of Quanternion
        Norm is square root from sum of squares of (a, b, c, d):

        sqrt(a^2 + b^2 + c^2 + d^2)

        :return: norm of Quanternion - its "length"

        Example:
        >>> q = Quaternion(1, 1, 1, 1)
        >>> q.norm()
        2.0
        """
        return math.sqrt(self.a ** 2 + self.b ** 2 + self.c ** 2 + self.d ** 2)

    def inverse(self):
        """
        Reverse the Quaternion

        :return: inverse copy of Quaternion

        Example:
        >>> q = Quaternion(4, 4, 4, 4)
        >>> q.inverse()
        Quaternion(0.0625, -0.0625, -0.0625, -0.0625)
        """
        norm_sq = self.norm() ** 2
        return Quaternion(
            self.a / norm_sq,
            -self.b / norm_sq,
            -self.c / norm_sq,
            -self.d / norm_sq
        )

    def __truediv__(self, other: Quaternion) -> Quaternion:
        """
        Implements multiplication operator '/'. Equivalent to <self / other>
        Division is multiplication of self and inverse other
        :param other: Quaternion - divider
        :return: result of dividing self by other
        Example:
        >>> q1 = Quaternion(4, 4, 4, 4)
        >>> q2 = Quaternion(2, 2, 2, 2)
        >>> q1 / q2
        Quaternion(2.0, 0.0, 0.0, 0.0)
        """
        return self * other.inverse()

    def normalize(self) -> Quaternion:
        """
        Normalize the Quaternion
        :return: new normalized Quaternion

        Example:
        >>> q = Quaternion(1, 1, 1, 1)
        >>> q.normalize()
        Quaternion(0.5, 0.5, 0.5, 0.5)
        """
        n = self.norm()
        return Quaternion(self.a / n, self.b / n, self.c / n, self.d / n)

    def conjugate(self) -> Quaternion:
        """
        Calculate conjugate of the Quaternion
        :return: new conjugate Quaternion

        Example:
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.conjugate()
        Quaternion(1, -2, -3, -4)
        """
        return Quaternion(self.a, -self.b, -self.c, -self.d)

    def vector(self) -> tuple[float, float, float]:
        """
        Return vector part of the Quaternion
        :return: vector with coefficients b, c, d

        Example:
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.vector()
        (2, 3, 4)
        """
        return self.b, self.c, self.d

    def rotate_vector(self, vector: tuple[float, float, float]) -> tuple[float, float, float]:
        """
        Rotate vector by Quaternion

        q * v * q^-1
        :param vector: vector to rotate
        :return: rotated vector

        Example:
        >>> q = Quaternion(1, 2, 3, 4)
        >>> v = (1, 1, 1)
        >>> q.rotate_vector(v)
        (0.19999999999999996, 1.0, 1.4)
        """
        q_vector = Quaternion(0, *vector)
        return (self * q_vector * self.inverse()).vector()

    @staticmethod
    def from_axis_angle(axis: tuple[float, float, float], angle: float) -> Quaternion:
        """
        Create Quaternion from axis and angle
        :param axis: axis of rotation
        :param angle: angle of rotation
        :return: new Quaternion

        Example:
        >>> axis = (0, 1, 0)
        >>> angle = math.radians(90)  # Rotate 90 degrees
        >>> q = Quaternion.from_axis_angle(axis, angle)
        >>> q
        Quaternion(0.7071067811865476, 0.0, 0.7071067811865476, 0.0)
        """
        half_angle = angle / 2
        sin_half_angle = math.sin(half_angle)
        return Quaternion(
            math.cos(half_angle),
            axis[0] * sin_half_angle,
            axis[1] * sin_half_angle,
            axis[2] * sin_half_angle
        )

    def to_axis_angle(self) -> tuple[tuple[float, float, float], float]:
        """
        Convert Quaternion to axis and angle
        :return: axis of rotation and angle of rotation
        """
        angle = 2 * math.acos(self.a)
        sin_half_angle = math.sqrt(1 - self.a * self.a)
        if sin_half_angle < 1e-10:
            return (1, 0, 0), 0
        axis = (self.b / sin_half_angle, self.c / sin_half_angle, self.d / sin_half_angle)
        return axis, angle

    def __repr__(self):
        """
        Return string representation of Quaternion
        :return: string representation of Quaternion
        """
        return f"Quaternion({self.a}, {self.b}, {self.c}, {self.d})"


if __name__ == "__main__":
    testmod(name="quternion", verbose=True)
