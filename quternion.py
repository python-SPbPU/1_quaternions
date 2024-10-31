from __future__ import annotations

import math


class Quaternion:
    def __init__(self, a: float, b: float, c: float, d: float):
        """
        Initialize Quaternion object with given coefficients a, b, c, d.

        Quaternion - number system extends the complex numbers.

        :param a: coefficient a
        :param b: coefficient b
        :param c: coefficient c
        :param d: coefficient d
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
        Norm is square root from sum of squares of (a, b, c, d).

        :return: norm of Quanternion - its "length"

        Example:
            >>> q = Quaternion(1, 2, 3, 4)
            >>> q.norm()
            5.477225575051661
        """
        return math.sqrt(self.a ** 2 + self.b ** 2 + self.c ** 2 + self.d ** 2)

    def inverse(self):
        """
        Reverse the Quaternion

        :return: inverse copy of Quaternion
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
        """
        return self * other.inverse()
