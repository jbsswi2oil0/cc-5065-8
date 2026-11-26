"""Minimal 3D vector / colour math on top of the Python standard library."""
import math


def _num(x):
    if isinstance(x, Vec3):
        return x
    return Vec3(x, x, x)


class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # arithmetic ---------------------------------------------------------
    def __add__(self, o):
        o = _num(o)
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __radd__(self, o):
        return self.__add__(o)

    def __sub__(self, o):
        o = _num(o)
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, o):
        o = _num(o)
        return Vec3(self.x * o.x, self.y * o.y, self.z * o.z)

    def __rmul__(self, o):
        return self.__mul__(o)

    def __truediv__(self, o):
        o = _num(o)
        return Vec3(self.x / o.x, self.y / o.y, self.z / o.z)

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o):
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    def length_squared(self):
        return self.dot(self)

    def length(self):
        return math.sqrt(self.length_squared())

    def normalized(self):
        n = self.length()
        return Vec3(self.x / n, self.y / n, self.z / n)

    def __getitem__(self, i):
        return (self.x, self.y, self.z)[i]

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __repr__(self):
        return f"Vec3({self.x:.4g}, {self.y:.4g}, {self.z:.4g})"


# geometry constants
ORIGIN = Vec3(0.0, 0.0, 0.0)
EPS = 1.0e-4


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def reflect(wi, n):
    return wi - 2.0 * wi.dot(n) * n


def refract(wi, n, eta):
    """Refract wi (pointing against normal convention) through a surface.

    eta = ior_incident / ior_transmitted. n points toward the incident side.
    Returns None on total internal reflection.
    """
    cosi = -wi.dot(n)
    sin2t = eta * eta * max(0.0, 1.0 - cosi * cosi)
    if sin2t > 1.0:
        return None
    cost = math.sqrt(1.0 - sin2t)
    return eta * wi + (eta * cosi - cost) * n


def schlick(cosine, ior):
    r0 = (1.0 - ior) / (1.0 + ior)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * (1.0 - cosine) ** 5
