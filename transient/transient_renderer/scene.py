"""Scene description: materials, primitives, point lights and a pinhole camera."""
from dataclasses import dataclass
from .vector import Vec3, reflect, refract, schlick, clamp

INF = float("inf")


class Material:
    DIFFUSE = "diffuse"
    MIRROR = "mirror"
    DIELECTRIC = "dielectric"

    def __init__(self, kind, albedo=Vec3(0.8, 0.8, 0.8), ior=1.5, name=""):
        self.kind = kind
        self.albedo = albedo
        self.ior = ior
        self.name = name


@dataclass
class Hit:
    t: float
    point: Vec3
    normal: Vec3
    primitive: object


class Ray:
    __slots__ = ("origin", "direction", "tmax")

    def __init__(self, origin, direction, tmax=INF):
        self.origin = origin
        self.direction = direction
        self.tmax = tmax


class Sphere:
    def __init__(self, center, radius, material, name="sphere"):
        self.center = center
        self.radius = radius
        self.material = material
        self.name = name

    def intersect(self, ray):
        oc = ray.origin - self.center
        b = oc.dot(ray.direction)
        c = oc.length_squared() - self.radius * self.radius
        disc = b * b - c
        if disc < 0.0:
            return None
        sq = disc ** 0.5
        t = -b - sq
        if t < 1e-4 or t > ray.tmax:
            t = -b + sq
            if t < 1e-4 or t > ray.tmax:
                return None
        p = ray.origin + t * ray.direction
        n = (p - self.center) / self.radius
        return Hit(t, p, n, self)


class Plane:
    def __init__(self, point, normal, material, name="plane"):
        self.point = point
        self.normal = normal.normalized()
        self.material = material
        self.name = name

    def intersect(self, ray):
        denom = ray.direction.dot(self.normal)
        if abs(denom) < 1e-8:
            return None
        t = (self.point - ray.origin).dot(self.normal) / denom
        if t < 1e-4 or t > ray.tmax:
            return None
        return Hit(t, ray.origin + t * ray.direction, self.normal, self)


class Rect:
    """Finite axis-aligned rectangle defined by a centre, two edge vectors
    and half-extents along them."""

    def __init__(self, center, e1, e2, half1, half2, material, name="rect"):
        self.center = center
        self.e1 = e1.normalized()
        self.e2 = e2.normalized()
        self.normal = self.e1.cross(self.e2).normalized()
        self.half1 = half1
        self.half2 = half2
        self.material = material
        self.name = name

    def intersect(self, ray):
        denom = ray.direction.dot(self.normal)
        if abs(denom) < 1e-8:
            return None
        t = (self.center - ray.origin).dot(self.normal) / denom
        if t < 1e-4 or t > ray.tmax:
            return None
        p = ray.origin + t * ray.direction
        rel = p - self.center
        if abs(rel.dot(self.e1)) > self.half1 + 1e-4:
            return None
        if abs(rel.dot(self.e2)) > self.half2 + 1e-4:
            return None
        n = self.normal
        if n.dot(ray.direction) > 0:
            n = -n
        return Hit(t, p, n, self)


class Box:
    """Axis-aligned box; used as room walls and as a dielectric medium volume."""

    def __init__(self, lo, hi, material, name="box"):
        self.lo = lo
        self.hi = hi
        self.material = material
        self.name = name

    def intersect(self, ray):
        tmin, tmax = -INF, INF
        naxis = -1
        for axis in range(3):
            o = ray.origin[axis]
            d = ray.direction[axis]
            lo, hi = self.lo[axis], self.hi[axis]
            if abs(d) < 1e-10:
                if o < lo or o > hi:
                    return None
                continue
            t1 = (lo - o) / d
            t2 = (hi - o) / d
            near, far = (t1, t2) if t1 < t2 else (t2, t1)
            if near > tmin:
                tmin = near
                naxis = axis
                nsign = -1.0 if d > 0 else 1.0
            tmax = min(tmax, far)
            if tmin > tmax:
                return None
        if tmin < 1e-4:
            # ray starts inside: exit surface is the relevant hit
            if tmax < 1e-4 or tmax > ray.tmax:
                return None
            t = tmax
            p = ray.origin + t * ray.direction
            n = self._exit_normal(p)
            return Hit(t, p, n, self)
        if tmin > ray.tmax:
            return None
        t = tmin
        p = ray.origin + t * ray.direction
        comp = [0.0, 0.0, 0.0]
        comp[naxis] = nsign
        n = Vec3(*comp)
        return Hit(t, p, n, self)

    def _exit_normal(self, p):
        # pick the face the point lies on
        for axis in range(3):
            if abs(p[axis] - self.lo[axis]) < 1e-4:
                c = [0.0, 0.0, 0.0]
                c[axis] = 1.0
                return Vec3(*c)
            if abs(p[axis] - self.hi[axis]) < 1e-4:
                c = [0.0, 0.0, 0.0]
                c[axis] = -1.0
                return Vec3(*c)
        return Vec3(0, 1, 0)

    def contains(self, p):
        return (self.lo.x <= p.x <= self.hi.x and
                self.lo.y <= p.y <= self.hi.y and
                self.lo.z <= p.z <= self.hi.z)


@dataclass
class PointLight:
    position: Vec3
    power: Vec3           # total emitted power (radiant flux), watts
    pulse_time: float     # emission time in nanoseconds
    name: str = "light"
    width: float = 0.0    # known probe response width (ns, std-dev for gaussian)


@dataclass
class Camera:
    eye: Vec3
    target: Vec3
    up: Vec3
    fov_y: float
    width: int
    height: int

    def basis(self):
        forward = (self.target - self.eye).normalized()
        right = forward.cross(self.up).normalized()
        up = right.cross(forward)
        return forward, right, up

    def ray_for(self, u, v):
        """u, v in [-1, 1] (image plane coords)."""
        import math
        forward, right, up = self.basis()
        half_h = math.tan(math.radians(self.fov_y) * 0.5)
        half_w = half_h * self.width / self.height
        d = (forward + right * (u * half_w) + up * (v * half_w)).normalized()
        return Ray(self.eye, d)


class Scene:
    def __init__(self, name="scene"):
        self.name = name
        self.primitives = []
        self.lights = []
        self.camera = None
        self.medium_ior = 1.0          # ior at camera location
        self.time_unit = "ns"
        self.length_unit = "m"
        # speed of light in vacuum, metres per nanosecond (exact-ish constant)
        self.c_vacuum = 0.299792458

    def add(self, primitive):
        self.primitives.append(primitive)
        return primitive

    def add_light(self, light):
        self.lights.append(light)
        return light

    def intersect(self, ray):
        best = None
        for prim in self.primitives:
            h = prim.intersect(ray)
            if h is not None and (best is None or h.t < best.t):
                best = h
        return best
