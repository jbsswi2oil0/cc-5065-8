"""Deterministic counter-based RNG so renders and tests are reproducible."""

_MASK = (1 << 64) - 1


def splitmix64(seed):
    z = (seed + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


class RNG:
    """PCG-style stream: each sample is splitmix64 of a per-pixel counter."""

    def __init__(self, seed=1):
        self.seed = seed & _MASK
        self.counter = 0

    def _next_u64(self):
        self.counter += 1
        return splitmix64(self.seed ^ (self.counter * 0xD1B54A32D192ED03))

    def uniform(self):
        return (self._next_u64() >> 11) / float(1 << 53)

    def range(self, lo, hi):
        return lo + (hi - lo) * self.uniform()

    def unit_vector(self):
        # cosine-weighted hemisphere sampling about +Z
        r1 = self.uniform()
        r2 = self.uniform()
        phi = 6.283185307179586 * r1
        r = math_sqrt(r2)
        from .vector import Vec3
        return Vec3(r * math_cos(phi), r * math_sin(phi), math_sqrt(1.0 - r2))

    def sphere_direction(self):
        from .vector import Vec3
        z = self.range(-1.0, 1.0)
        a = self.uniform() * 6.283185307179586
        r = (max(0.0, 1.0 - z * z)) ** 0.5
        return Vec3(r * math_cos(a), r * math_sin(a), z)


import math as _math
math_sqrt = _math.sqrt
math_cos = _math.cos
math_sin = _math.sin
