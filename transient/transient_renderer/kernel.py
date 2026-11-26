"""Temporal deposition kernels and time-bin integration.

Boundary rule (documented and testable):
    bin k covers the half-open interval  [t0 + k*dt, t0 + (k+1)*dt).
    An arrival exactly on a boundary t0+k*dt is assigned to bin k (the
    interval's left edge), i.e. index = floor((t - t0) / dt).

Kernels:
    point    : Dirac deposition into the single bin containing t.
    box      : uniform energy over [t-w/2, t+w/2], split analytically across
               every bin the interval touches (the "two-gate split" case).
    gaussian : normal density with std-dev w, truncated at +/- 4w. The lost
               tail is renormalized so the total deposited energy equals the
               path weight; the truncated fraction is reported by the kernel.

Every kernel deposits exactly the path weight somewhere (up to clipping at
the outer grid edge, which is also reported), so the time-bin integral equals
the steady-state result for the same path set.
"""
import math


class Kernel:
    POINT = "point"
    BOX = "box"
    GAUSSIAN = "gaussian"

    def __init__(self, kind=POINT, width=0.0, truncate_sigmas=4.0):
        if kind not in (self.POINT, self.BOX, self.GAUSSIAN):
            raise ValueError(f"unknown kernel {kind!r}")
        self.kind = kind
        self.width = float(width)
        self.truncate_sigmas = truncate_sigmas

    def support(self, t):
        """Return (lo, hi) of the kernel's finite support around time t."""
        if self.kind == self.POINT or self.width <= 0.0:
            return t, t
        if self.kind == self.BOX:
            h = 0.5 * self.width
            return t - h, t + h
        h = self.truncate_sigmas * self.width
        return t - h, t + h

    def fraction_in_interval(self, t, lo, hi):
        """Fraction of kernel energy falling in [lo, hi)."""
        if self.kind == self.POINT or self.width <= 0.0:
            return 1.0 if lo <= t < hi else 0.0
        if self.kind == self.BOX:
            klo, khi = t - 0.5 * self.width, t + 0.5 * self.width
            a = max(lo, klo)
            b = min(hi, khi)
            return max(0.0, b - a) / self.width
        s = self.width
        n = self.truncate_sigmas
        klo, khi = t - n * s, t + n * s
        a = max(lo, klo)
        b = min(hi, khi)
        if b <= a:
            return 0.0
        # normal CDF, then renormalize by the mass inside +/- n sigma
        cdf = lambda x: 0.5 * (1.0 + math.erf((x - t) / (s * math.sqrt(2.0))))
        kept = cdf(khi) - cdf(klo)
        return (cdf(b) - cdf(a)) / kept


def make_grid(t_min, dt, n_bins):
    return {
        "t_min": float(t_min),
        "dt": float(dt),
        "n_bins": int(n_bins),
        "edges": [t_min + i * dt for i in range(n_bins + 1)],
    }


def bin_index(t, grid):
    idx = int(math.floor((t - grid["t_min"]) / grid["dt"]))
    return idx


def deposit(bins, t, weight, kernel, grid):
    """Add path weight to time bins using the kernel.

    Returns dict with energy actually deposited and energy lost to the outer
    grid edge, so callers can verify conservation.
    """
    n = grid["n_bins"]
    dt = grid["dt"]
    t0 = grid["t_min"]
    lo, hi = kernel.support(t)

    first = max(0, int(math.floor((lo - t0) / dt)))
    last = min(n - 1, int(math.floor((hi - t0) / dt)))
    if kernel.kind == Kernel.POINT or kernel.width <= 0.0:
        idx = bin_index(t, grid)
        if 0 <= idx < n:
            _add(bins, idx, weight)
            return {"deposited": 1.0, "lost_edge": 0.0}
        return {"deposited": 0.0, "lost_edge": 1.0}

    deposited_frac = 0.0
    is_rgb = isinstance(weight, (tuple, list))
    for k in range(first, last + 1):
        blo = t0 + k * dt
        frac = kernel.fraction_in_interval(t, blo, blo + dt)
        if frac > 0.0:
            w = [c * frac for c in weight] if is_rgb else weight * frac
            _add(bins, k, w)
            deposited_frac += frac
    return {"deposited": deposited_frac, "lost_edge": 1.0 - deposited_frac}


def _add(bins, idx, value):
    """Add a scalar or RGB weight to a bin array."""
    cur = bins[idx]
    if isinstance(value, (tuple, list)) or hasattr(value, "x"):
        if hasattr(value, "x"):
            if cur == 0 or cur == 0.0:
                cur = [0.0, 0.0, 0.0]
            cur[0] += value.x
            cur[1] += value.y
            cur[2] += value.z
        else:
            if cur == 0 or cur == 0.0:
                cur = [0.0, 0.0, 0.0]
            for c in range(3):
                cur[c] += value[c]
        bins[idx] = cur
    else:
        bins[idx] = cur + value


def gate_energy(bins, t_start, t_end, grid):
    """Integrate bins inside an exposure gate [t_start, t_end)."""
    dt = grid["dt"]
    t0 = grid["t_min"]
    a = max(0, int(math.floor((t_start - t0) / dt)))
    b = min(grid["n_bins"] - 1, int(math.floor((t_end - 1e-12 - t0) / dt)))
    total = 0.0
    for k in range(a, b + 1):
        v = bins[k]
        total += sum(v) / 3.0 if isinstance(v, list) else v
    return total


def centroid(bins, grid):
    """Arrival-time centroid (energy-weighted mean) over populated bins."""
    num = 0.0
    den = 0.0
    for k, v in enumerate(bins):
        e = sum(v) / 3.0 if isinstance(v, list) else v
        if e:
            tc = grid["t_min"] + (k + 0.5) * grid["dt"]
            num += tc * e
            den += e
    return (num / den) if den > 0 else float("nan")
