"""Temporal deposition kernels and time-bin integration.

Boundary rule (documented and testable):
    bin k covers the half-open interval  [t0 + k*dt, t0 + (k+1)*dt).
    An arrival exactly on a boundary t0+k*dt is assigned to bin k (the
    interval's left edge), i.e. index = floor((t - t0) / dt).

    The recording interval itself is exactly [t_min, t_end).  t_end defaults
    to the grid edge t_min + n_bins*dt but render() pins it to the requested
    t_max, so shortening or shifting the recording window clips energy at the
    declared end instead of letting a rounded n_bins carry the edge outward.
    An arrival at or beyond t_end (or before t_min) is reported as lost; it is
    never folded back into the last bin.

    The boundary test adds a tiny relative epsilon (1e-9 bin units) so exact
    multiples such as 0.3/0.1 = 2.9999999999999996 still land in the interval's
    left bin, matching the documented rule rather than raw binary noise.

Kernels:
    point    : Dirac deposition into the single bin containing t.
    box      : uniform energy over [t-w/2, t+w/2], split analytically across
               every bin the interval touches (the "two-gate split" case).
    gaussian : normal density with std-dev w, truncated at +/- 4w. The lost
               tail is renormalized so the total deposited energy equals the
               path weight; the truncated fraction is reported by the kernel.

Every kernel deposits exactly the path weight somewhere (up to clipping at
the recording interval edges, which is also reported), so the time-bin
integral equals the steady-state result for the same path set.

Re-binning policy: deposits are always derived from exact arrival times
(`PathRecord.arrival_ns`).  Moving the recording origin therefore re-bins the
same arrival times and is reversible; carrying *integrated* bins across a
half-bin shift would require inventing sub-bin order and is not used.
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


def make_grid(t_min, dt, n_bins, t_end=None):
    """Build a uniform time grid covering exactly [t_min, t_end).

    n_bins sets the number of bins, but the recording interval's right edge is
    t_end (default t_min + n_bins*dt).  Passing an explicit t_end lets a
    caller keep the declared window even when (t_end - t_min)/dt does not
    divide evenly; the last bin may then be narrower than dt.
    """
    t_min = float(t_min)
    dt = float(dt)
    if t_end is None:
        t_end = t_min + n_bins * dt
    else:
        t_end = float(t_end)
    return {
        "t_min": t_min,
        "dt": dt,
        "n_bins": int(n_bins),
        "t_end": t_end,
        "edges": [t_min + i * dt for i in range(n_bins)] + [t_end],
    }


def _inside(t, grid):
    """Half-open membership test for the recording interval [t_min, t_end)."""
    return grid["t_min"] <= t < grid["t_end"]


def bin_index(t, grid, _eps=1e-9):
    """Index of the bin whose half-open interval [t0+k*dt, t0+(k+1)*dt) holds t.

    Rounds to the nearest multiple when within _eps bins, so exact decimal
    boundaries (0.3, 0.7, ...) are not split by binary representation error.
    """
    raw = (t - grid["t_min"]) / grid["dt"]
    nearest = round(raw)
    if abs(raw - nearest) <= _eps:
        raw = float(nearest)
    return int(math.floor(raw))


def bin_index_exclusive(t, grid, _eps=1e-9):
    """Index of the last bin that ends at or before t (an exclusive end).

    Used for [start, end) gates: a t landing exactly on bin boundary k maps to
    k-1, so the bin starting at t is excluded.
    """
    raw = (t - grid["t_min"]) / grid["dt"]
    nearest = round(raw)
    if abs(raw - nearest) <= _eps:
        return int(nearest) - 1
    return int(math.ceil(raw)) - 1


def deposit(bins, t, weight, kernel, grid):
    """Add path weight to time bins using the kernel.

    Returns dict with energy actually deposited and energy lost to the
    recording interval's edges, so callers can verify conservation.  Energy
    from an arrival outside [t_min, t_end) is reported as lost; it is never
    pulled into the nearest edge bin.
    """
    n = grid["n_bins"]
    dt = grid["dt"]
    t0 = grid["t_min"]
    t_end = grid["t_end"]

    if kernel.kind == Kernel.POINT or kernel.width <= 0.0:
        if _inside(t, grid):
            idx = min(bin_index(t, grid), n - 1)
            if 0 <= idx < n:
                _add(bins, idx, weight)
                return {"deposited": 1.0, "lost_edge": 0.0}
        return {"deposited": 0.0, "lost_edge": 1.0}

    # Finite-support kernel: only the part intersecting [t_min, t_end) is
    # deposited; the rest (including a pulse wholly outside the window) is
    # reported as lost at the edges.
    lo, hi = kernel.support(t)
    a = max(lo, t0)
    b = min(hi, t_end)
    if b <= a:
        return {"deposited": 0.0, "lost_edge": 1.0}

    first = max(0, bin_index(a, grid))
    last = min(n - 1, bin_index(b, grid))
    deposited_frac = 0.0
    is_rgb = isinstance(weight, (tuple, list))
    for k in range(first, last + 1):
        blo = t0 + k * dt
        bhi = min(t_end, blo + dt)
        frac = kernel.fraction_in_interval(t, blo, bhi)
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
    """Integrate bins inside an exposure gate [t_start, t_end).

    A gate boundary landing exactly on a bin boundary excludes the bin that
    starts at t_end (half-open, matching the deposition rule).
    """
    a = max(0, bin_index(t_start, grid))
    b = min(grid["n_bins"] - 1, bin_index_exclusive(t_end, grid))
    total = 0.0
    for k in range(a, b + 1):
        v = bins[k]
        total += sum(v) / 3.0 if isinstance(v, list) else v
    return total


def bin_center(k, grid):
    """Centre of bin k, honouring a possibly narrower final bin."""
    edges = grid["edges"]
    return 0.5 * (edges[k] + edges[k + 1])


def centroid(bins, grid):
    """Arrival-time centroid (energy-weighted mean) over populated bins."""
    num = 0.0
    den = 0.0
    for k, v in enumerate(bins):
        e = sum(v) / 3.0 if isinstance(v, list) else v
        if e:
            num += bin_center(k, grid) * e
            den += e
    return (num / den) if den > 0 else float("nan")
