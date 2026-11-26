"""Adjacent OPL straddling a bin boundary: point vs finite-width deposition."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transient_renderer.kernel import Kernel, make_grid, deposit, centroid


def main():
    dt = 0.2
    grid = make_grid(0.0, dt, 100)
    boundary = 5.0                      # exact boundary between bins 24/25
    w = 1.0

    # two paths with adjacent OPL, one landing exactly on the boundary
    arrivals = (boundary - 0.13, boundary)
    weights = (2.0, 3.0)

    # point deposition: each path lands in exactly one bin, total conserved
    pb = [0.0] * 100
    for t, wt in zip(arrivals, weights):
        deposit(pb, t, wt, Kernel(Kernel.POINT), grid)
    assert abs(sum(pb) - sum(weights)) < 1e-12
    nz = [(k, v) for k, v in enumerate(pb) if v > 0]
    print("point deposition non-zero bins:", nz)
    assert nz == [(24, 2.0), (25, 3.0)]

    # finite-width box pulse: boundary path is split across bins but the
    # whole-segment light energy is still exactly the path weight
    bb = [0.0] * 100
    kernel = Kernel(Kernel.BOX, width=w)
    for t, wt in zip(arrivals, weights):
        deposit(bb, t, wt, kernel, grid)
    total = sum(bb)
    print("box total energy:", round(total, 6), "(expected 5.0)")
    assert abs(total - 5.0) < 1e-12

    # the boundary path alone spreads over (w/dt)+1 = 6 bins: support
    # [4.5, 5.5] covers bins 22..27, split symmetrically at bin 25's edge
    single = [0.0] * 100
    deposit(single, boundary, 1.0, Kernel(Kernel.BOX, w), grid)
    spread = [k for k, v in enumerate(single) if v > 0]
    print("boundary box pulse spread over bins:", spread[0], "..", spread[-1])
    assert spread[0] == 22 and spread[-1] == 27
    assert abs(sum(single) - 1.0) < 1e-12
    # symmetric energy on either side of the boundary
    left = sum(single[:25])
    right = sum(single[25:])
    assert abs(left - 0.5) < 1e-12 and abs(right - 0.5) < 1e-12

    # arrival centroid: point and box kernels must both recover the
    # energy-weighted mean of the true arrivals
    true_centroid = sum(t * wt for t, wt in zip(arrivals, weights)) / 5.0
    c_point = centroid(pb, grid)
    c_box = centroid(bb, grid)
    print(f"true centroid {true_centroid:.4f} | point {c_point:.4f} | box {c_box:.4f}")
    # point uses bin centres, box is continuous; both stay within half a bin
    assert abs(c_point - true_centroid) < dt
    assert abs(c_box - true_centroid) < dt

    # a pulse straddling the outer grid edge reports lost energy instead of
    # silently dropping it
    edge = [0.0] * 10
    g_edge = make_grid(0.0, dt, 10)
    res = deposit(edge, 1.95, 1.0, Kernel(Kernel.BOX, 0.4), g_edge)
    print("edge deposition kept %.4f, lost %.4f" %
          (res["deposited"], res["lost_edge"]))
    assert res["lost_edge"] > 0.0
    assert abs(res["deposited"] + res["lost_edge"] - 1.0) < 1e-12

    print("BOUNDARY STRADDLE TESTS PASS")


if __name__ == "__main__":
    main()
