"""Recording-origin remap, reversible re-binning and tail-bin echo regressions.

Covers the reported defect: after moving the recording origin (t_min) the last
time bin could collect one extra echo, and shortening the right end of the
recording interval could pull an out-of-range reflection back into the final
cell instead of clipping it.

Checks, in order:
  1. Two-pulse integration: one arrival exactly on a bin edge, one inside the
     last bin; verify whole-interval integral and edge loss for point / box.
  2. Cross-edge pulse (straddles the *outer* recording edge) vs a pulse truly
     outside the recording range: two separate regressions.
  3. Reversible re-bin: shift the origin by half a bin and back; the arrival
     table, in-bin energy and displayed peak centroid are reproduced exactly.
  4. Strong echo in the last bin, then shorten t_max: the clipped reflection's
     energy is *recorded as lost*, never folded into the final cell.
  5. Different bin widths still preserve the whole-segment integral.
  6. Selection between the two candidate implementations: (a) keep exact
     arrival times and re-bin, vs (b) keep integrated bins and subdivide the
     edge bin. Only (a) is reversible, so (a) is the implemented one.

Run: python3 -I tests/test_origin_remap.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transient_renderer.kernel import (
    Kernel, make_grid, deposit, centroid, bin_index, bin_center)
from transient_renderer.render import render, rebin_at_origin
from transient_renderer.tracer import PathRecord

C = 0.299792458
DT = 0.1
PASS = []


def check(name, ok, detail=""):
    PASS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    assert ok, name


def grid(t_min, t_max, dt=DT):
    n = max(0, int(math.ceil((t_max - t_min) / dt - 1e-9)))
    return make_grid(t_min, dt, n, t_end=t_max)


def _binary_bins(bins):
    return [1 if v else 0 for v in bins]


# --------------------------------------------------------------------------
# 1. two-pulse integration with an edge arrival and a last-bin arrival
# --------------------------------------------------------------------------
def test_two_pulse_edge_integral():
    g = grid(0.0, 10.0)
    edge_t = 5.0          # exact boundary between bins 49 and 50
    tail_t = 9.95         # inside the last bin [9.9, 10.0)
    weights = (2.0, 3.0)

    # point deposition: edge arrival follows the half-open rule (bin 50),
    # tail arrival lands in the last bin.
    pb = [0.0] * g["n_bins"]
    lost = 0.0
    for t, w in zip((edge_t, tail_t), weights):
        lost += deposit(pb, t, w, Kernel(Kernel.POINT), g)["lost_edge"]
    nz = [(k, v) for k, v in enumerate(pb) if v]
    check("two-pulse point: edge->bin50, tail->last bin",
          nz == [(50, 2.0), (99, 3.0)], f"{nz}")
    check("two-pulse point: whole-interval integral = sum of weights",
          abs(sum(pb) - sum(weights)) < 1e-12 and lost == 0.0,
          f"sum={sum(pb):.6f} lost={lost:.3f}")

    # box kernel width 0.2: the edge arrival splits 50/50 across bins 49/50;
    # the tail pulse spills 0.05 ns past the window end (0.25 of its width).
    bb = [0.0] * g["n_bins"]
    lost = 0.0
    kb = Kernel(Kernel.BOX, 0.2)
    for t, w in zip((edge_t, tail_t), weights):
        r = deposit(bb, t, w, kb, g)
        lost += w * r["lost_edge"]
    # edge pulse (weight 2) split 1/1 across bins 49,50
    check("two-pulse box: edge pulse splits 1.0/1.0",
          abs(bb[49] - 1.0) < 1e-12 and abs(bb[50] - 1.0) < 1e-12,
          f"b49={bb[49]:.4f} b50={bb[50]:.4f}")
    # tail pulse (weight 3) at 9.95, width 0.2 -> support [9.85,10.05]:
    #   0.05 ns in bin 98 (0.25), 0.10 ns in bin 99 (0.50), 0.05 ns clipped.
    check("two-pulse box: tail pulse split 0.75 / 1.5 / clipped 0.75",
          abs(bb[98] - 0.75) < 1e-12 and abs(bb[99] - 1.5) < 1e-12
          and abs(lost - 0.75) < 1e-12,
          f"b98={bb[98]:.4f} b99={bb[99]:.4f} lost={lost:.4f}")
    check("two-pulse box: integral + edge loss = sum of weights",
          abs((sum(bb) + lost) - sum(weights)) < 1e-12,
          f"{sum(bb)+lost:.6f}")


# --------------------------------------------------------------------------
# 2. cross-edge vs truly-outside: two distinct regressions
# --------------------------------------------------------------------------
def test_cross_edge_vs_outside():
    g = grid(0.0, 10.0)

    # (a) cross-edge: pulse straddles the outer edge but has real support
    # inside -> partial deposit + partial loss, reported separately.
    b = [0.0] * g["n_bins"]
    r = deposit(b, 9.95, 1.0, Kernel(Kernel.BOX, 0.2), g)
    check("cross-edge pulse: partial deposit + partial loss",
          abs(r["deposited"] - 0.75) < 1e-12
          and abs(r["lost_edge"] - 0.25) < 1e-12,
          f"deposited={r['deposited']:.3f} lost={r['lost_edge']:.3f}")
    check("cross-edge pulse: conserved",
          abs(sum(b) + r["lost_edge"] - 1.0) < 1e-12)

    # (b) truly outside: wholly past t_end or before t_min -> 100% lost,
    # no bin (not even the nearest) receives anything. Times are chosen
    # beyond every kernel's half-support below.
    for t in (10.5, 25.0, -0.5):
        b = [0.0] * g["n_bins"]
        for kern in (Kernel(Kernel.POINT), Kernel(Kernel.BOX, 0.2),
                     Kernel(Kernel.GAUSSIAN, 0.1)):
            r = deposit(b, t, 1.0, kern, g)
            check(f"outside pulse t={t} {kern.kind}: fully lost, no bin",
                  r["lost_edge"] == 1.0 and r["deposited"] == 0.0
                  and not any(b),
                  f"lost={r['lost_edge']}")


# --------------------------------------------------------------------------
# crafted two-echo result: known path lengths, edge + last-bin arrival
# --------------------------------------------------------------------------
def _crafted_result(arrivals, weights, dt=DT, t_max=10.0, resolution=(2, 1)):
    """Render a tiny base result, then swap in exactly-known echo records and
    rebuild bins from them so the arrival table is fully controlled."""
    base = render("two_gate", samples=1, max_bounces=0, dt_ns=dt,
                  t_max=t_max, resolution=resolution, seed=7)
    records = []
    for i, (t, w) in enumerate(zip(arrivals, weights)):
        records.append(PathRecord(
            px=i % resolution[0], py=i // resolution[0],
            arrival_ns=t, geom_length_m=t * C, opl_m=t * C,
            weight=(w, w, w), light_index=0, bounces=0, medium_segments=0))
    base["records"] = records
    base["route_energy"] = {base["scene"].lights[0].name:
                            sum(sum(r.weight) / 3.0 for r in records)}
    return rebin_at_origin(base, 0.0, t_max)


def _lum(res, px):
    return [sum(b) / 3.0 for b in res["pixel_time"][px]]


def _total(res):
    return sum(sum(_lum(res, p))
               for p in range(res["resolution"][0] * res["resolution"][1]))


# --------------------------------------------------------------------------
# 3. reversible half-bin shift: arrival table, in-bin energy, peak centroid
# --------------------------------------------------------------------------
def test_reversible_half_bin_shift():
    # Echo 1 arrives exactly on a bin edge (5.0 ns); echo 2 sits inside the
    # last bin (9.95 ns). Known OPLs are c*t, so arrivals are independently
    # recomputable from the path lengths.
    arrivals = (5.0, 9.95)
    weights = (2.0, 3.0)
    r0 = _crafted_result(arrivals, weights)
    check("crafted: arrival table is the exact known echoes",
          [rec.arrival_ns for rec in r0["records"]] == list(arrivals),
          f"{[rec.arrival_ns for rec in r0['records']]}")
    check("crafted: echo1 on bin edge (bin 50), echo2 in last bin",
          abs(bin_index(5.0, r0["time_grid"]) - 50) == 0
          and bin_index(9.95, r0["time_grid"]) == r0["time_grid"]["n_bins"] - 1)

    # move origin down by half a bin, then back
    r_shift = rebin_at_origin(r0, -0.05, 10.0)
    r_back = rebin_at_origin(r_shift, 0.0, 10.0)

    # arrival table preserved through both moves (records never re-traced)
    check("shift round-trip: arrival table unchanged",
          [rec.arrival_ns for rec in r_back["records"]] == list(arrivals))

    # in-bin energy reproduced exactly
    check("shift round-trip: total in-bin energy preserved",
          abs(_total(r_back) - _total(r0)) < 1e-12
          and abs(_total(r_shift) - _total(r0)) < 1e-12,
          f"{_total(r0):.6f} -> {_total(r_shift):.6f} -> {_total(r_back):.6f}")

    # every bin reproduced bit-exactly (the key reversibility property)
    maxd = 0.0
    for p in range(r0["resolution"][0] * r0["resolution"][1]):
        for a, b in zip(_lum(r0, p), _lum(r_back, p)):
            maxd = max(maxd, abs(a - b))
    check("shift round-trip: per-bin energies bit-exact", maxd == 0.0,
          f"max diff={maxd}")

    # displayed peak centroid over the whole image: each origin stays within
    # half a bin of the true energy-weighted arrival; the round trip restores
    # it exactly.
    true_ct = sum(t * w for t, w in zip(arrivals, weights)) / sum(weights)

    def image_centroid(res):
        pooled = [0.0] * res["time_grid"]["n_bins"]
        for p in range(res["resolution"][0] * res["resolution"][1]):
            for k, e in enumerate(_lum(res, p)):
                pooled[k] += e
        return centroid(pooled, res["time_grid"])

    c0 = image_centroid(r0)
    cs = image_centroid(r_shift)
    cb = image_centroid(r_back)
    check("centroid within half a bin at each origin",
          abs(c0 - true_ct) <= DT / 2 + 1e-9
          and abs(cs - true_ct) <= DT / 2 + 1e-9,
          f"true={true_ct:.4f} c0={c0:.4f} cs={cs:.4f}")
    check("centroid restored exactly after round trip",
          abs(cb - c0) < 1e-12, f"{c0:.6f} -> {cb:.6f}")


# --------------------------------------------------------------------------
# 4. strong last-bin echo, then shorten t_max: clip, record, never refold
# --------------------------------------------------------------------------
def test_strong_echo_clipped_when_window_shortened():
    strong = 100.0
    r_full = _crafted_result((9.97,), (strong,), t_max=10.0)
    check("strong echo present in last bin before shortening",
          abs(_total(r_full) - strong) < 1e-9
          and abs(_lum(r_full, 0)[-1] - strong) < 1e-9,
          f"last bin={_lum(r_full, 0)[-1]:.3f}")

    # shorten the right end a little, below the echo's arrival
    r_cut = rebin_at_origin(r_full, 0.0, 9.90)
    last_bin = _lum(r_cut, 0)[-1]
    check("shortened window clips the strong echo",
          abs(_total(r_cut)) < 1e-12, f"in-window energy={_total(r_cut):.3e}")
    check("clipped echo is not folded into the final cell",
          abs(last_bin) < 1e-12, f"last bin={last_bin:.3e}")
    check("clipped echo energy is recorded as edge loss",
          abs(r_cut["energy_lost_edge"] - strong) < 1e-9,
          f"lost={r_cut['energy_lost_edge']:.3f} expected {strong}")

    # a width that no longer tiles the window: last bin narrower, still clipped
    r_cut2 = rebin_at_origin(r_full, 0.0, 9.95)
    check("shortened to 9.95 (narrow last bin): echo 9.97 still clipped",
          abs(r_cut2["energy_lost_edge"] - strong) < 1e-9
          and abs(_lum(r_cut2, 0)[-1]) < 1e-12,
          f"lost={r_cut2['energy_lost_edge']:.3f} last={_lum(r_cut2,0)[-1]:.3e}")


# --------------------------------------------------------------------------
# 5. different bin widths preserve the whole-segment integral
# --------------------------------------------------------------------------
def test_bin_width_preserves_integral():
    arrivals = (5.0, 9.95, 2.34, 9.75)
    weights = (2.0, 3.0, 1.0, 4.0)
    total = sum(weights)
    for dt in (0.05, 0.1, 0.2, 0.25, 0.5):
        g = grid(0.0, 10.0, dt)
        for origin in (0.0, -dt / 2, dt / 4):
            g = grid(origin, 10.0, dt)
            b = [0.0] * g["n_bins"]
            lost = 0.0
            for t, w in zip(arrivals, weights):
                lost += w * deposit(b, t, w, Kernel(Kernel.POINT), g)["lost_edge"]
            check(f"dt={dt} origin={origin:+.3f}: integral + loss = total",
                  abs(sum(b) + lost - total) < 1e-9,
                  f"{sum(b)+lost:.6f} vs {total:.1f}")

        # box kernel with a width narrower than the bin: same conservation
        g = grid(0.0, 10.0, dt)
        b = [0.0] * g["n_bins"]
        lost = 0.0
        for t, w in zip(arrivals, weights):
            lost += w * deposit(b, t, w, Kernel(Kernel.BOX, dt * 0.5),
                                g)["lost_edge"]
        check(f"dt={dt} box(0.5dt): integral + loss = total",
              abs(sum(b) + lost - total) < 1e-9,
              f"{sum(b)+lost:.6f}")


# --------------------------------------------------------------------------
# 6. candidate comparison: exact-arrival re-bin vs integrated edge subdivision
# --------------------------------------------------------------------------
def _rebin_integrated_uniform(bins, old_grid, new_grid):
    """Alternative implementation: keep integrated bins, assume each bin's
    energy is uniform across its width, and re-integrate onto the new grid
    (subdividing whatever falls on an edge). This loses sub-bin order."""
    out = [0.0] * new_grid["n_bins"]
    oe, ne = old_grid["edges"], new_grid["edges"]
    for k, e in enumerate(bins):
        if e == 0.0:
            continue
        lo, hi = oe[k], oe[k + 1]
        width = hi - lo
        for j in range(new_grid["n_bins"]):
            a, b = max(lo, ne[j]), min(hi, ne[j + 1])
            if b > a:
                out[j] += e * (b - a) / width
    return out


def test_candidate_comparison_selects_exact_arrivals():
    # Edge arrival deposited as a point pulse in bin 50 at origin 0.
    g0 = grid(0.0, 10.0)
    b0 = [0.0] * g0["n_bins"]
    deposit(b0, 5.0, 1.0, Kernel(Kernel.POINT), g0)

    g1 = grid(-0.05, 10.0)   # origin moved down half a bin

    # (a) implemented: re-deposit the exact arrival -> all in the bin that
    # actually contains t=5.0
    b_a = [0.0] * g1["n_bins"]
    deposit(b_a, 5.0, 1.0, Kernel(Kernel.POINT), g1)
    nz_a = [k for k, v in enumerate(b_a) if v]
    check("(a) exact-arrival re-bin keeps the pulse in one bin",
          nz_a == [bin_index(5.0, g1)], f"bins={nz_a}")

    # (b) alternative: integrated-bin uniform subdivision smears the pulse
    # across the new edge at 5.0 (no sub-bin info to know it was a Dirac).
    b_b = _rebin_integrated_uniform(b0, g0, g1)
    nz_b = [k for k, v in enumerate(b_b) if v > 1e-12]
    check("(b) integrated edge-subdivision smears across the new edge",
          len(nz_b) == 2, f"bins={nz_b}")

    # round trip back to origin 0: (a) exact, (b) lossy
    b_a_back = [0.0] * g0["n_bins"]
    # re-deposit exact arrival again (same information source)
    deposit(b_a_back, 5.0, 1.0, Kernel(Kernel.POINT), g0)
    b_b_back = _rebin_integrated_uniform(b_b, g1, g0)

    diff_a = max(abs(x - y) for x, y in zip(b0, b_a_back))
    diff_b = max(abs(x - y) for x, y in zip(b0, b_b_back))
    check("(a) exact-arrival round trip is bit-exact", diff_a == 0.0,
          f"max diff={diff_a}")
    check("(b) integrated edge-subdivision round trip is lossy", diff_b > 1e-3,
          f"max diff={diff_b:.4f} -> therefore not selected")

    # 'arrival table' survives exactly under (a): the records are the source
    # of truth and are not mutated by re-binning.
    r0 = _crafted_result((5.0, 9.95), (2.0, 3.0))
    r1 = rebin_at_origin(r0, -0.05, 10.0)
    check("(a) arrival table untouched by re-bin",
          [x.arrival_ns for x in r1["records"]]
          == [x.arrival_ns for x in r0["records"]])


def main():
    test_two_pulse_edge_integral()
    test_cross_edge_vs_outside()
    test_reversible_half_bin_shift()
    test_strong_echo_clipped_when_window_shortened()
    test_bin_width_preserves_integral()
    test_candidate_comparison_selects_exact_arrivals()

    print()
    failed = [n for n, ok in PASS if not ok]
    print(f"{len(PASS) - len(failed)}/{len(PASS)} checks passed")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
    print("ALL ORIGIN-REMAP TESTS PASS")


if __name__ == "__main__":
    main()
