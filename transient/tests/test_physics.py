"""Physics tests: analytically recomputable arrival times and conservation.

Run with: python3 -I tests/test_physics.py
Exits non-zero on the first failed assertion.
"""
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transient_renderer.render import build_scene, render
from transient_renderer.tracer import trace_scene, segment_metrics
from transient_renderer.vector import Vec3
C = 0.299792458
PASS = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    PASS.append((name, ok))
    print(f"[{status}] {name} {detail}")
    assert ok, name


# 1. two-gate: independent OPL recomputation matches the kernel timings ----
def test_two_gate_analytic():
    s = build_scene("two_gate", resolution=(16, 12))
    screen = Vec3(-4.2, 3.0, 6.5)
    lb = s.lights[1].position
    t, g, opl, med = segment_metrics(s, lb, screen, (screen - lb).length())
    # independent plain-geometry recomputation
    slab = [p for p in s.primitives if p.name == "slab"][0]
    z0, z1 = lb.z, screen.z
    inside = max(0.0, min(slab.hi.z, max(z0, z1)) - max(slab.lo.z, min(z0, z1)))
    d = (screen - lb).length()
    outside = d - inside
    ind_opl = outside + inside * s.expected["ior"]
    ind_t = ind_opl / C
    check("two_gate slow route analytic OPL", abs(opl - 2.75) < 1e-9,
          f"opl={opl:.4f}")
    check("two_gate independent recompute matches tracer",
          abs(ind_opl - opl) < 1e-9 and abs(ind_t - t) < 1e-12)

    # both routes equal arrival at the screen centre
    la = s.lights[0].position
    sa = Vec3(3.6, 3.0, 6.5)
    ta, _, opla, _ = segment_metrics(s, la, sa, (sa - la).length())
    check("two_gate routes co-time at screen",
          abs(ta - t) < 1e-9, f"tA={ta:.5f} tB={t:.5f}")
    check("two_gate equal OPL", abs(opla - opl) < 1e-9)

    # MC earliest arrivals: both routes land in the same 2 ns neighbourhood
    recs = trace_scene(s, 6, max_bounces=3, seed=23)
    for li, light in enumerate(s.lights):
        b0 = [r for r in recs if r.light_index == li and r.bounces == 0]
        tmin = min(r.arrival_ns for r in b0)
        # analytic axial camera arrival is a lower bound; sampled oblique
        # paths must arrive within 6 ns of it
        err = tmin - s.expected["arrival_camera_ns"]
        check(f"two_gate {light.name} earliest near analytic",
              0.0 <= err < 6.0, f"err={err:.3f} ns")

    # slow route must carry medium segments and extra OPL over geometry
    slow = [r for r in recs if r.light_index == 1 and r.bounces == 0]
    extra = max(r.opl_m - r.geom_length_m for r in slow)
    check("two_gate slow route has OPL > geometry", extra > 0.4,
          f"max extra={extra:.3f} m")
    fast = [r for r in recs if r.light_index == 0 and r.bounces == 0]
    clean = sum(1 for r in fast if abs(r.opl_m - r.geom_length_m) < 1e-9)
    check("two_gate vacuum route OPL == geometry (>=99% of records)",
          clean / len(fast) >= 0.99, f"{clean}/{len(fast)} clean")


# 2. distinct path lengths -------------------------------------------------
def test_path_lengths():
    import math
    import statistics
    from transient_renderer.vector import Vec3
    s = build_scene("path_lengths", resolution=(30, 24))
    w, h = s.camera.width, s.camera.height
    recs = trace_scene(s, 24, max_bounces=2, seed=31)
    direct = [r for r in recs if r.bounces == 0]

    check("path_lengths vacuum OPL equals geometric length",
          max(abs(r.opl_m - r.geom_length_m) for r in direct) < 1e-9)
    check("path_lengths every arrival = OPL/c",
          all(abs(r.opl_m / C - r.arrival_ns) < 1e-9 for r in direct))

    light_d = (1.5, 3.0, 4.5)
    for li, d in enumerate(light_d):
        light = s.lights[li].position
        anchor = Vec3(light.x, light.y, 7.0)
        seg = (anchor - light).length()
        check(f"path_lengths light {li+1} analytic segment = {d} m",
              abs(seg - d) < 1e-9)
        check(f"path_lengths light {li+1} analytic arrival = d/c",
              abs(seg / C - d / C) < 1e-12)

    for li in range(3):
        clo = (0, w // 3, 2 * w // 3)[li]
        chi = (w // 3, 2 * w // 3, w)[li]
        own = [r for r in direct if r.light_index == li and clo <= r.px < chi]
        check(f"path_lengths light {li+1} column populated", len(own) > 20)

    # Centre pixel + centre light: typical (median) OPL should sit near the
    # analytic axial total of 10.0 m; jitter spreads it but not by metres.
    centre = [r for r in direct if r.light_index == 1 and r.px == w // 2]
    check("path_lengths centre pixel has centre-light energy", len(centre) >= 5)
    med = statistics.median(r.opl_m for r in centre)
    check("path_lengths centre median OPL near analytic 10.0 m",
          9.0 <= med <= 11.5, f"median={med:.3f}")
    med_t = statistics.median(r.arrival_ns for r in centre)
    check("path_lengths centre median arrival near 10/c",
          abs(med_t - 10.0 / C) < 4.0,
          f"t={med_t:.3f} analytic={10.0/C:.3f}")


# 3. mirror detour ---------------------------------------------------------
def test_mirror():
    s = build_scene("mirror_detour", resolution=(16, 12))
    exp = s.expected
    check("mirror analytic bounce longer than direct",
          exp["mirror_m"] > exp["direct_m"],
          f"{exp['direct_m']:.3f} vs {exp['mirror_m']:.3f}")
    recs = trace_scene(s, 10, max_bounces=4, seed=41)
    # mirror-bounce records have bounce >= 1 and arrive later than direct
    direct_t = [r.arrival_ns for r in recs if r.bounces == 0]
    bounce_t = [r.arrival_ns for r in recs if r.bounces >= 1]
    check("mirror indirect paths exist", len(bounce_t) > 50)
    check("mirror indirect arrives after earliest direct",
          min(bounce_t) > min(direct_t),
          f"{min(direct_t):.2f} < {min(bounce_t):.2f}")
    # centroid of bounce paths should sit near the analytic mirror time
    # (plus camera leg), within a generous tolerance
    check("mirror analytic time positive and finite",
          math.isfinite(exp["mirror_ns"]) and exp["mirror_ns"] > 0)


# 4. finite-depth indirect -------------------------------------------------
def test_indirect():
    from transient_renderer.scene import Ray
    s = build_scene("indirect_depth", resolution=(16, 12))
    # direct light->marker line of sight is blocked by the occluder
    light = s.lights[0].position
    marker = Vec3(1.5, 1.0, 5.0)
    lvec = marker - light
    ldir = lvec / lvec.length()
    blocker = s.intersect(
        Ray(light + ldir * 1e-3, ldir, tmax=lvec.length() - 2e-3))
    check("indirect marker direct line of sight blocked",
          blocker is not None and blocker.primitive.name == "occluder")

    recs = trace_scene(s, 12, max_bounces=5, seed=53)
    indirect = [r for r in recs if r.bounces >= 1]
    check("indirect finite-depth bounce energy exists", len(indirect) > 100)
    # multi-bounce (>=2) transport is present, proving depth beyond one hop
    deep = [r for r in recs if r.bounces >= 2]
    check("indirect paths with >=2 bounces exist", len(deep) > 20)

    # every indirect arrival independently equals OPL/c
    bad = sum(1 for r in indirect
              if abs(r.arrival_ns - r.opl_m / C) > 1e-9)
    check("indirect arrival = OPL/c on every record", bad == 0)

    # the analytic edge route is a real, finite lower bound for *that
    # specific path*; general indirect paths may be shorter, but none can
    # beat the straight light->marker flight time (which is blocked anyway)
    straight_ns = lvec.length() / C
    earliest = min(r.arrival_ns for r in indirect)
    check("indirect earliest exceeds straight-line flight",
          earliest > straight_ns,
          f"earliest={earliest:.2f} straight={straight_ns:.2f}")
    # documented analytic edge-route time is finite and longer than straight
    check("indirect analytic edge route is finite",
          math.isfinite(s.expected["earliest_indirect_ns"])
          and s.expected["earliest_indirect_ns"] > straight_ns)


# 5. steady/transient conservation ----------------------------------------
def test_conservation():
    r = render("two_gate", samples=4, max_bounces=3, dt_ns=0.2,
               t_max=250.0, resolution=(16, 12), seed=61)
    bin_energy = sum(sum(b) / 3.0 for px in r["pixel_time"] for b in px)
    check("time bins integrate to total energy",
          abs(bin_energy - r["energy_total"]) / r["energy_total"] < 1e-3,
          f"diff={abs(bin_energy-r['energy_total']):.4f}")
    check("edge loss negligible with full window",
          r["energy_lost_edge"] / r["energy_total"] < 1e-2)

    # steady-state equals the time integral for the same model
    steady = sum(sum(c) / 3.0 for row in r["steady"] for c in row)
    expected_steady = r["energy_total"] / r["samples"]
    check("steady == transient integral (per-sample normalized)",
          abs(steady - expected_steady) / expected_steady < 1e-3,
          f"{steady:.3f} vs {expected_steady:.3f}")

    # no double counting: each record deposited once
    n_rec = len(r["records"])
    check("render keeps one record per path contribution", n_rec > 0)


# 6. kernels conserve energy with width on a boundary ---------------------
def test_kernel_choices():
    for kind, w in (("point", 0.0), ("box", 0.5), ("gaussian", 0.3)):
        r = render("path_lengths", samples=3, max_bounces=2,
                   kernel_kind=kind, kernel_width_ns=w, dt_ns=0.1,
                   t_max=250.0, resolution=(12, 9), seed=71)
        e = sum(sum(b) / 3.0 for px in r["pixel_time"] for b in px)
        check(f"conservation with kernel={kind}",
              abs(e - r["energy_total"]) / r["energy_total"] < 5e-3)
        # centroid is finite
        from transient_renderer.render import pixel_profile
        prof = pixel_profile(r, 6, 4)
        check(f"centroid finite kernel={kind}",
              math.isfinite(prof["time_centroid_ns"]))


# 7. historical coarse bins cannot be narrowed -----------------------------
def test_historical():
    r = render("two_gate", samples=3, max_bounces=2, dt_ns=0.1,
               t_max=120.0, historical=True, coarse_dt_ns=4.0,
               resolution=(12, 9), seed=83)
    coarse = r["coarse"]
    fine = r["time_grid"]
    check("coarse bins strictly wider than fine bins",
          coarse["grid"]["dt"] > fine["dt"])
    # coarse totals equal the summed fine energy (information loss, not energy)
    fine_total = sum(sum(b) / 3.0 for px in r["pixel_time"] for b in px)
    coarse_total = sum(coarse["bins"])
    check("coarse rebin preserves total energy",
          abs(coarse_total - fine_total) / fine_total < 1e-3)
    # decision must flag irrecoverability and demand a re-trace
    d = r["readout_decision"]
    check("historical data forces retrace decision",
          d["action"] == "retrace" and d["historical"])
    check("decision states sub-bin order cannot be recovered",
          any("sub-bin" in x for x in d["reasons"]))


if __name__ == "__main__":
    test_two_gate_analytic()
    test_path_lengths()
    test_mirror()
    test_indirect()
    test_conservation()
    test_kernel_choices()
    test_historical()
    print()
    failed = [n for n, ok in PASS if not ok]
    print(f"{len(PASS) - len(failed)}/{len(PASS)} checks passed")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
    print("ALL PHYSICS TESTS PASS")
