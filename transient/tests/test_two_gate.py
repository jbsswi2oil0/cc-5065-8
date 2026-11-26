"""Two-gate headline experiment: pulse on the gate boundary is split."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transient_renderer.render import build_scene
from transient_renderer.kernel import Kernel

C = 0.299792458


def main():
    scene = build_scene("two_gate", resolution=(20, 15))
    t_cam = scene.t_common
    print("common camera arrival (ns):", round(t_cam, 4))

    # gates whose boundary sits exactly on the common arrival
    g1 = {"name": "G1", "start": t_cam - 2.0, "end": t_cam}
    g2 = {"name": "G2", "start": t_cam, "end": t_cam + 2.0}

    for kind, width in (("point", 0.0), ("box", 0.5), ("gaussian", 0.3)):
        k = Kernel(kind, width)
        rows = []
        for li, light in enumerate(scene.lights):
            f1 = k.fraction_in_interval(t_cam, g1["start"], g1["end"])
            f2 = k.fraction_in_interval(t_cam, g2["start"], g2["end"])
            rows.append((light.name, f1, f2))
        print(f"\nkernel={kind} width={width} ns, boundary at t={t_cam:.3f}")
        for name, f1, f2 in rows:
            print(f"  {name:12s} G1={f1:.4f} G2={f2:.4f} sum={f1+f2:.4f}")
            assert abs((f1 + f2) - 1.0) < 1e-9
        if kind == "point":
            # Dirac on the boundary follows the half-open rule: all in G2
            assert all(abs(f2 - 1.0) < 1e-12 for _, _, f2 in rows)
        else:
            # symmetric finite-width pulse: analytic 50/50 split
            for _, f1, f2 in rows:
                assert abs(f1 - 0.5) < 1e-9 and abs(f2 - 0.5) < 1e-9

    # analytic co-arrival: independent OPL recomputation for both routes
    from transient_renderer.vector import Vec3
    from transient_renderer.tracer import segment_metrics
    screen_b = Vec3(-4.2, 3.0, 6.5)
    screen_a = Vec3(3.6, 3.0, 6.5)
    lb = scene.lights[1].position
    la = scene.lights[0].position
    tb, _, opl_b, _ = segment_metrics(scene, lb, screen_b,
                                      (screen_b - lb).length())
    ta, _, opl_a, _ = segment_metrics(scene, la, screen_a,
                                      (screen_a - la).length())
    print("\nindependent recomputation:")
    print(f"  vacuum route: OPL={opl_a:.4f} m, screen t={ta:.5f} ns")
    print(f"  slow route:   OPL={opl_b:.4f} m, screen t={tb:.5f} ns")
    assert abs(opl_a - opl_b) < 1e-9 and abs(ta - tb) < 1e-12
    # slow route: geometric 2.0 m but OPL 2.75 m
    assert abs((screen_b - lb).length() - 2.0) < 1e-9
    assert abs(opl_b - 2.75) < 1e-9
    print("\nTWO-GATE SPLIT AND CO-ARRIVAL PASS")


if __name__ == "__main__":
    main()
