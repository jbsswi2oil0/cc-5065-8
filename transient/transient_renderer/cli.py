"""Command-line helper for the transient renderer.

The graphical experiment window (web UI) is the primary entry point; see
docs/USAGE.md. This CLI only assists transient rendering from scripts and
batch checks.

Examples:
  python3 -m transient_renderer.cli render --scene two_gate --samples 16
  python3 -I cli.py render --scene path_lengths --kernel gaussian --width 0.3
  python3 -I cli.py test-self
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transient_renderer.render import render, pixel_profile
from transient_renderer.kernel import bin_center
from transient_renderer.scenes import PRESETS, C


def _summary(result):
    g = result["time_grid"]
    peak_bin = max(
        range(g["n_bins"]),
        key=lambda k: sum(sum(px[k]) for px in result["pixel_time"]))
    peak_t = bin_center(peak_bin, g)
    return {
        "scene": result["scene_name"],
        "resolution": result["resolution"],
        "samples": result["samples"],
        "kernel": result["kernel"],
        "n_time_bins": g["n_bins"],
        "time_window_ns": [g["t_min"], g["t_end"]],
        "peak_time_ns": peak_t,
        "energy_total": result["energy_total"],
        "edge_loss": result["energy_lost_edge"],
        "route_energy": result["route_energy"],
        "gates": [
            {"name": x["name"], "window_ns": [x["start_ns"], x["end_ns"]],
             "energy": x["energy"], "per_route": x["per_route"]}
            for x in result["gate_readout"]["gates"]
        ],
        "leakage_fraction": result["gate_readout"]["leakage_fraction"],
        "readout": result["readout_decision"],
        "expected_analytic": result["expected"],
    }


def cmd_render(args):
    gates = None
    if args.gate:
        gates = []
        for i, spec in enumerate(args.gate):
            lo, hi = (float(x) for x in spec.split(","))
            gates.append({"name": f"G{i+1}", "start": lo, "end": hi})
    result = render(
        args.scene, samples=args.samples, max_bounces=args.bounces,
        kernel_kind=args.kernel, kernel_width_ns=args.width,
        dt_ns=args.dt, t_min=args.t_min, t_max=args.t_max,
        historical=args.historical, coarse_dt_ns=args.coarse_dt,
        gates=gates, resolution=(args.width_px, args.height_px),
        seed=args.seed)
    summary = _summary(result)
    if args.pixel:
        px, py = (int(v) for v in args.pixel.split(","))
        summary["pixel_profile"] = pixel_profile(result, px, py)
    out = json.dumps(summary, indent=2, default=_json_default)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)
        print("wrote", args.output)
    else:
        print(out)
    return 0


def _json_default(o):
    if hasattr(o, "__dict__"):
        return o.__dict__
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


def cmd_test_self(args):
    here = os.path.dirname(os.path.abspath(__file__))
    tests = os.path.join(here, "..", "tests")
    rc = 0
    for script in ("test_kernels.py", "test_physics.py"):
        path = os.path.abspath(os.path.join(tests, script))
        print(f"=== {script} ===")
        rc |= os.system(f"{sys.executable} -I {path}")
    return rc


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="transient_renderer.cli",
        description="Transient light-transport renderer (CLI helper).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("render", help="run a transient render")
    pr.add_argument("--scene", choices=sorted(PRESETS), default="two_gate")
    pr.add_argument("--samples", type=int, default=12)
    pr.add_argument("--bounces", type=int, default=5)
    pr.add_argument("--kernel", choices=("point", "box", "gaussian"),
                    default="point")
    pr.add_argument("--width", type=float, default=0.0,
                    help="probe response width in ns")
    pr.add_argument("--dt", type=float, default=0.1, help="bin width in ns")
    pr.add_argument("--t-min", type=float, default=0.0)
    pr.add_argument("--t-max", type=float, default=250.0)
    pr.add_argument("--historical", action="store_true")
    pr.add_argument("--coarse-dt", type=float, default=2.0)
    pr.add_argument("--gate", action="append",
                    help="gate as start,end in ns (repeatable, max 2)")
    pr.add_argument("--pixel", help="px,py to include a temporal profile")
    pr.add_argument("--width-px", type=int, default=32)
    pr.add_argument("--height-px", type=int, default=24)
    pr.add_argument("--seed", type=int, default=12345)
    pr.add_argument("--output", help="write JSON summary to this path")
    pr.set_defaults(func=cmd_render)

    pt = sub.add_parser("test-self", help="run kernel and physics tests")
    pt.set_defaults(func=cmd_test_self)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
