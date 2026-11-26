"""Render orchestration: path tracing + time/OPL binning + gate readout."""
import math
from .kernel import Kernel, make_grid, deposit, gate_energy, centroid
from .tracer import trace_scene
from .scenes import PRESETS


def build_scene(preset, resolution=None):
    fn = PRESETS[preset]
    if resolution is not None:
        return fn(resolution=resolution)
    return fn()


def render(preset, samples=8, max_bounces=6, kernel_kind="point",
           kernel_width_ns=0.0, dt_ns=0.05, t_min=0.0, t_max=250.0,
           opl_bin_m=0.05, historical=False, coarse_dt_ns=2.0,
           gates=None, resolution=None, seed=12345, progress=None):
    scene = build_scene(preset, resolution)
    records = trace_scene(scene, samples_per_pixel=samples,
                          max_bounces=max_bounces, seed=seed,
                          progress=progress)

    n_time = int(round((t_max - t_min) / dt_ns))
    time_grid = make_grid(t_min, dt_ns, n_time)
    n_opl = int(math.ceil((t_max * scene.c_vacuum * 2.2) / opl_bin_m))
    opl_max = n_opl * opl_bin_m
    opl_grid = make_grid(0.0, opl_bin_m, n_opl)
    kernel = Kernel(kernel_kind, kernel_width_ns)

    width, height = scene.camera.width, scene.camera.height
    # per-pixel RGB time bins
    pixel_time = [[[0.0, 0.0, 0.0] for _ in range(n_time)]
                  for _ in range(width * height)]
    pixel_opl = [[0.0] * n_opl for _ in range(width * height)]
    steady = [[[0.0, 0.0, 0.0] for _ in range(width)] for _ in range(height)]
    route_energy = {}
    energy_lost_edge = 0.0
    energy_total = 0.0

    for rec in records:
        idx = rec.py * width + rec.px
        w = rec.weight
        lum = sum(w) / 3.0
        energy_total += lum
        res = deposit(pixel_time[idx], rec.arrival_ns, list(w), kernel,
                      time_grid)
        energy_lost_edge += lum * res["lost_edge"]
        # OPL uses point deposition (length is an exact path property)
        ob = int(rec.opl_m / opl_bin_m)
        if 0 <= ob < n_opl:
            pixel_opl[idx][ob] += lum
        for c in range(3):
            steady[rec.py][rec.px][c] += w[c] / samples
        key = scene.lights[rec.light_index].name if 0 <= rec.light_index < len(scene.lights) else f"route{rec.light_index}"
        route_energy[key] = route_energy.get(key, 0.0) + lum

    # historical coarse-bin record: re-bin the *integrated* energy, keeping
    # no sub-bin information (proves ordering cannot be recovered).
    coarse = None
    if historical:
        n_coarse = int(round((t_max - t_min) / coarse_dt_ns))
        cgrid = make_grid(t_min, coarse_dt_ns, n_coarse)
        coarse_bins = [0.0] * n_coarse
        fine_per_coarse = max(1, int(round(coarse_dt_ns / dt_ns)))
        for px_bins in pixel_time:
            for k, rgb in enumerate(px_bins):
                ck = min(n_coarse - 1, k // fine_per_coarse)
                coarse_bins[ck] += (rgb[0] + rgb[1] + rgb[2]) / 3.0
        coarse = {"grid": cgrid, "bins": coarse_bins}

    frames = _frames(pixel_time, time_grid, width, height)
    result = {
        "preset": preset,
        "scene_name": scene.name,
        "units": {"time": scene.time_unit, "length": scene.length_unit,
                  "c_vacuum_m_per_ns": scene.c_vacuum},
        "resolution": [width, height],
        "samples": samples,
        "kernel": {"kind": kernel_kind, "width_ns": kernel_width_ns,
                   "boundary_rule": "[t0+k*dt, t0+(k+1)*dt); boundary t0+k*dt -> bin k"},
        "time_grid": time_grid,
        "opl_grid": {"t_min": 0.0, "dt": opl_bin_m, "n_bins": n_opl,
                     "opl_max_m": opl_max},
        "frames": frames,
        "steady": steady,
        "pixel_time": pixel_time,
        "pixel_opl": pixel_opl,
        "route_energy": route_energy,
        "energy_total": energy_total,
        "energy_lost_edge": energy_lost_edge,
        "expected": getattr(scene, "expected", {}),
        "scene": scene,
        "records": records,
        "coarse": coarse,
        "gates": gates or default_gates(scene),
    }
    result["gate_readout"] = evaluate_gates(result)
    result["readout_decision"] = readout_decision(result)
    return result


def default_gates(scene):
    t = getattr(scene, "t_common", None)
    if t is None:
        return [{"name": "G1", "start": 0.0, "end": 5.0},
                {"name": "G2", "start": 5.0, "end": 10.0}]
    # boundary on the analytic common arrival; screen sampling spreads the
    # direct pulse around it, so the gates span that observed support
    return [{"name": "G1", "start": t - 2.0, "end": t + 12.0},
            {"name": "G2", "start": t + 12.0, "end": t + 30.0}]


def _frames(pixel_time, grid, width, height):
    """Per-time-bin image slices (luminance), one frame per bin."""
    n = grid["n_bins"]
    frames = []
    for k in range(n):
        img = [0.0] * (width * height)
        for px in range(width * height):
            b = pixel_time[px][k]
            img[px] = (b[0] + b[1] + b[2]) / 3.0
        frames.append(img)
    return frames


def pixel_profile(result, px, py):
    width = result["resolution"][0]
    idx = py * width + px
    tb = result["pixel_time"][idx]
    ob = result["pixel_opl"][idx]
    grid = result["time_grid"]
    return {
        "px": px, "py": py,
        "time_bins": [(b[0] + b[1] + b[2]) / 3.0 for b in tb],
        "time_rgb": tb,
        "opl_bins": ob,
        "time_centroid_ns": centroid(tb, grid),
        "steady_rgb": result["steady"][py][px],
        "total_energy": sum((b[0] + b[1] + b[2]) / 3.0 for b in tb),
    }


def evaluate_gates(result):
    grid = result["time_grid"]
    width, height = result["resolution"]
    total = result["energy_total"] or 1e-12
    out = []
    for gate in result["gates"]:
        e = 0.0
        per_route = {}
        for px in range(width * height):
            e += gate_energy(result["pixel_time"][px], gate["start"],
                             gate["end"], grid)
        # route split: recompute per light using the records directly through
        # the same kernel, so a pulse on a boundary is split correctly
        for rec in result["records"]:
            k = Kernel(result["kernel"]["kind"], result["kernel"]["width_ns"])
            frac = k.fraction_in_interval(
                rec.arrival_ns, gate["start"], gate["end"])
            name = result["scene"].lights[rec.light_index].name
            per_route[name] = per_route.get(name, 0.0) + (
                sum(rec.weight) / 3.0) * frac
        out.append({
            "name": gate["name"],
            "start_ns": gate["start"],
            "end_ns": gate["end"],
            "energy": e,
            "fraction_of_total": e / total,
            "per_route": per_route,
        })
    gate_sum = sum(g["energy"] for g in out)
    return {
        "gates": out,
        "inside_gate_energy": gate_sum,
        "outside_gate_leakage": max(0.0, total - gate_sum),
        "inside_fraction": gate_sum / total,
        "leakage_fraction": max(0.0, total - gate_sum) / total,
        "total_energy": total,
    }


def readout_decision(result):
    """Decide whether current bins resolve the two routes, or whether the
    readout must re-trace with narrower gates / can only state a range."""
    ro = result["gate_readout"]
    kernel_width = result["kernel"]["width_ns"]
    dt = result["time_grid"]["dt"]
    g1, g2 = (ro["gates"] + [None, None])[:2]
    reasons = []

    # leakage check: tails outside both gates
    if ro["leakage_fraction"] > 0.1:
        reasons.append(
            f"{ro['leakage_fraction']*100:.1f}% of energy lies outside the "
            "two gates; widen gates or re-trace a wider window.")

    # can the two routes be distinguished?
    route_keys = list(result["route_energy"].keys())
    separable = True
    if len(route_keys) >= 2 and g1 and g2:
        g1_routes = g1["per_route"]
        g2_routes = g2["per_route"]
        for rk in route_keys:
            a = g1_routes.get(rk, 0.0)
            b = g2_routes.get(rk, 0.0)
            tot = a + b
            if tot > 0 and min(a, b) / tot > 0.25:
                separable = False
                reasons.append(
                    f"route {rk} splits >25% across both gates; arrival is "
                    "not localized to one gate.")

    # historical coarse data can never reveal sub-bin order
    if result.get("coarse"):
        cdt = result["coarse"]["grid"]["dt"]
        reasons.append(
            f"Historical record only stores whole {cdt:g} ns bin totals. "
            "Narrowing gates cannot recover sub-bin arrival order; only a "
            "re-trace can.")
        separable = False

    # pulse wider than a bin: gate splitting is valid (kernel known) but a
    # point arrival claim is not
    if kernel_width > 0 and kernel_width >= dt:
        reasons.append(
            f"Probe response width {kernel_width:g} ns >= bin {dt:g} ns: "
            "energy is split across gates via the known kernel; report a "
            "distribution, not a single arrival instant.")

    if separable and ro["leakage_fraction"] <= 0.1:
        action = "resolved"
        detail = ("Routes are distinguishable within the gates and leakage "
                  "is below 10%; no re-trace needed.")
    else:
        # state the resolvable time range rather than a false instant
        span = max(dt, kernel_width)
        if result.get("coarse"):
            span = max(span, result["coarse"]["grid"]["dt"])
        action = "retrace"
        detail = (f"Arrival can only be stated within a +-{span/2:g} ns "
                  "range at current readout. Re-trace with narrower bins/"
                  "gates to localize it.")
    return {"action": action, "resolvable_half_width_ns":
            max(dt, kernel_width) / 2.0,
            "historical": bool(result.get("coarse")),
            "reasons": reasons, "detail": detail}


def steady_total(result):
    return sum(sum(c) / 3.0 for row in result["steady"] for c in row)
