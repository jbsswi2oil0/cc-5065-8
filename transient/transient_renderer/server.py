"""HTTP backend for the transient rendering experiment window.

Every render endpoint calls transient_renderer.render() live; no precomputed
or fixed slices are ever returned. The CLI remains a separate helper.
"""
import json
import math
import threading
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .render import render, pixel_profile, evaluate_gates, readout_decision
from .scenes import PRESETS

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "web")

# traced result cache keyed by tracing parameters (gates excluded), so gate
# edits reuse the exact path records instead of re-tracing
_trace_cache = {}
_cache_lock = threading.Lock()


def _trace_key(p):
    keys = ("scene", "samples", "bounces", "kernel", "kernel_width", "dt",
            "t_min", "t_max", "historical", "coarse_dt", "width", "height",
            "seed")
    return tuple((k, p.get(k)) for k in keys)


def _trace(params):
    key = _trace_key(params)
    with _cache_lock:
        if key in _trace_cache:
            return _trace_cache[key]
    result = render(
        params["scene"],
        samples=int(params.get("samples", 12)),
        max_bounces=int(params.get("bounces", 5)),
        kernel_kind=params.get("kernel", "point"),
        kernel_width_ns=float(params.get("kernel_width", 0.0)),
        dt_ns=float(params.get("dt", 0.1)),
        t_min=float(params.get("t_min", 0.0)),
        t_max=float(params.get("t_max", 250.0)),
        historical=bool(params.get("historical", False)),
        coarse_dt_ns=float(params.get("coarse_dt", 2.0)),
        gates=params.get("gates"),
        resolution=(int(params.get("width", 36)), int(params.get("height", 27))),
        seed=int(params.get("seed", 12345)),
    )
    with _cache_lock:
        _trace_cache[key] = result
    return result


def run_render(params):
    result = _trace(params)
    if params.get("gates"):
        result["gates"] = [
            {"name": g.get("name", f"G{i+1}"),
             "start": float(g["start"]), "end": float(g["end"])}
            for i, g in enumerate(params["gates"])]
        result["gate_readout"] = evaluate_gates(result)
        result["readout_decision"] = readout_decision(result)
    return serialize(result)


def regate(params, gates):
    """Re-evaluate exposure gates on existing traced records only."""
    result = _trace(params)
    result["gates"] = [
        {"name": g.get("name", f"G{i+1}"),
         "start": float(g["start"]), "end": float(g["end"])}
        for i, g in enumerate(gates)]
    result["gate_readout"] = evaluate_gates(result)
    result["readout_decision"] = readout_decision(result)
    return serialize(result)


def serialize(result, ui_max_bins=260):
    width, height = result["resolution"]
    n = result["time_grid"]["n_bins"]
    stride = max(1, math.ceil(n / ui_max_bins))

    def thin(values):
        if stride == 1:
            return list(values)
        out = []
        for i in range(0, len(values), stride):
            chunk = values[i:i + stride]
            out.append(sum(chunk) / len(chunk))
        return out

    thin_frames = [thin(frame) for frame in result["frames"]][::stride]
    pixel_time_lum = [
        thin([(b[0] + b[1] + b[2]) / 3.0 for b in px_bins])
        for px_bins in result["pixel_time"]
    ]
    pixel_opl = [thin(bins) for bins in result["pixel_opl"]]
    grid = result["time_grid"]
    ui_grid = {"t_min": grid["t_min"], "dt": grid["dt"] * stride,
               "n_bins": len(thin_frames)}
    coarse = result["coarse"]
    if coarse is not None:
        coarse = {"grid": coarse["grid"], "bins": coarse["bins"]}
    return {
        "scene": result["preset"],
        "scene_name": result["scene_name"],
        "units": result["units"],
        "resolution": [width, height],
        "samples": result["samples"],
        "kernel": result["kernel"],
        "time_grid": ui_grid,
        "full_time_grid": {k: v for k, v in grid.items() if k != "edges"},
        "opl_grid": result["opl_grid"],
        "frames": thin_frames,
        "steady": _normalize_steady(result["steady"]),
        "pixel_time": pixel_time_lum,
        "pixel_opl": pixel_opl,
        "route_energy": result["route_energy"],
        "energy_total": result["energy_total"],
        "energy_lost_edge": result["energy_lost_edge"],
        "gate_readout": result["gate_readout"],
        "readout_decision": result["readout_decision"],
        "expected_analytic": result["expected"],
        "coarse": coarse,
        "gates": result["gates"],
        "bin_stride": stride,
    }


def _normalize_steady(steady):
    m = max(max(sum(c) / 3.0 for c in row) for row in steady) or 1.0
    return [[[c / m for c in rgb] for rgb in row] for row in steady]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._serve_static("index.html", "text/html; charset=utf-8")
        if path == "/app.js":
            return self._serve_static("app.js", "application/javascript; charset=utf-8")
        if path == "/styles.css":
            return self._serve_static("styles.css", "text/css; charset=utf-8")
        if path == "/api/scenes":
            return self._send(200, {"scenes": sorted(PRESETS.keys())})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", 0))
        try:
            params = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})
        if path == "/api/render":
            try:
                return self._send(200, run_render(params))
            except Exception as exc:  # noqa: BLE001 - report to UI
                return self._send(500, {"error": str(exc)})
        if path == "/api/pixel":
            # pixel profile from an already-rendered parameter set
            try:
                full = run_render(params)
                px = int(params["px"])
                py = int(params["py"])
                prof = _profile_from_payload(full, px, py)
                return self._send(200, prof)
            except Exception as exc:  # noqa: BLE001
                return self._send(500, {"error": str(exc)})
        if path == "/api/regate":
            try:
                gates = params.pop("gates")
                return self._send(200, regate(params, gates))
            except Exception as exc:  # noqa: BLE001
                return self._send(500, {"error": str(exc)})
        self._send(404, {"error": "not found"})

    def _serve_static(self, name, ctype):
        full = os.path.join(WEB_DIR, name)
        if not os.path.isfile(full):
            return self._send(404, {"error": f"{name} missing"})
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype)


def _profile_from_payload(payload, px, py):
    width = payload["resolution"][0]
    idx = py * width + px
    lum = payload["pixel_time"][idx]
    ob = payload["pixel_opl"][idx]
    grid = payload["time_grid"]
    total = sum(lum)
    centroid = (
        sum((grid["t_min"] + (k + 0.5) * grid["dt"]) * e
            for k, e in enumerate(lum)) / total
    ) if total > 0 else None
    return {
        "px": px, "py": py,
        "time_luminance": lum,
        "opl_bins": ob,
        "centroid_ns": centroid,
        "steady_rgb": payload["steady"][py][px],
        "total_energy": total,
        "ui_dt_ns": grid["dt"],
    }


def serve(host="127.0.0.1", port=8000):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"transient experiment window: http://{host}:{port}/")
    httpd.serve_forever()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    serve(a.host, a.port)
