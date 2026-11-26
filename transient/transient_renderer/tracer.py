"""Transient unidirectional path tracer.

Every camera path keeps three independent accumulators per segment:
    * geom_length : sum of Euclidean segment lengths (metres)
    * opl         : sum of segment_length * segment_ior  (metres)
    * flight_time : sum of segment_length / (c_vacuum / segment_ior) (ns)
so multi-bounce delays are never estimated from the camera-to-hit straight
line. Arrival time = light.pulse_time + flight_time.

At every diffuse vertex a next-event-estimation contribution is recorded
(complete path to a light, with its own arrival time), then the random walk
continues for indirect bounces. A single camera path can therefore produce
several transient records at different arrival times.
"""
import math
from dataclasses import dataclass

from .rng import RNG
from .vector import Vec3, reflect, refract, schlick, clamp
from .scene import Material, Ray


@dataclass
class PathRecord:
    px: int
    py: int
    arrival_ns: float
    geom_length_m: float
    opl_m: float
    weight: tuple          # RGB radiance weight
    light_index: int
    bounces: int
    medium_segments: int


def _luminance(v):
    return 0.2126 * v.x + 0.7152 * v.y + 0.0722 * v.z


def _ior_at(scene, point):
    for prim in scene.primitives:
        if prim.material.kind == Material.DIELECTRIC and hasattr(prim, "contains"):
            if prim.contains(point):
                return prim.material.ior
    return scene.medium_ior


def trace_pixel(scene, px, py, samples, rng, max_bounces=8, rr_start=4):
    cam = scene.camera
    records = []
    half_h = math.tan(math.radians(cam.fov_y) * 0.5)
    half_w = half_h * cam.width / cam.height
    forward, right, up = cam.basis()

    for _ in range(samples):
        u = 2.0 * (px + rng.uniform()) / cam.width - 1.0
        v = 1.0 - 2.0 * (py + rng.uniform()) / cam.height
        d = (forward + right * (u * half_w) + up * (v * half_h)).normalized()
        records.extend(_trace_path(scene, cam.eye, d, rng, px, py,
                                   max_bounces, rr_start))
    return records


def _trace_path(scene, origin, direction, rng, px, py, max_bounces, rr_start):
    origin = Vec3(origin.x, origin.y, origin.z)
    direction = direction.normalized()
    throughput = Vec3(1.0, 1.0, 1.0)
    ior = _ior_at(scene, origin)
    flight_ns = 0.0
    geom = 0.0
    opl = 0.0
    medium_segments = 0
    out = []

    for bounce in range(max_bounces + 1):
        hit = scene.intersect(Ray(origin, direction))
        if hit is None:
            return out
        speed = scene.c_vacuum / ior
        flight_ns += hit.t / speed
        geom += hit.t
        opl += hit.t * ior
        mat = hit.primitive.material

        if mat.kind == Material.DIELECTRIC:
            entering = hit.normal.dot(direction) < 0.0
            n = hit.normal if entering else -hit.normal
            new_ior = mat.ior if entering else scene.medium_ior
            eta = ior / new_ior
            fr = schlick(clamp(-direction.dot(n), -1.0, 1.0), eta)
            if rng.uniform() < fr:
                direction = reflect(direction, n)
            else:
                td = refract(direction, n, eta)
                if td is None:
                    direction = reflect(direction, n)
                else:
                    direction = td
                    ior = new_ior
                    if entering:
                        medium_segments += 1
            origin = hit.point + direction * 1e-4
            continue

        # direct illumination at this vertex -------------------------------
        if mat.kind == Material.DIFFUSE:
            for li, light in enumerate(scene.lights):
                to_l = light.position - hit.point
                dist = to_l.length()
                ldir = to_l / dist
                ndl = hit.normal.dot(ldir)
                if ndl <= 0.0:
                    continue
                blocker = scene.intersect(
                    Ray(hit.point + ldir * 1e-4, ldir, tmax=dist - 2e-4))
                if blocker is not None and blocker.primitive.material.kind != Material.DIELECTRIC:
                    continue
                seg_time, seg_geom, seg_opl, seg_med = segment_metrics(
                    scene, hit.point, light.position, dist)
                contrib = (throughput * mat.albedo * light.power
                           * (ndl / (4.0 * math.pi * dist * dist)))
                out.append(PathRecord(
                    px=px, py=py,
                    arrival_ns=light.pulse_time + flight_ns + seg_time,
                    geom_length_m=geom + seg_geom,
                    opl_m=opl + seg_opl,
                    weight=(contrib.x, contrib.y, contrib.z),
                    light_index=li,
                    bounces=bounce,
                    medium_segments=medium_segments + seg_med,
                ))

        # BSDF continuation -------------------------------------------------
        if mat.kind == Material.MIRROR:
            direction = reflect(direction, hit.normal)
        elif mat.kind == Material.DIFFUSE:
            w = mat.albedo
            if bounce >= rr_start:
                p = clamp(_luminance(w), 0.05, 0.95)
                if rng.uniform() > p:
                    return out
                w = w / p
            throughput = throughput * w
            direction = _cosine_hemisphere(hit.normal, rng)
        else:
            return out

        origin = hit.point + direction * 1e-4

    return out


def _box_intervals(scene, a, b, dist):
    """Parametric t-intervals along a->b that lie inside dielectric boxes."""
    eps = 1e-6
    dirv = (b - a) / dist
    intervals = []
    for prim in scene.primitives:
        if prim.material.kind != Material.DIELECTRIC or not hasattr(prim, "lo"):
            continue
        t_near, t_far = -float("inf"), float("inf")
        ok = True
        for axis in range(3):
            o = a[axis]
            d = dirv[axis]
            lo, hi = prim.lo[axis], prim.hi[axis]
            if abs(d) < 1e-12:
                if o < lo - eps or o > hi + eps:
                    ok = False
                    break
                continue
            t1, t2 = (lo - o) / d, (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)
            if t_near > t_far:
                ok = False
                break
        if ok and t_far > 0.0 and t_near < dist:
            intervals.append((max(0.0, t_near), min(dist, t_far),
                              prim.material.ior))
    intervals.sort()
    merged = []
    for lo, hi, ior in intervals:
        if merged and lo <= merged[-1][1] + eps:
            mlo, mhi, mior = merged[-1]
            merged[-1] = (mlo, max(mhi, hi), max(mior, ior))
        else:
            merged.append((lo, hi, ior))
    return merged


def segment_metrics(scene, a, b, dist):
    """Flight time, geometric length, OPL and medium count for a straight
    segment, integrating dielectric boxes it passes through."""
    intervals = _box_intervals(scene, a, b, dist)
    total_time = 0.0
    total_opl = 0.0
    med = 0
    cursor = 0.0
    base_ior = scene.medium_ior
    for lo, hi, ior in intervals:
        if lo > cursor:
            d = lo - cursor
            total_time += d / (scene.c_vacuum / base_ior)
            total_opl += d * base_ior
        d = hi - lo
        if d > 0.0:
            total_time += d / (scene.c_vacuum / ior)
            total_opl += d * ior
            med += 1
        cursor = hi
    if cursor < dist:
        d = dist - cursor
        total_time += d / (scene.c_vacuum / base_ior)
        total_opl += d * base_ior
    return total_time, dist, total_opl, med


def _cosine_hemisphere(normal, rng):
    local = rng.unit_vector()
    helper = Vec3(0, 1, 0) if abs(normal.y) < 0.95 else Vec3(1, 0, 0)
    t = normal.cross(helper).normalized()
    bvec = t.cross(normal)
    return (t * local.x + bvec * local.y + normal * local.z).normalized()


def trace_scene(scene, samples_per_pixel=16, max_bounces=8, seed=12345,
                progress=None):
    width = scene.camera.width
    height = scene.camera.height
    all_records = []
    for py in range(height):
        for px in range(width):
            rng = RNG(seed + px * 1_000_003 + py * 9176)
            all_records.extend(trace_pixel(
                scene, px, py, samples_per_pixel, rng,
                max_bounces=max_bounces))
        if progress is not None:
            progress((py + 1) / height)
    return all_records
