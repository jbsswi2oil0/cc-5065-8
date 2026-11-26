"""Preset scenes. All distances are metres, all times are nanoseconds.

c_vacuum = 0.299792458 m/ns. Flight time through a segment is
    sum(length_i * ior_i) / c_vacuum
i.e. optical path length (OPL) divided by c.
"""
import math
from .vector import Vec3
from .scene import (Scene, Sphere, Plane, Box, PointLight, Camera, Material)

C = 0.299792458


def _room(name, width=12.0, height=6.0, depth=14.0):
    scene = Scene(name)
    scene.c_vacuum = C
    wall = Material(Material.DIFFUSE, Vec3(0.55, 0.55, 0.58), name="wall")
    floor_m = Material(Material.DIFFUSE, Vec3(0.35, 0.32, 0.30), name="floor")
    scene.add(Plane(Vec3(0, 0, 0), Vec3(0, 1, 0), floor_m, "floor"))
    scene.add(Plane(Vec3(0, height, 0), Vec3(0, -1, 0), wall, "ceiling"))
    scene.add(Plane(Vec3(-width / 2, 0, 0), Vec3(1, 0, 0), wall, "left"))
    scene.add(Plane(Vec3(width / 2, 0, 0), Vec3(-1, 0, 0), wall, "right"))
    scene.add(Plane(Vec3(0, 0, depth), Vec3(0, 0, -1), wall, "back"))
    return scene


# --------------------------------------------------------------------------
# Scene 1: two-gate experiment
# --------------------------------------------------------------------------
def two_gate(ior=1.5, resolution=(56, 42)):
    """Two equal-OPL routes arrive at the same instant on a diffuse screen.

    The camera looks at a vertical screen on the back wall (z=6.5 m). Both
    point lights face the screen from the camera side.

    Route A (vacuum): L_A in air, screen distance OPL = 2.75 m.
    Route B (slow medium): L_B geometrically 2.0 m from the screen, 1.5 m of
        that inside a dielectric slab (ior), so OPL_B = 0.5 + 1.5*ior
        = 2.75 m at ior 1.5.
    Both pulses reach the screen at t = 2.75/c ns. Arrival at the camera
    adds the fixed screen->camera leg (~6.5 m / c) to both routes equally.
    """
    s = _room("two_gate")
    w, h = resolution
    s.camera = Camera(Vec3(0, 3.0, 0), Vec3(0, 3.0, 1), Vec3(0, 1, 0),
                      fov_y=55, width=w, height=h)

    screen_m = Material(Material.DIFFUSE, Vec3(0.9, 0.9, 0.9), "screen")
    # two screen panels at the same depth z=6.5
    s.add(Plane(Vec3(0.0, 3.0, 6.5), Vec3(0, 0, -1), screen_m, "screen_slow"))
    s.add(Plane(Vec3(3.2, 3.0, 6.5), Vec3(0, 0, -1), screen_m, "screen_fast"))

    box_m = Material(Material.DIELECTRIC, ior=ior, name="slow_medium")
    # 1.5 m thick slab covering the LEFT half of the screen only, from the
    # left wall (x=-6) to x=-2.5; any ray ending at the right half stays
    # outside it so the vacuum route never enters the medium.
    s.add(Box(Vec3(-6.0, 1.0, 5.0), Vec3(-2.5, 5.0, 6.5), box_m, "slab"))

    vacuum_opl = 0.5 + 1.5 * ior          # 2.75 m at ior 1.5
    # route A: vacuum light, axial, 2.75 m from the right screen panel
    s.add_light(PointLight(
        Vec3(3.6, 3.0, 6.5 - vacuum_opl), Vec3(60, 60, 60),
        pulse_time=0.0, name="L_A_vacuum"))
    # route B: light 2.0 m before the left screen; 0.5 m air then 1.5 m slab
    s.add_light(PointLight(
        Vec3(-4.2, 3.0, 4.5), Vec3(60, 60, 60),
        pulse_time=0.0, name="L_B_slow"))

    s.t_common = (vacuum_opl + 6.5) / C
    s.screen_z = 6.5
    s.expected = {
        "vacuum_geometric_m": vacuum_opl,
        "slow_geometric_m": 2.0,
        "slow_opl_m": vacuum_opl,
        "arrival_screen_ns": vacuum_opl / C,
        "screen_camera_m": 6.5,
        "arrival_camera_ns": (vacuum_opl + 6.5) / C,
        "slab_thickness_m": 1.5,
        "ior": ior,
    }
    return s


# --------------------------------------------------------------------------
# Scene 2: distinct path lengths
# --------------------------------------------------------------------------
def path_lengths(resolution=(56, 42)):
    """Three point lights at known distances from a common back screen.

    A diffuse screen at z=7 m faces the camera. Three lights sit on parallel
    optical axes (one per screen column) at light->screen distances of 1.5,
    3.0 and 4.5 m, so their screen arrival times are d/c and their ordering
    is unambiguous. The camera leg is identical for all three columns.
    """
    s = _room("path_lengths")
    w, h = resolution
    s.camera = Camera(Vec3(0, 3, 0), Vec3(0, 3, 1), Vec3(0, 1, 0),
                      fov_y=50, width=w, height=h)
    screen_m = Material(Material.DIFFUSE, Vec3(0.9, 0.9, 0.9), "screen")
    s.add(Plane(Vec3(0, 3.0, 7.0), Vec3(0, 0, -1), screen_m, "screen"))
    for i, d in enumerate((1.5, 3.0, 4.5)):
        x = (-2.6, 0.0, 2.6)[i]
        s.add_light(PointLight(
            Vec3(x, 3.0, 7.0 - d), Vec3(50, 50, 50),
            pulse_time=0.0, name=f"L{i+1}_{d}m"))
        s.add(Sphere(Vec3(x, 1.0, 7.0 - d), 0.2,
                     Material(Material.DIFFUSE, Vec3(0.8, 0.8, 0.8)),
                     f"marker_{d}m"))
    s.screen_z = 7.0
    s.expected = {f"light_{d}m_screen_ns": d / C for d in (1.5, 3.0, 4.5)}
    return s


# --------------------------------------------------------------------------
# Scene 3: mirror detour
# --------------------------------------------------------------------------
def mirror_detour(resolution=(56, 42)):
    """Light reaches a diffuse marker directly and via a flat mirror.

    The mirror path length is analytically the same as a straight segment to
    the mirror image of the marker, giving an independently recomputable
    arrival time.
    """
    s = _room("mirror_detour")
    w, h = resolution
    s.camera = Camera(Vec3(0, 2.0, 0), Vec3(0, 2.4, 1), Vec3(0, 1, 0),
                      fov_y=55, width=w, height=h)
    # diffuse back wall the camera sees directly
    marker_m = Material(Material.DIFFUSE, Vec3(0.85, 0.85, 0.85), "marker")
    s.add(Sphere(Vec3(-1.6, 1.4, 6.0), 0.4, marker_m, "marker"))
    # finite flat mirror in the upper-right of the view; the camera still
    # sees the diffuse back wall and marker elsewhere
    from .scene import Rect
    mirror_m = Material(Material.MIRROR, name="mirror")
    m_center = Vec3(2.6, 3.6, 4.2)
    m_normal = Vec3(-0.55, 0.0, 0.835).normalized()
    # build an orthonormal frame on the mirror plane (vertical + tangent)
    m_up = Vec3(0, 1, 0)
    m_tan = m_up.cross(m_normal).normalized()
    s.add(Rect(m_center, m_tan, m_up, 0.9, 0.9, mirror_m, "mirror"))
    light_pos = Vec3(-2.5, 4.2, 2.8)
    s.add_light(PointLight(light_pos, Vec3(70, 70, 70),
                           pulse_time=0.0, name="L"))
    # analytic lengths, independently recomputed from geometry
    reflected = _mirror_image(light_pos, m_center, m_normal)
    marker = s.primitives[-2].center
    direct_len = (light_pos - marker).length()
    bounce_len = (reflected - marker).length()
    s.expected = {
        "direct_ns": direct_len / C,
        "mirror_ns": bounce_len / C,
        "direct_m": direct_len,
        "mirror_m": bounce_len,
    }
    return s


def _mirror_image(p, point, normal):
    d = p - point
    return p - 2.0 * d.dot(normal) * normal


# --------------------------------------------------------------------------
# Scene 4: finite-depth indirect illumination
# --------------------------------------------------------------------------
def indirect_depth(resolution=(56, 42)):
    """A shielded marker is lit only by bounce light off a side wall.

    Direct line of sight from the light to the marker is blocked by an
    occluder; the earliest transport is light -> side wall -> marker, whose
    analytic two-segment length sets the expected earliest arrival.
    """
    s = _room("indirect_depth")
    w, h = resolution
    s.camera = Camera(Vec3(0, 2.5, 0), Vec3(0, 2.5, 1), Vec3(0, 1, 0),
                      fov_y=55, width=w, height=h)
    white = Material(Material.DIFFUSE, Vec3(0.9, 0.9, 0.9), "white")
    marker_pos = Vec3(1.5, 1.0, 5.0)
    s.add(Sphere(marker_pos, 0.5, white, "marker"))
    # full-height occluder wall spanning most of the room width; the only
    # unblocked light route to the marker is around the left edge
    blocker = Material(Material.DIFFUSE, Vec3(0.2, 0.2, 0.25), "blocker")
    s.add(Box(Vec3(-3.5, 0.0, 3.4), Vec3(4.5, 6.0, 3.9), blocker, "occluder"))
    s.add_light(PointLight(Vec3(1.5, 4.2, 2.6), Vec3(80, 80, 80),
                           pulse_time=0.0, name="L"))
    # analytic one-bounce route around the occluder's left edge (x=-3.5)
    light = s.lights[0].position
    edge = Vec3(-3.5, 2.5, 3.65)
    one_bounce = ((edge - light).length()
                  + (marker_pos - edge).length())
    s.expected = {
        "earliest_indirect_m": one_bounce,
        "earliest_indirect_ns": one_bounce / C,
    }
    return s


PRESETS = {
    "two_gate": two_gate,
    "path_lengths": path_lengths,
    "mirror_detour": mirror_detour,
    "indirect_depth": indirect_depth,
}
