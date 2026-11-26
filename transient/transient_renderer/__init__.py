"""Transient light-transport renderer (pure standard library)."""
from .render import render, pixel_profile, evaluate_gates, readout_decision
from .scenes import PRESETS
from .kernel import Kernel

__all__ = ["render", "pixel_profile", "evaluate_gates", "readout_decision",
           "PRESETS", "Kernel"]
