"""Orbital Stability Platform -- Phase 1: orbital rendezvous.

A small, modular toolkit for 2D orbital dynamics and Clohessy-Wiltshire
relative-motion rendezvous, built on numpy / scipy / matplotlib / pandas.

Layout
------
constants          shared SI constants and default LEO geometry
physics            2D two-body dynamics, energy / angular momentum, period
orbital_elements   Cartesian <-> classical (planar) orbital elements
integrators        Euler, RK4, velocity-Verlet steppers and the integrate driver
cw                 Clohessy-Wiltshire / Hill dynamics, STM and LVLH transforms
controllers        PD rendezvous controller and APF obstacle avoidance
simulation         high-level propagation drivers + non-linear truth model
analysis           conservation tracking, error metrics, validation tables
plotting           headless matplotlib helpers
"""

__version__ = "0.1.0"

from . import (
    analysis,
    constants,
    controllers,
    cw,
    integrators,
    orbital_elements,
    physics,
    plotting,
    simulation,
)

__all__ = [
    "analysis",
    "constants",
    "controllers",
    "cw",
    "integrators",
    "orbital_elements",
    "physics",
    "plotting",
    "simulation",
    "__version__",
]
