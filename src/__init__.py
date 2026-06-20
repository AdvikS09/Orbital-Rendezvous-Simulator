"""Orbital rendezvous research engine.

CW-based spacecraft rendezvous with a PD controller, validated against a
higher-fidelity (J2 + drag) orbital model. SI units throughout; 3D LVLH
relative-motion state ``[x, y, z, vx, vy, vz]``.

Modules
-------
orbit          two-body dynamics, RK4 propagation, orbital elements, energy/period
perturbations  J2 and atmospheric drag with selectable force-model toggles
lvlh           inertial <-> LVLH frame transforms, relative position/velocity
cw             Clohessy-Wiltshire dynamics, state-transition matrix, propagation
controller     configurable PD rendezvous controller
validation     conservation checks, error metrics, gain sweep, CW-vs-high-fidelity
plotting       headless matplotlib helpers
"""

__version__ = "1.0.0"

from . import controller, cw, lvlh, orbit, perturbations, plotting, validation

__all__ = [
    "orbit",
    "perturbations",
    "lvlh",
    "cw",
    "controller",
    "validation",
    "plotting",
    "__version__",
]
