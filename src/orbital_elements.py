"""Conversion between 2D Cartesian state and classical orbital elements.

Only the planar (in-plane) elements are meaningful for the 2D model:

    a      semi-major axis           [m]
    e      eccentricity              [-]
    omega  argument of periapsis     [rad]  (angle of periapsis from +x axis)
    nu     true anomaly              [rad]

Inclination and right ascension are out of scope for the planar problem and are
deliberately omitted. These routines are primarily used to validate orbits and
to provide human-readable diagnostics.
"""

import numpy as np

from .constants import MU_EARTH
from .physics import orbital_period

_TOL = 1e-12


def state_to_elements(state, mu=MU_EARTH):
    """Convert a planar Cartesian state ``[x, y, vx, vy]`` to orbital elements.

    Returns a dict with keys ``a, e, omega, nu, h, energy, period``.
    """
    state = np.asarray(state, dtype=float)
    r_vec = state[:2]
    v_vec = state[2:]
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    energy = 0.5 * v * v - mu / r
    # Scalar specific angular momentum (z component of r x v).
    h = r_vec[0] * v_vec[1] - r_vec[1] * v_vec[0]

    # Eccentricity vector (planar form).
    e_vec = ((v * v - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = np.linalg.norm(e_vec)

    # Semi-major axis from the vis-viva / energy relation.
    a = -mu / (2.0 * energy) if abs(energy) > _TOL else np.inf

    if e > _TOL:
        omega = np.arctan2(e_vec[1], e_vec[0])
        cos_nu = np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0)
        nu = np.arccos(cos_nu)
        if np.dot(r_vec, v_vec) < 0.0:
            nu = 2.0 * np.pi - nu
    else:
        # Circular orbit: periapsis undefined, measure position angle directly.
        omega = 0.0
        nu = np.arctan2(r_vec[1], r_vec[0])

    period = orbital_period(a, mu) if np.isfinite(a) and a > 0.0 else np.inf

    return {
        "a": a,
        "e": e,
        "omega": omega,
        "nu": nu,
        "h": h,
        "energy": energy,
        "period": period,
    }


def elements_to_state(a, e, nu, omega=0.0, mu=MU_EARTH):
    """Convert planar orbital elements to a Cartesian state ``[x, y, vx, vy]``."""
    p = a * (1.0 - e * e)  # semi-latus rectum
    r = p / (1.0 + e * np.cos(nu))

    # Perifocal frame (periapsis along local +x).
    r_pf = np.array([r * np.cos(nu), r * np.sin(nu)])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu)])

    c, s = np.cos(omega), np.sin(omega)
    rot = np.array([[c, -s], [s, c]])  # rotate perifocal -> inertial by omega

    r_vec = rot @ r_pf
    v_vec = rot @ v_pf
    return np.concatenate((r_vec, v_vec))


__all__ = ["state_to_elements", "elements_to_state"]
