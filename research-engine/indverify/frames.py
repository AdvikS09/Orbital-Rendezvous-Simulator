"""Target orbit and LVLH (Hill) frame (SPEC sections 1 and 3).

Vector arguments use the component-first layout: shape (3,) for a single
vector or (3, m) for m vectors.
"""

import math

import numpy as np

from .constants import DEFAULT_ALTITUDE, INCLINATION, MU, R_E


class TargetOrbit:
    """Circular two-body target (chief) orbit, SPEC section 1."""

    def __init__(self, altitude=DEFAULT_ALTITUDE):
        self.altitude = float(altitude)
        self.R = R_E + self.altitude
        self.n = math.sqrt(MU / self.R ** 3)
        self.period = 2.0 * math.pi / self.n

    def initial_state(self):
        """r_t(0) = (R, 0, 0), v_t(0) = n R (0, cos i, sin i)."""
        ci, si = math.cos(INCLINATION), math.sin(INCLINATION)
        r = np.array([self.R, 0.0, 0.0])
        v = self.n * self.R * np.array([0.0, ci, si])
        return r, v

    def state(self, t):
        """Closed-form circular target state at time(s) t (used by tests)."""
        th = self.n * np.asarray(t, dtype=float)
        ci, si = math.cos(INCLINATION), math.sin(INCLINATION)
        c, s = np.cos(th), np.sin(th)
        r = self.R * np.array([c, s * ci, s * si])
        v = self.n * self.R * np.array([-s, c * ci, c * si])
        return r, v


def lvlh_axes(r, v):
    """Unit vectors x^, y^, z^ of the LVLH frame and the frame rate w.

    x^ = r/|r|, z^ = (r x v)/|r x v|, y^ = z^ x x^, w = |r x v| / |r|^2
    (frame angular velocity is (0, 0, w) in LVLH components).
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    rn = np.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    xh = r / rn
    h = _cross(r, v)
    hn = np.sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2])
    zh = h / hn
    yh = _cross(zh, xh)
    w = hn / (rn * rn)
    return xh, yh, zh, w


def _cross(a, b):
    """Cross product of component-first arrays (3,) or (3, m)."""
    return np.stack([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def dcm_eci_to_lvlh(r, v):
    """C (rows x^T, y^T, z^T) mapping ECI components to LVLH, single state."""
    xh, yh, zh, w = lvlh_axes(r, v)
    return np.vstack([xh, yh, zh]), float(w)


def inertial_to_rel(r_t, v_t, r_c, v_c):
    """rho = C (r_c - r_t), rhodot = C (v_c - v_t) - w x rho  (SPEC section 3)."""
    xh, yh, zh, w = lvlh_axes(r_t, v_t)
    dr = np.asarray(r_c, dtype=float) - np.asarray(r_t, dtype=float)
    dv = np.asarray(v_c, dtype=float) - np.asarray(v_t, dtype=float)
    rho = np.stack([_dot(xh, dr), _dot(yh, dr), _dot(zh, dr)])
    cdv = np.stack([_dot(xh, dv), _dot(yh, dv), _dot(zh, dv)])
    # w x rho with w = (0, 0, w):  (-w rho_y, w rho_x, 0)
    rhodot = np.stack([cdv[0] + w * rho[1],
                       cdv[1] - w * rho[0],
                       cdv[2]])
    return rho, rhodot


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def rel_to_inertial(r_t, v_t, rho, rhodot):
    """r_c = r_t + C^T rho, v_c = v_t + C^T (rhodot + w x rho)  (SPEC section 3)."""
    r_t = np.asarray(r_t, dtype=float)
    v_t = np.asarray(v_t, dtype=float)
    rho = np.asarray(rho, dtype=float)
    rhodot = np.asarray(rhodot, dtype=float)
    xh, yh, zh, w = lvlh_axes(r_t, v_t)
    vrel = np.stack([rhodot[0] - w * rho[1],
                     rhodot[1] + w * rho[0],
                     rhodot[2]])
    r_c = r_t + xh * rho[0] + yh * rho[1] + zh * rho[2]
    v_c = v_t + xh * vrel[0] + yh * vrel[1] + zh * vrel[2]
    return r_c, v_c
