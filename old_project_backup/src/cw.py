"""Clohessy-Wiltshire / Hill linearised relative motion (planar).

The chaser's motion is expressed in the target's Local-Vertical/Local-Horizontal
(LVLH) frame, centred on a target moving on a circular reference orbit:

    x  : radial      (R-bar, positive away from Earth)
    y  : along-track (V-bar, positive along the velocity direction)
    state = [x, y, vx, vy]            (SI)

For a circular reference orbit with mean motion ``n`` the linearised equations
of relative motion (with control accelerations ``[ax, ay]``) are::

    x'' - 2 n y' - 3 n^2 x = ax
    y'' + 2 n x'           = ay

This module also provides the closed-form state-transition matrix (for
validating the numerical integration) and the exact non-linear LVLH frame
transforms used to compare CW against full two-body propagation.
"""

import numpy as np


def mean_motion_circular(radius, mu):
    """Mean motion of a circular reference orbit: n = sqrt(mu / r^3) [rad/s]."""
    return np.sqrt(mu / radius**3)


def cw_system_matrices(n):
    """State-space matrices (A, B) such that ``state' = A @ state + B @ u``.

    state = [x, y, vx, vy], u = [ax, ay].
    """
    A = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [3.0 * n * n, 0.0, 0.0, 2.0 * n],
            [0.0, 0.0, -2.0 * n, 0.0],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    return A, B


def _resolve_control(control, state, t):
    """Normalise a control specification to a length-2 acceleration vector."""
    if control is None:
        return np.zeros(2)
    if callable(control):
        return np.asarray(control(state, t), dtype=float)
    return np.asarray(control, dtype=float)


def cw_derivatives(state, t, n, control=None):
    """First-order RHS of the CW equations: ``d(state)/dt``.

    ``control`` may be ``None``, a fixed length-2 acceleration, or a callable
    ``control(state, t) -> [ax, ay]`` (e.g. a feedback controller).
    """
    state = np.asarray(state, dtype=float)
    A, B = cw_system_matrices(n)
    u = _resolve_control(control, state, t)
    return A @ state + B @ u


def cw_state_transition(n, t):
    """Closed-form CW state-transition matrix Phi(t) for unforced motion.

    ``state(t) = Phi(t) @ state(0)`` with state ordered [x, y, vx, vy].
    """
    nt = n * t
    s = np.sin(nt)
    c = np.cos(nt)
    return np.array(
        [
            [4.0 - 3.0 * c, 0.0, s / n, 2.0 * (1.0 - c) / n],
            [6.0 * (s - nt), 1.0, -2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * nt) / n],
            [3.0 * n * s, 0.0, c, 2.0 * s],
            [-6.0 * n * (1.0 - c), 0.0, -2.0 * s, 4.0 * c - 3.0],
        ],
        dtype=float,
    )


def propagate_cw_analytic(state0, times, n):
    """Propagate the unforced CW state analytically at each time in ``times``."""
    state0 = np.asarray(state0, dtype=float)
    out = np.empty((len(times), 4), dtype=float)
    for i, t in enumerate(times):
        out[i] = cw_state_transition(n, t) @ state0
    return out


# --- Exact non-linear LVLH frame transforms -------------------------------
# Used to convert between an inertial two-body state and the target-centred
# rotating LVLH frame, so CW can be compared against high-fidelity truth.


def lvlh_basis(target_state):
    """Return (radial_hat, alongtrack_hat, omega, r) for a target inertial state.

    ``omega`` is the LVLH frame angular rate (theta-dot) about +z [rad/s];
    ``r`` is the target's orbital radius [m].
    """
    target_state = np.asarray(target_state, dtype=float)
    r_vec = target_state[:2]
    v_vec = target_state[2:]
    r = np.linalg.norm(r_vec)
    radial_hat = r_vec / r
    # +90 deg rotation of the radial direction -> along-track for a prograde
    # (counter-clockwise) orbit.
    alongtrack_hat = np.array([-radial_hat[1], radial_hat[0]])
    omega = (r_vec[0] * v_vec[1] - r_vec[1] * v_vec[0]) / (r * r)
    return radial_hat, alongtrack_hat, omega, r


def inertial_to_lvlh(target_state, deputy_state):
    """Express ``deputy_state`` in the target's rotating LVLH frame.

    Returns the relative state ``[x, y, vx, vy]`` (radial, along-track).
    """
    radial_hat, alongtrack_hat, omega, _ = lvlh_basis(target_state)
    rot = np.array([radial_hat, alongtrack_hat])  # inertial -> LVLH

    dr = np.asarray(deputy_state, float)[:2] - np.asarray(target_state, float)[:2]
    dv = np.asarray(deputy_state, float)[2:] - np.asarray(target_state, float)[2:]

    p = rot @ dr
    # Rotating-frame velocity: v_rot = R @ dv - omega x p   (omega along +z).
    omega_cross_p = np.array([-omega * p[1], omega * p[0]])
    v = rot @ dv - omega_cross_p
    return np.array([p[0], p[1], v[0], v[1]])


def lvlh_to_inertial(target_state, rel_state):
    """Inverse of :func:`inertial_to_lvlh`: LVLH relative state -> inertial state."""
    radial_hat, alongtrack_hat, omega, _ = lvlh_basis(target_state)
    rot = np.array([radial_hat, alongtrack_hat])  # inertial -> LVLH
    rot_t = rot.T  # LVLH -> inertial

    rel_state = np.asarray(rel_state, dtype=float)
    p = rel_state[:2]
    v = rel_state[2:]

    dr = rot_t @ p
    omega_cross_p = np.array([-omega * p[1], omega * p[0]])
    dv = rot_t @ (v + omega_cross_p)

    target_state = np.asarray(target_state, dtype=float)
    return np.concatenate((target_state[:2] + dr, target_state[2:] + dv))


__all__ = [
    "mean_motion_circular",
    "cw_system_matrices",
    "cw_derivatives",
    "cw_state_transition",
    "propagate_cw_analytic",
    "lvlh_basis",
    "inertial_to_lvlh",
    "lvlh_to_inertial",
]
