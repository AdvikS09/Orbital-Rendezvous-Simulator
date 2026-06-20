"""LVLH (Local-Vertical/Local-Horizontal) relative-motion frame transforms.

The LVLH frame is centred on a *target* (chief) spacecraft and defined from its
inertial state:

    x  radial      (R-bar)  = r_hat            (away from Earth)
    z  cross-track (H-bar)  = h_hat = (r x v)  (orbit normal)
    y  along-track (V-bar)  = z_hat x x_hat    (completes the right-handed triad)

For a circular orbit y_hat points along the velocity direction. Relative states
are 6-vectors ``[x, y, z, vx, vy, vz]`` in this frame.

The frame rotates with angular velocity ``omega = (r x v) / r^2`` (along h_hat).
The velocity transform therefore carries the transport term ``omega x rho``:

    rho      = R (r_dep - r_tgt)
    rho_dot  = R (v_dep - v_tgt) - omega_body x rho

with ``R`` the inertial->LVLH rotation (its rows are x_hat, y_hat, z_hat).
"""

import numpy as np

from .orbit import MU_EARTH  # noqa: F401  (re-exported convenience)


def lvlh_rotation_matrix(target_state):
    """Rotation matrix R mapping inertial vectors to LVLH coordinates.

    Rows are the LVLH basis vectors (x=radial, y=along-track, z=cross-track)
    expressed in the inertial frame, so ``v_lvlh = R @ v_inertial``.
    """
    target_state = np.asarray(target_state, dtype=float)
    r_vec, v_vec = target_state[:3], target_state[3:]
    x_hat = r_vec / np.linalg.norm(r_vec)
    h_vec = np.cross(r_vec, v_vec)
    z_hat = h_vec / np.linalg.norm(h_vec)
    y_hat = np.cross(z_hat, x_hat)
    return np.vstack((x_hat, y_hat, z_hat))


def lvlh_angular_velocity(target_state):
    """LVLH frame angular velocity in the inertial frame: omega = (r x v)/r^2."""
    target_state = np.asarray(target_state, dtype=float)
    r_vec, v_vec = target_state[:3], target_state[3:]
    r2 = np.dot(r_vec, r_vec)
    return np.cross(r_vec, v_vec) / r2


def inertial_to_lvlh(target_state, deputy_state):
    """Express a deputy ECI state in the target's rotating LVLH frame.

    Returns the relative state ``[x, y, z, vx, vy, vz]`` (radial, along-track,
    cross-track).
    """
    target_state = np.asarray(target_state, dtype=float)
    deputy_state = np.asarray(deputy_state, dtype=float)

    rot = lvlh_rotation_matrix(target_state)
    omega_body = rot @ lvlh_angular_velocity(target_state)

    dr = deputy_state[:3] - target_state[:3]
    dv = deputy_state[3:] - target_state[3:]

    rho = rot @ dr
    rho_dot = rot @ dv - np.cross(omega_body, rho)
    return np.concatenate((rho, rho_dot))


def lvlh_to_inertial(target_state, rel_state):
    """Inverse of :func:`inertial_to_lvlh`: LVLH relative state -> deputy ECI."""
    target_state = np.asarray(target_state, dtype=float)
    rel_state = np.asarray(rel_state, dtype=float)

    rot = lvlh_rotation_matrix(target_state)
    omega_body = rot @ lvlh_angular_velocity(target_state)

    rho = rel_state[:3]
    rho_dot = rel_state[3:]

    dr = rot.T @ rho
    dv = rot.T @ (rho_dot + np.cross(omega_body, rho))
    return np.concatenate((target_state[:3] + dr, target_state[3:] + dv))


def relative_position(target_state, deputy_state):
    """Relative position of the deputy in the target LVLH frame [m]."""
    return inertial_to_lvlh(target_state, deputy_state)[:3]


def relative_velocity(target_state, deputy_state):
    """Relative velocity of the deputy in the target LVLH frame [m/s]."""
    return inertial_to_lvlh(target_state, deputy_state)[3:]


__all__ = [
    "lvlh_rotation_matrix",
    "lvlh_angular_velocity",
    "inertial_to_lvlh",
    "lvlh_to_inertial",
    "relative_position",
    "relative_velocity",
]
