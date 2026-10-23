import numpy as np

from .orbit import MU_EARTH  # noqa: F401  (re-exported convenience)


def lvlh_rotation_matrix(target_state):
    target_state = np.asarray(target_state, dtype=float)
    r_vec, v_vec = target_state[:3], target_state[3:]
    x_hat = r_vec / np.linalg.norm(r_vec)
    h_vec = np.cross(r_vec, v_vec)
    z_hat = h_vec / np.linalg.norm(h_vec)
    y_hat = np.cross(z_hat, x_hat)
    return np.vstack((x_hat, y_hat, z_hat))


def lvlh_angular_velocity(target_state):
    target_state = np.asarray(target_state, dtype=float)
    r_vec, v_vec = target_state[:3], target_state[3:]
    r2 = np.dot(r_vec, r_vec)
    return np.cross(r_vec, v_vec) / r2


def inertial_to_lvlh(target_state, deputy_state):
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
    return inertial_to_lvlh(target_state, deputy_state)[:3]


def relative_velocity(target_state, deputy_state):
    return inertial_to_lvlh(target_state, deputy_state)[3:]


__all__ = [
    "lvlh_rotation_matrix",
    "lvlh_angular_velocity",
    "inertial_to_lvlh",
    "lvlh_to_inertial",
    "relative_position",
    "relative_velocity",
]
