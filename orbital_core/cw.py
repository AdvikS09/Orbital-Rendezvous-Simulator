import numpy as np

from .orbit import MU_EARTH, mean_motion


def cw_mean_motion(radius, mu=MU_EARTH):
    return mean_motion(radius, mu)


def cw_system_matrices(n):
    A = np.zeros((6, 6))
    A[0, 3] = 1.0
    A[1, 4] = 1.0
    A[2, 5] = 1.0
    A[3, 0] = 3.0 * n * n
    A[3, 4] = 2.0 * n
    A[4, 3] = -2.0 * n
    A[5, 2] = -n * n

    B = np.zeros((6, 3))
    B[3, 0] = 1.0
    B[4, 1] = 1.0
    B[5, 2] = 1.0
    return A, B


def _resolve_control(control, state, t):
    if control is None:
        return np.zeros(3)
    if callable(control):
        return np.asarray(control(state, t), dtype=float)
    return np.asarray(control, dtype=float)


def cw_derivatives(state, t, n, control=None):
    state = np.asarray(state, dtype=float)
    A, B = cw_system_matrices(n)
    u = _resolve_control(control, state, t)
    return A @ state + B @ u


def cw_state_transition(n, t):
    nt = n * t
    s, c = np.sin(nt), np.cos(nt)
    phi = np.zeros((6, 6))

    ip_idx = [0, 1, 3, 4]
    ip = np.array([
        [4.0 - 3.0 * c, 0.0, s / n, 2.0 * (1.0 - c) / n],
        [6.0 * (s - nt), 1.0, -2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * nt) / n],
        [3.0 * n * s, 0.0, c, 2.0 * s],
        [-6.0 * n * (1.0 - c), 0.0, -2.0 * s, 4.0 * c - 3.0],
    ])
    for a, ia in enumerate(ip_idx):
        for b, ib in enumerate(ip_idx):
            phi[ia, ib] = ip[a, b]

    phi[2, 2] = c
    phi[2, 5] = s / n
    phi[5, 2] = -n * s
    phi[5, 5] = c
    return phi


def propagate_cw_analytic(state0, times, n):
    state0 = np.asarray(state0, dtype=float)
    times = np.asarray(times, dtype=float)
    out = np.empty((times.size, 6), dtype=float)
    for i, t in enumerate(times):
        out[i] = cw_state_transition(n, t) @ state0
    return out


def propagate_cw(state0, times, n, controller=None):
    state0 = np.asarray(state0, dtype=float)
    times = np.asarray(times, dtype=float)
    states = np.empty((times.size, 6), dtype=float)
    controls = np.zeros((times.size, 3), dtype=float)
    states[0] = state0
    controls[0] = _resolve_control(controller, states[0], times[0])

    deriv = lambda s, t: cw_derivatives(s, t, n, controller)
    for i in range(times.size - 1):
        dt = times[i + 1] - times[i]
        k1 = deriv(states[i], times[i])
        k2 = deriv(states[i] + 0.5 * dt * k1, times[i] + 0.5 * dt)
        k3 = deriv(states[i] + 0.5 * dt * k2, times[i] + 0.5 * dt)
        k4 = deriv(states[i] + dt * k3, times[i] + dt)
        states[i + 1] = states[i] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        controls[i + 1] = _resolve_control(controller, states[i + 1], times[i + 1])
    return states, controls


__all__ = [
    "cw_mean_motion",
    "cw_system_matrices",
    "cw_derivatives",
    "cw_state_transition",
    "propagate_cw_analytic",
    "propagate_cw",
]
