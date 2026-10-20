import numpy as np

from .orbit import MUEARTH, mean_motion


def cwmeanmotion(radius, mu=MUEARTH):
    return mean_motion(radius, mu)


def cwsystemmats(n):
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


def resctrl(control, state, t):
    if control is None:
        return np.zeros(3)
    if callable(control):
        return np.asarray(control(state, t), dtype=float)
    return np.asarray(control, dtype=float)


def cwderivs(state, t, n, control=None):
    state = np.asarray(state, dtype=float)
    A, B = cwsystemmats(n)
    u = resctrl(control, state, t)
    return A @ state + B @ u


def cwstatetrans(n, t):
    nt = n * t
    s, c = np.sin(nt), np.cos(nt)
    phi = np.zeros((6, 6))

    ipidx = [0, 1, 3, 4]
    ip = np.array([
        [4.0 - 3.0 * c, 0.0, s / n, 2.0 * (1.0 - c) / n],
        [6.0 * (s - nt), 1.0, -2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * nt) / n],
        [3.0 * n * s, 0.0, c, 2.0 * s],
        [-6.0 * n * (1.0 - c), 0.0, -2.0 * s, 4.0 * c - 3.0],
    ])
    for a, ia in enumerate(ipidx):
        for b, ib in enumerate(ipidx):
            phi[ia, ib] = ip[a, b]

    phi[2, 2] = c
    phi[2, 5] = s / n
    phi[5, 2] = -n * s
    phi[5, 5] = c
    return phi


def propcwanal(state0, times, n):
    state0 = np.asarray(state0, dtype=float)
    times = np.asarray(times, dtype=float)
    out = np.empty((times.size, 6), dtype=float)
    for i, t in enumerate(times):
        out[i] = cwstatetrans(n, t) @ state0
    return out


def propcw(state0, times, n, controller=None):
    state0 = np.asarray(state0, dtype=float)
    times = np.asarray(times, dtype=float)
    states = np.empty((times.size, 6), dtype=float)
    controls = np.zeros((times.size, 3), dtype=float)
    states[0] = state0
    controls[0] = resctrl(controller, states[0], times[0])

    deriv = lambda s, t: cwderivs(s, t, n, controller)
    for i in range(times.size - 1):
        dt = times[i + 1] - times[i]
        k1 = deriv(states[i], times[i])
        k2 = deriv(states[i] + 0.5 * dt * k1, times[i] + 0.5 * dt)
        k3 = deriv(states[i] + 0.5 * dt * k2, times[i] + 0.5 * dt)
        k4 = deriv(states[i] + dt * k3, times[i] + dt)
        states[i + 1] = states[i] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        controls[i + 1] = resctrl(controller, states[i + 1], times[i + 1])
    return states, controls
