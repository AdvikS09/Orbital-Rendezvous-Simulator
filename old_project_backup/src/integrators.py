"""Numerical integrators for the orbital / relative-motion ODEs.

Two families of fixed-step integrators are provided:

* First-order steppers operate on a derivative function ``deriv(state, t)`` that
  returns ``d(state)/dt`` for ``state = [x, y, vx, vy]``:
      - ``euler_step``   : explicit (forward) Euler  -- O(dt), included only as a
                           deliberately poor baseline for comparison.
      - ``rk4_step``     : classic 4th-order Runge-Kutta -- O(dt^4), accurate but
                           not symplectic (tiny secular energy drift).

* The Velocity-Verlet (a.k.a. leapfrog) stepper operates on an acceleration
  function ``accel(position, t)`` of position only. It is symplectic, so for
  conservative systems (e.g. two-body gravity) the energy error stays bounded
  rather than drifting:
      - ``velocity_verlet_step``

``integrate`` is the common driver. For ``method`` in {"euler", "rk4"} the
``func`` argument is the derivative function; for {"verlet", "leapfrog"} it is
the acceleration function.
"""

import numpy as np


def euler_step(deriv, state, t, dt):
    """One explicit-Euler step. ``deriv(state, t) -> d(state)/dt``."""
    state = np.asarray(state, dtype=float)
    return state + dt * np.asarray(deriv(state, t), dtype=float)


def rk4_step(deriv, state, t, dt):
    """One classic RK4 step. ``deriv(state, t) -> d(state)/dt``."""
    state = np.asarray(state, dtype=float)
    k1 = np.asarray(deriv(state, t), dtype=float)
    k2 = np.asarray(deriv(state + 0.5 * dt * k1, t + 0.5 * dt), dtype=float)
    k3 = np.asarray(deriv(state + 0.5 * dt * k2, t + 0.5 * dt), dtype=float)
    k4 = np.asarray(deriv(state + dt * k3, t + dt), dtype=float)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def velocity_verlet_step(accel, state, t, dt):
    """One velocity-Verlet step for ``state = [position..., velocity...]``.

    ``accel(position, t) -> acceleration`` must depend on position (and possibly
    time) only -- valid for conservative forces such as two-body gravity.
    """
    state = np.asarray(state, dtype=float)
    half = state.size // 2
    pos = state[:half]
    vel = state[half:]
    a0 = np.asarray(accel(pos, t), dtype=float)
    pos_new = pos + vel * dt + 0.5 * a0 * dt * dt
    a1 = np.asarray(accel(pos_new, t + dt), dtype=float)
    vel_new = vel + 0.5 * (a0 + a1) * dt
    return np.concatenate((pos_new, vel_new))


_STEPPERS = {
    "euler": euler_step,
    "rk4": rk4_step,
    "verlet": velocity_verlet_step,
    "leapfrog": velocity_verlet_step,
}


def available_methods():
    """Return the sorted list of supported integrator names."""
    return sorted(_STEPPERS)


def integrate(func, state0, t0, t1, dt, method="rk4"):
    """Integrate from ``t0`` to ``t1`` with a fixed step ``dt``.

    Parameters
    ----------
    func : callable
        ``deriv(state, t)`` for euler/rk4, or ``accel(position, t)`` for
        verlet/leapfrog.
    state0 : array_like
        Initial state ``[x, y, vx, vy]``.
    t0, t1, dt : float
        Start time, end time and step (SI seconds). ``dt`` should divide
        ``t1 - t0``; the number of steps is rounded to the nearest integer.
    method : str
        One of :func:`available_methods`.

    Returns
    -------
    times : ndarray, shape (N+1,)
    states : ndarray, shape (N+1, len(state0))
    """
    method = method.lower()
    if method not in _STEPPERS:
        raise ValueError(
            f"unknown integrator {method!r}; choose from {available_methods()}"
        )
    step = _STEPPERS[method]

    state0 = np.asarray(state0, dtype=float)
    n_steps = int(round((t1 - t0) / dt))
    if n_steps < 1:
        raise ValueError("integration span is shorter than one step")

    times = t0 + dt * np.arange(n_steps + 1)
    states = np.empty((n_steps + 1, state0.size), dtype=float)
    states[0] = state0
    for i in range(n_steps):
        states[i + 1] = step(func, states[i], times[i], dt)
    return times, states


__all__ = [
    "euler_step",
    "rk4_step",
    "velocity_verlet_step",
    "available_methods",
    "integrate",
]
