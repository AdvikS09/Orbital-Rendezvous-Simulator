"""High-level simulation drivers tying together dynamics and integrators.

Every driver returns a :class:`SimulationResult` so that the analysis and
plotting layers have a uniform container to work with.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .constants import MU_EARTH
from .cw import cw_derivatives, inertial_to_lvlh, lvlh_to_inertial
from .integrators import integrate
from .physics import two_body_acceleration, two_body_derivatives


@dataclass
class SimulationResult:
    """Container for a time history of states (and optional control inputs)."""

    t: np.ndarray
    states: np.ndarray
    method: str = ""
    controls: "np.ndarray | None" = None

    @property
    def positions(self):
        return self.states[:, :2]

    @property
    def velocities(self):
        return self.states[:, 2:]


def propagate_two_body(state0, t_span, dt, mu=MU_EARTH, method="rk4"):
    """Propagate a 2D two-body orbit.

    ``t_span`` is ``(t0, t1)`` in seconds. For the symplectic Verlet/leapfrog
    method the acceleration function is supplied; otherwise the full derivative
    function is used.
    """
    t0, t1 = t_span
    if method.lower() in ("verlet", "leapfrog"):
        func = lambda pos, t: two_body_acceleration(pos, mu)
    else:
        func = lambda state, t: two_body_derivatives(state, t, mu)
    times, states = integrate(func, state0, t0, t1, dt, method=method)
    return SimulationResult(t=times, states=states, method=method)


def _eval_control(controller, state, t):
    if controller is None:
        return np.zeros(2)
    return np.asarray(controller(state, t), dtype=float)


def simulate_cw(state0, t_span, dt, n, controller=None, method="rk4"):
    """Simulate CW relative motion, optionally under closed-loop control.

    ``controller`` is any callable ``controller(state, t) -> [ax, ay]`` (e.g. a
    :class:`src.controllers.PDController`). The realised control history is
    recomputed along the trajectory and stored on the result.
    """
    t0, t1 = t_span
    func = lambda state, t: cw_derivatives(state, t, n, controller)
    times, states = integrate(func, state0, t0, t1, dt, method=method)
    controls = np.array(
        [_eval_control(controller, states[i], times[i]) for i in range(len(times))]
    )
    return SimulationResult(t=times, states=states, method=method, controls=controls)


def propagate_nonlinear_relative(
    target_state0, rel_state0, t_eval, mu=MU_EARTH, rtol=1e-12, atol=1e-12
):
    """High-fidelity relative motion from two full non-linear two-body orbits.

    The target (chief) and the deputy are each propagated as independent
    point-mass orbits with an adaptive RK45 integrator, then the deputy is
    expressed in the target's rotating LVLH frame to give the "truth" relative
    trajectory used to assess the CW linearisation.

    Parameters
    ----------
    target_state0 : array_like
        Target inertial state ``[x, y, vx, vy]`` at ``t_eval[0]``.
    rel_state0 : array_like
        Initial relative state ``[x, y, vx, vy]`` in the target LVLH frame.
    t_eval : array_like
        Times at which to report the relative state (seconds).
    """
    target_state0 = np.asarray(target_state0, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)
    deputy_state0 = lvlh_to_inertial(target_state0, rel_state0)

    fun = lambda t, s: two_body_derivatives(s, t, mu)
    span = (float(t_eval[0]), float(t_eval[-1]))

    sol_target = solve_ivp(
        fun, span, target_state0, t_eval=t_eval, rtol=rtol, atol=atol, method="RK45"
    )
    sol_deputy = solve_ivp(
        fun, span, deputy_state0, t_eval=t_eval, rtol=rtol, atol=atol, method="RK45"
    )
    if not (sol_target.success and sol_deputy.success):
        raise RuntimeError("non-linear propagation failed in solve_ivp")

    rel = np.empty((len(t_eval), 4), dtype=float)
    for k in range(len(t_eval)):
        rel[k] = inertial_to_lvlh(sol_target.y[:, k], sol_deputy.y[:, k])
    return SimulationResult(t=t_eval, states=rel, method="nonlinear")


__all__ = [
    "SimulationResult",
    "propagate_two_body",
    "simulate_cw",
    "propagate_nonlinear_relative",
]
