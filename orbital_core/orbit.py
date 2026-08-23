"""Core orbital mechanics: constants, two-body dynamics, RK4, elements, energy.

All quantities are SI (metres, seconds, kilograms, radians). State vectors are
3D inertial (Earth-Centred Inertial, ECI)::

    state = [x, y, z, vx, vy, vz]      position = state[:3], velocity = state[3:]

This module is the foundation the rest of the engine imports from (constants,
the RK4 propagator, and the orbital-element conversions).
"""

import numpy as np

# --- Physical constants and Earth parameters ------------------------------
G = 6.67430e-11               # gravitational constant [m^3 kg^-1 s^-2]
MU_EARTH = 3.986004418e14     # Earth standard gravitational parameter [m^3/s^2]
M_EARTH = MU_EARTH / G        # Earth mass [kg]
R_EARTH = 6_378_137.0         # Earth equatorial radius (WGS-84) [m]
J2_EARTH = 1.08262668e-3      # Earth J2 oblateness coefficient [-]
OMEGA_EARTH = 7.2921159e-5    # Earth rotation rate [rad/s] (for drag co-rotation)

DEFAULT_ALTITUDE = 400_000.0                  # default LEO altitude [m]
DEFAULT_RADIUS = R_EARTH + DEFAULT_ALTITUDE   # default circular orbit radius [m]


# --- Two-body dynamics ----------------------------------------------------
def two_body_acceleration(position, mu=MU_EARTH):
    """Point-mass gravitational acceleration: a = -mu r / |r|^3."""
    position = np.asarray(position, dtype=float)
    r = np.linalg.norm(position)
    if r == 0.0:
        raise ValueError("two_body_acceleration: zero position vector")
    return -mu * position / r**3


def two_body_derivatives(state, t, mu=MU_EARTH):
    """ODE RHS d(state)/dt = [v, a] for the unperturbed two-body problem."""
    state = np.asarray(state, dtype=float)
    return np.concatenate((state[3:], two_body_acceleration(state[:3], mu)))


# --- RK4 integration ------------------------------------------------------
def rk4_step(deriv, state, t, dt):
    """One classic 4th-order Runge-Kutta step. ``deriv(state, t) -> dstate/dt``."""
    state = np.asarray(state, dtype=float)
    k1 = np.asarray(deriv(state, t), dtype=float)
    k2 = np.asarray(deriv(state + 0.5 * dt * k1, t + 0.5 * dt), dtype=float)
    k3 = np.asarray(deriv(state + 0.5 * dt * k2, t + 0.5 * dt), dtype=float)
    k4 = np.asarray(deriv(state + dt * k3, t + dt), dtype=float)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate(deriv, state0, t_grid):
    """Propagate ``state0`` over ``t_grid`` with fixed-step RK4 between samples.

    Parameters
    ----------
    deriv : callable ``deriv(state, t) -> dstate/dt``
    state0 : array_like, the initial state
    t_grid : array_like of monotonically increasing times [s]

    Returns
    -------
    states : ndarray, shape (len(t_grid), len(state0))
    """
    t_grid = np.asarray(t_grid, dtype=float)
    state0 = np.asarray(state0, dtype=float)
    states = np.empty((t_grid.size, state0.size), dtype=float)
    states[0] = state0
    for i in range(t_grid.size - 1):
        dt = t_grid[i + 1] - t_grid[i]
        states[i + 1] = rk4_step(deriv, states[i], t_grid[i], dt)
    return states


# --- Circular-orbit setup and Keplerian helpers ---------------------------
def circular_speed(radius, mu=MU_EARTH):
    """Speed for a circular orbit of given radius: v = sqrt(mu/r)."""
    return np.sqrt(mu / radius)


def orbital_period(semi_major_axis, mu=MU_EARTH):
    """Keplerian period: T = 2*pi*sqrt(a^3/mu)."""
    return 2.0 * np.pi * np.sqrt(semi_major_axis**3 / mu)


def mean_motion(semi_major_axis, mu=MU_EARTH):
    """Mean motion: n = sqrt(mu/a^3) [rad/s]."""
    return np.sqrt(mu / semi_major_axis**3)


def circular_orbit_state(radius, mu=MU_EARTH, inclination=0.0):
    """Initial ECI state for a circular orbit at the ascending node on +x.

    The orbit plane is inclined by ``inclination`` about the x-axis, giving a
    prograde orbit. ``inclination = 0`` lies in the x-y plane.
    """
    v = circular_speed(radius, mu)
    return np.array(
        [radius, 0.0, 0.0, 0.0, v * np.cos(inclination), v * np.sin(inclination)],
        dtype=float,
    )


# --- Conserved quantities -------------------------------------------------
def specific_energy(state, mu=MU_EARTH):
    """Specific orbital energy: eps = v^2/2 - mu/r [J/kg]."""
    state = np.asarray(state, dtype=float)
    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:])
    return 0.5 * v * v - mu / r


def angular_momentum_vector(state):
    """Specific angular momentum vector h = r x v [m^2/s]."""
    state = np.asarray(state, dtype=float)
    return np.cross(state[:3], state[3:])


def specific_angular_momentum(state):
    """Magnitude of the specific angular momentum [m^2/s]."""
    return np.linalg.norm(angular_momentum_vector(state))


# --- Orbital-element conversions (3D classical elements) ------------------
_TOL = 1e-11


def rv_to_elements(state, mu=MU_EARTH):
    """Convert an ECI state to classical orbital elements.

    Returns a dict with ``a, e, i, raan, argp, nu`` (radians) plus ``h``,
    ``energy`` and ``period``. Circular / equatorial singularities fall back to
    zero for the undefined angle.
    """
    state = np.asarray(state, dtype=float)
    r_vec, v_vec = state[:3], state[3:]
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    node_vec = np.cross([0.0, 0.0, 1.0], h_vec)  # points toward ascending node
    node = np.linalg.norm(node_vec)

    e_vec = ((v * v - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = np.linalg.norm(e_vec)

    energy = 0.5 * v * v - mu / r
    a = -mu / (2.0 * energy) if abs(energy) > _TOL else np.inf

    inc = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0)) if h > _TOL else 0.0

    if node > _TOL:
        raan = np.arccos(np.clip(node_vec[0] / node, -1.0, 1.0))
        if node_vec[1] < 0.0:
            raan = 2.0 * np.pi - raan
    else:
        raan = 0.0

    if node > _TOL and e > _TOL:
        argp = np.arccos(np.clip(np.dot(node_vec, e_vec) / (node * e), -1.0, 1.0))
        if e_vec[2] < 0.0:
            argp = 2.0 * np.pi - argp
    else:
        argp = 0.0

    if e > _TOL:
        nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0.0:
            nu = 2.0 * np.pi - nu
    else:
        # Circular orbit: use argument of latitude measured from the node (or +x).
        ref = node_vec if node > _TOL else np.array([1.0, 0.0, 0.0])
        ref_n = np.linalg.norm(ref)
        nu = np.arccos(np.clip(np.dot(ref, r_vec) / (ref_n * r), -1.0, 1.0))
        if r_vec[2] < 0.0:
            nu = 2.0 * np.pi - nu

    period = orbital_period(a, mu) if np.isfinite(a) and a > 0.0 else np.inf
    return {
        "a": a, "e": e, "i": inc, "raan": raan, "argp": argp, "nu": nu,
        "h": h, "energy": energy, "period": period,
    }


def elements_to_rv(a, e, i, raan, argp, nu, mu=MU_EARTH):
    """Convert classical orbital elements (radians) to an ECI state."""
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * np.cos(nu))

    # Perifocal (PQW) position and velocity.
    r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)

    # PQW -> ECI rotation matrix (3-1-3 sequence).
    rot = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])
    return np.concatenate((rot @ r_pf, rot @ v_pf))


def measure_period(t, states):
    """Estimate the orbital period from the swept in-plane angle.

    Projects the position onto the initial orbit plane and interpolates the time
    at which a full 2*pi revolution is completed. Returns ``np.nan`` if less than
    one revolution is present.
    """
    t = np.asarray(t, dtype=float)
    pos = np.asarray(states)[:, :3]
    r0 = pos[0]
    h0 = np.cross(r0, np.asarray(states)[0, 3:])
    e_x = r0 / np.linalg.norm(r0)
    e_z = h0 / np.linalg.norm(h0)
    e_y = np.cross(e_z, e_x)  # in-plane, 90 deg ahead of e_x

    proj_x = pos @ e_x
    proj_y = pos @ e_y
    theta = np.unwrap(np.arctan2(proj_y, proj_x))
    if theta[-1] < theta[0]:
        theta = -theta
    target = theta[0] + 2.0 * np.pi
    if theta[-1] < target:
        return np.nan
    return float(np.interp(target, theta, t))


__all__ = [
    "G", "MU_EARTH", "M_EARTH", "R_EARTH", "J2_EARTH", "OMEGA_EARTH",
    "DEFAULT_ALTITUDE", "DEFAULT_RADIUS",
    "two_body_acceleration", "two_body_derivatives", "rk4_step", "propagate",
    "circular_speed", "orbital_period", "mean_motion", "circular_orbit_state",
    "specific_energy", "angular_momentum_vector", "specific_angular_momentum",
    "rv_to_elements", "elements_to_rv", "measure_period",
]
