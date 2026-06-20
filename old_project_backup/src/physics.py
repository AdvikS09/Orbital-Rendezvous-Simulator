"""Two-body Earth-satellite dynamics in a 2D inertial plane.

State-vector convention used throughout the platform::

    state = [x, y, vx, vy]          (SI: metres, metres/second)
    position = state[:2]
    velocity = state[2:]

The only force modelled here is point-mass Earth gravity, which makes the
problem an ideal testbed for verifying integrator accuracy via the conserved
quantities (specific orbital energy and specific angular momentum).
"""

import numpy as np

from .constants import MU_EARTH


def two_body_acceleration(position, mu=MU_EARTH):
    """Gravitational acceleration at ``position`` for a point-mass primary.

    a = -mu * r / |r|^3
    """
    position = np.asarray(position, dtype=float)
    r = np.linalg.norm(position)
    if r == 0.0:
        raise ValueError("two_body_acceleration: position has zero magnitude")
    return -mu * position / r**3


def two_body_derivatives(state, t, mu=MU_EARTH):
    """First-order ODE right-hand side, d(state)/dt = [vx, vy, ax, ay].

    The ``t`` argument is accepted for a uniform ``deriv(state, t)`` interface
    even though two-body gravity is time-invariant.
    """
    state = np.asarray(state, dtype=float)
    velocity = state[2:]
    acceleration = two_body_acceleration(state[:2], mu)
    return np.concatenate((velocity, acceleration))


def circular_speed(radius, mu=MU_EARTH):
    """Speed required for a circular orbit of the given radius: v = sqrt(mu/r)."""
    return np.sqrt(mu / radius)


def orbital_period(semi_major_axis, mu=MU_EARTH):
    """Keplerian orbital period: T = 2*pi*sqrt(a^3/mu)."""
    return 2.0 * np.pi * np.sqrt(semi_major_axis**3 / mu)


def mean_motion(semi_major_axis, mu=MU_EARTH):
    """Mean motion: n = sqrt(mu/a^3) [rad/s]."""
    return np.sqrt(mu / semi_major_axis**3)


def circular_orbit_state(radius, mu=MU_EARTH):
    """Initial state for a circular orbit of the given radius.

    The satellite is placed on the +x axis moving in the +y direction, giving a
    counter-clockwise (prograde, positive angular momentum) orbit.
    """
    speed = circular_speed(radius, mu)
    return np.array([radius, 0.0, 0.0, speed], dtype=float)


def specific_orbital_energy(state, mu=MU_EARTH):
    """Specific mechanical energy: eps = v^2/2 - mu/r [J/kg]."""
    state = np.asarray(state, dtype=float)
    r = np.linalg.norm(state[:2])
    v = np.linalg.norm(state[2:])
    return 0.5 * v * v - mu / r


def specific_angular_momentum(state):
    """z-component of r x v for the planar problem: h = x*vy - y*vx [m^2/s]."""
    x, y, vx, vy = np.asarray(state, dtype=float)
    return x * vy - y * vx


__all__ = [
    "two_body_acceleration",
    "two_body_derivatives",
    "circular_speed",
    "orbital_period",
    "mean_motion",
    "circular_orbit_state",
    "specific_orbital_energy",
    "specific_angular_momentum",
]
