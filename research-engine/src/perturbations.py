"""Orbital perturbations: J2 oblateness and atmospheric drag, with toggles.

The :class:`ForceModel` assembles the total acceleration from selectable terms,
giving the four required configurations:

    two-body only   ForceModel(use_j2=False, use_drag=False)
    J2 only         ForceModel(use_j2=True,  use_drag=False)
    drag only       ForceModel(use_j2=False, use_drag=True)
    J2 + drag       ForceModel(use_j2=True,  use_drag=True)

Convenience constructors ``ForceModel.two_body()`` etc. are provided. The model
exposes ``acceleration(state, t)`` and ``derivatives(state, t)`` so it plugs
straight into :func:`src.orbit.propagate`.
"""

from dataclasses import dataclass

import numpy as np

from .orbit import J2_EARTH, MU_EARTH, OMEGA_EARTH, R_EARTH, two_body_acceleration


def j2_acceleration(position, mu=MU_EARTH, r_earth=R_EARTH, j2=J2_EARTH):
    """J2 perturbing acceleration in ECI [m/s^2].

    Standard zonal-J2 gradient::

        a = -(3/2) J2 (mu/r^2) (Re/r)^2 * [ (1 - 5 z^2/r^2) x/r,
                                            (1 - 5 z^2/r^2) y/r,
                                            (3 - 5 z^2/r^2) z/r ]
    """
    x, y, z = np.asarray(position, dtype=float)
    r = np.sqrt(x * x + y * y + z * z)
    factor = -1.5 * j2 * mu * r_earth**2 / r**5
    zr2 = 5.0 * z * z / (r * r)
    return factor * np.array([x * (1.0 - zr2), y * (1.0 - zr2), z * (3.0 - zr2)])


def exponential_density(altitude, rho0=3.9e-12, h_ref=400_000.0, scale_height=60_000.0):
    """Exponential atmospheric density model [kg/m^3].

    ``rho = rho0 * exp(-(altitude - h_ref) / scale_height)``. Defaults give a
    representative density at 400 km (moderate solar activity). Density is
    clamped to zero below the surface guard to avoid spurious values.
    """
    if altitude < -h_ref:  # well below the surface; treat as no atmosphere
        return 0.0
    return rho0 * np.exp(-(altitude - h_ref) / scale_height)


def drag_acceleration(
    position, velocity, cd=2.2, area=1.0, mass=100.0,
    r_earth=R_EARTH, omega_earth=OMEGA_EARTH, density_func=exponential_density,
):
    """Atmospheric drag acceleration in ECI [m/s^2].

        a = -0.5 * rho * (Cd*A/m) * |v_rel| * v_rel

    where ``v_rel`` is velocity relative to the co-rotating atmosphere,
    ``v_rel = v - omega_earth x r``.
    """
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    altitude = np.linalg.norm(position) - r_earth
    rho = density_func(altitude)
    if rho <= 0.0:
        return np.zeros(3)
    omega_vec = np.array([0.0, 0.0, omega_earth])
    v_rel = velocity - np.cross(omega_vec, position)
    speed = np.linalg.norm(v_rel)
    ballistic = cd * area / mass  # Cd*A/m [m^2/kg]
    return -0.5 * rho * ballistic * speed * v_rel


@dataclass
class ForceModel:
    """Selectable force model for orbital propagation.

    Attributes
    ----------
    use_j2, use_drag : bool       enable each perturbation
    mu, r_earth, j2 : float       gravity / oblateness parameters
    cd, area, mass : float        drag ballistic parameters (Cd, A [m^2], m [kg])
    omega_earth : float           Earth rotation rate for drag co-rotation
    """

    use_j2: bool = False
    use_drag: bool = False
    mu: float = MU_EARTH
    r_earth: float = R_EARTH
    j2: float = J2_EARTH
    cd: float = 2.2
    area: float = 1.0
    mass: float = 100.0
    omega_earth: float = OMEGA_EARTH

    # --- Convenience constructors for the four required toggles ---
    @classmethod
    def two_body(cls, **kw):
        return cls(use_j2=False, use_drag=False, **kw)

    @classmethod
    def j2_only(cls, **kw):
        return cls(use_j2=True, use_drag=False, **kw)

    @classmethod
    def drag_only(cls, **kw):
        return cls(use_j2=False, use_drag=True, **kw)

    @classmethod
    def j2_drag(cls, **kw):
        return cls(use_j2=True, use_drag=True, **kw)

    @property
    def label(self):
        if self.use_j2 and self.use_drag:
            return "two-body + J2 + drag"
        if self.use_j2:
            return "two-body + J2"
        if self.use_drag:
            return "two-body + drag"
        return "two-body"

    def acceleration(self, state, t=0.0):
        """Total ECI acceleration for the enabled terms."""
        state = np.asarray(state, dtype=float)
        pos, vel = state[:3], state[3:]
        acc = two_body_acceleration(pos, self.mu)
        if self.use_j2:
            acc = acc + j2_acceleration(pos, self.mu, self.r_earth, self.j2)
        if self.use_drag:
            acc = acc + drag_acceleration(
                pos, vel, self.cd, self.area, self.mass, self.r_earth, self.omega_earth
            )
        return acc

    def derivatives(self, state, t=0.0):
        """ODE RHS d(state)/dt = [v, a] for :func:`src.orbit.propagate`."""
        state = np.asarray(state, dtype=float)
        return np.concatenate((state[3:], self.acceleration(state, t)))


__all__ = [
    "j2_acceleration",
    "exponential_density",
    "drag_acceleration",
    "ForceModel",
]
