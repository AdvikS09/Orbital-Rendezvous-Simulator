from dataclasses import dataclass

import numpy as np

from .orbit import J2_EARTH, MU_EARTH, OMEGA_EARTH, R_EARTH, two_body_acceleration


def j2_acceleration(position, mu=MU_EARTH, r_earth=R_EARTH, j2=J2_EARTH):
    x, y, z = np.asarray(position, dtype=float)
    r = np.sqrt(x * x + y * y + z * z)
    factor = -1.5 * j2 * mu * r_earth**2 / r**5
    zr2 = 5.0 * z * z / (r * r)
    return factor * np.array([x * (1.0 - zr2), y * (1.0 - zr2), z * (3.0 - zr2)])


def exponential_density(altitude, rho0=3.9e-12, h_ref=400_000.0, scale_height=60_000.0):
    if altitude < -h_ref:
        return 0.0
    return rho0 * np.exp(-(altitude - h_ref) / scale_height)


def drag_acceleration(
    position, velocity, cd=2.2, area=1.0, mass=100.0,
    r_earth=R_EARTH, omega_earth=OMEGA_EARTH, density_func=exponential_density,
):
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    altitude = np.linalg.norm(position) - r_earth
    rho = density_func(altitude)
    if rho <= 0.0:
        return np.zeros(3)
    omega_vec = np.array([0.0, 0.0, omega_earth])
    v_rel = velocity - np.cross(omega_vec, position)
    speed = np.linalg.norm(v_rel)
    ballistic = cd * area / mass
    return -0.5 * rho * ballistic * speed * v_rel


@dataclass
class ForceModel:
    use_j2: bool = False
    use_drag: bool = False
    mu: float = MU_EARTH
    r_earth: float = R_EARTH
    j2: float = J2_EARTH
    cd: float = 2.2
    area: float = 1.0
    mass: float = 100.0
    omega_earth: float = OMEGA_EARTH

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
        state = np.asarray(state, dtype=float)
        return np.concatenate((state[3:], self.acceleration(state, t)))


__all__ = [
    "j2_acceleration",
    "exponential_density",
    "drag_acceleration",
    "ForceModel",
]
