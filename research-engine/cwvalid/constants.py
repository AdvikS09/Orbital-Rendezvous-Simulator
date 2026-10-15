"""Physical constants and study defaults for the CW closed-loop validity extension.

All physical constants are imported from the existing engine (``src.orbit``) so
the extension uses exactly the values documented in the manuscript (Table 3).
SI units throughout.
"""

import math

from src.orbit import J2_EARTH, MU_EARTH, OMEGA_EARTH, R_EARTH

#: Reference (chief/target) circular-orbit altitude used by the main study [m].
DEFAULT_ALTITUDE = 400_000.0
#: Reference circular-orbit radius [m].
DEFAULT_RADIUS = R_EARTH + DEFAULT_ALTITUDE
#: Inclination used for the inertial (ECI) branch [rad]. Irrelevant for the
#: two-body comparisons; matters only when J2 is switched on.
DEFAULT_INCLINATION = math.radians(51.6)

#: Altitude floor used as a physical-domain termination for nonlinear branches [m].
ALTITUDE_FLOOR = 100_000.0


def mean_motion(radius=DEFAULT_RADIUS, mu=MU_EARTH):
    """n = sqrt(mu / R^3) [rad/s]."""
    return math.sqrt(mu / radius**3)


def period(radius=DEFAULT_RADIUS, mu=MU_EARTH):
    """Keplerian period 2*pi/n [s]."""
    return 2.0 * math.pi / mean_motion(radius, mu)


__all__ = [
    "MU_EARTH", "R_EARTH", "J2_EARTH", "OMEGA_EARTH",
    "DEFAULT_ALTITUDE", "DEFAULT_RADIUS", "DEFAULT_INCLINATION", "ALTITUDE_FLOOR",
    "mean_motion", "period",
]
