"""Physical constants and default mission parameters.

All quantities are SI (metres, seconds, kilograms, Newtons). These values are
shared across the whole platform so that every module agrees on the same
gravitational parameter, Earth radius and default Low-Earth-Orbit geometry.
"""

# --- Fundamental / Earth gravitational parameters -------------------------
G = 6.67430e-11               # Newtonian gravitational constant [m^3 kg^-1 s^-2]
MU_EARTH = 3.986004418e14     # Earth standard gravitational parameter mu = G*M [m^3/s^2]
M_EARTH = MU_EARTH / G        # Earth mass [kg]
R_EARTH = 6_378_137.0         # Earth equatorial radius (WGS-84) [m]
J2_EARTH = 1.08262668e-3      # Earth J2 oblateness coefficient (reserved for later fidelity)

# --- Default mission: circular Low Earth Orbit at 400 km altitude ---------
DEFAULT_ALTITUDE = 400_000.0                  # altitude above Earth's surface [m]
DEFAULT_RADIUS = R_EARTH + DEFAULT_ALTITUDE   # circular orbit radius from Earth centre [m]

__all__ = [
    "G",
    "MU_EARTH",
    "M_EARTH",
    "R_EARTH",
    "J2_EARTH",
    "DEFAULT_ALTITUDE",
    "DEFAULT_RADIUS",
]
