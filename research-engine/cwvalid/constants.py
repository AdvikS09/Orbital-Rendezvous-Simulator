import math

from src.orbit import J2EARTH, MUEARTH, OMEGAEARTH, REARTH

DEFALT = 400_000.0
DEFRAD = REARTH + DEFALT
DEFINC = math.radians(51.6)

ALTFLOOR = 100_000.0


def mean_motion(radius=DEFRAD, mu=MUEARTH):
    return math.sqrt(mu / radius**3)


def period(radius=DEFRAD, mu=MUEARTH):
    return 2.0 * math.pi / mean_motion(radius, mu)
