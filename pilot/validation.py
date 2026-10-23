from __future__ import annotations

import math
from dataclasses import dataclass, field

from orbital_core import orbit

MIN_ALTITUDE_KM = 150.0
MAX_ALTITUDE_KM = 40_000.0

MIN_INCLINATION_DEG = 0.0
MAX_INCLINATION_DEG = 180.0

MIN_DT_S = 0.01
MAX_DT_S = 600.0

MIN_DURATION_S = 10.0
MAX_DURATION_ORBITS = 50.0

CW_CHARACTERISED_SEPARATION_M = 50_000.0
CW_TRUSTWORTHY_SEPARATION_M = 1_000.0
MAX_SEPARATION_M = 500_000.0
MAX_REL_VELOCITY_MPS = 100.0

MIN_MASS_KG = 0.1
MAX_MASS_KG = 100_000.0
MIN_AREA_M2 = 0.001
MAX_AREA_M2 = 1_000.0
MIN_CD = 0.0
MAX_CD = 10.0

MIN_GAIN = 0.0
MAX_KP = 1.0
MAX_KD = 10.0
MIN_MAX_ACCEL = 1e-6
MAX_MAX_ACCEL = 10.0

SECONDS_PER_STEP = {
    "cw": 6.0e-5,
    "orbit": 6.5e-5,
    "high_fidelity": 4.2e-4,
    "comparison": 4.2e-4,
}


class ValidationError(ValueError):
    pass


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_steps: int = 0
    estimated_seconds: float = 0.0

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValidationError("; ".join(self.errors))


def check_number(value, name: str, minimum: float, maximum: float,
                 unit: str = "") -> tuple[float | None, str | None]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f"{name} must be a number (got {value!r})."
    if math.isnan(number):
        return None, f"{name} must be a number, not NaN."
    if math.isinf(number):
        return None, f"{name} must be finite, not infinite."
    suffix = f" {unit}" if unit else ""
    if number < minimum:
        return None, f"{name} must be at least {minimum:g}{suffix} (got {number:g}{suffix})."
    if number > maximum:
        return None, f"{name} must be at most {maximum:g}{suffix} (got {number:g}{suffix})."
    return number, None


def check_vector(values, name: str, limit: float, unit: str) -> tuple[list | None, list[str]]:
    errors: list[str] = []
    try:
        items = list(values)
    except TypeError:
        return None, [f"{name} must be three numbers."]
    if len(items) != 3:
        return None, [f"{name} must have exactly three components (got {len(items)})."]
    out = []
    for component, axis in zip(items, ("x", "y", "z")):
        number, error = check_number(component, f"{name} {axis}", -limit, limit, unit)
        if error:
            errors.append(error)
        else:
            out.append(number)
    return (out if not errors else None), errors


def validate_simulation_request(*, altitude_km=None, inclination_deg=None,
                                duration_s=None, dt_s=None, n_orbits=None,
                                rel_pos=None, rel_vel=None, kp=None, kd=None,
                                max_accel=None, mass=None, area=None, cd=None,
                                model="cw", max_steps=40_000,
                                max_seconds=20.0) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    altitude = inclination = duration = dt = None

    if altitude_km is not None:
        altitude, error = check_number(altitude_km, "Altitude",
                                       MIN_ALTITUDE_KM, MAX_ALTITUDE_KM, "km")
        if error:
            errors.append(error)
        elif altitude < 200.0:
            warnings.append(
                f"At {altitude:g} km the exponential atmosphere in this engine is "
                "extrapolated well below the 400 km altitude it is calibrated for, "
                "so drag results there are indicative only.")

    if inclination_deg is not None:
        inclination, error = check_number(inclination_deg, "Inclination",
                                          MIN_INCLINATION_DEG, MAX_INCLINATION_DEG, "deg")
        if error:
            errors.append(error)

    if dt_s is not None:
        dt, error = check_number(dt_s, "Timestep", MIN_DT_S, MAX_DT_S, "s")
        if error:
            errors.append(error)

    if duration_s is not None:
        duration, error = check_number(duration_s, "Duration", MIN_DURATION_S,
                                       float("inf"), "s")
        if error:
            errors.append(error)

    if duration is not None and altitude is not None:
        period = orbit.orbital_period(orbit.R_EARTH + altitude * 1000.0)
        if duration > MAX_DURATION_ORBITS * period:
            errors.append(
                f"Duration must be at most {MAX_DURATION_ORBITS:g} orbits "
                f"({MAX_DURATION_ORBITS * period / 3600:.1f} hours at this altitude); "
                f"you asked for {duration / period:.1f} orbits.")

    if n_orbits is not None:
        _, error = check_number(n_orbits, "Number of orbits", 0.01, MAX_DURATION_ORBITS)
        if error:
            errors.append(error)

    if rel_pos is not None:
        position, position_errors = check_vector(
            rel_pos, "Relative position", MAX_SEPARATION_M, "m")
        errors.extend(position_errors)
        if position is not None:
            separation = math.sqrt(sum(component * component for component in position))
            if separation == 0.0 and rel_vel is not None:
                velocity_norm = _safe_norm(rel_vel)
                if velocity_norm == 0.0:
                    errors.append(
                        "The chaser starts exactly on the target with zero relative "
                        "velocity, so there is nothing to simulate. Offset at least "
                        "one axis.")
            if separation > CW_CHARACTERISED_SEPARATION_M:
                warnings.append(
                    f"Initial separation is {separation / 1000:.1f} km. The "
                    "Clohessy-Wiltshire model is a linearisation about the target's "
                    "orbit; this repository characterises it only out to 50 km "
                    "(experiments/run_separation_sweep.py). Treat this result as "
                    "illustrative, not quantitative.")
            elif separation > CW_TRUSTWORTHY_SEPARATION_M:
                warnings.append(
                    f"Initial separation is {separation:.0f} m. Published CW-vs-"
                    "reference agreement in this repository is demonstrated at "
                    "separations up to about 1 km; error grows with separation, and "
                    "fastest for a radial offset.")

    if rel_vel is not None:
        _, velocity_errors = check_vector(
            rel_vel, "Relative velocity", MAX_REL_VELOCITY_MPS, "m/s")
        errors.extend(velocity_errors)

    if kp is not None:
        value, error = check_number(kp, "Kp", MIN_GAIN, MAX_KP, "1/s^2")
        if error:
            errors.append(error)
        elif value == 0.0:
            warnings.append("Kp is zero, so there is no proportional (position) feedback.")
    if kd is not None:
        value, error = check_number(kd, "Kd", MIN_GAIN, MAX_KD, "1/s")
        if error:
            errors.append(error)

    if max_accel is not None:
        _, error = check_number(max_accel, "Max thrust acceleration",
                                MIN_MAX_ACCEL, MAX_MAX_ACCEL, "m/s^2")
        if error:
            errors.append(error)

    if mass is not None:
        _, error = check_number(mass, "Mass", MIN_MASS_KG, MAX_MASS_KG, "kg")
        if error:
            errors.append(error)
    if area is not None:
        _, error = check_number(area, "Cross-section area", MIN_AREA_M2, MAX_AREA_M2, "m^2")
        if error:
            errors.append(error)
    if cd is not None:
        _, error = check_number(cd, "Drag coefficient", MIN_CD, MAX_CD)
        if error:
            errors.append(error)

    n_steps = 0
    if duration is not None and dt is not None and dt > 0:
        n_steps = int(round(duration / dt))
    elif n_orbits is not None and altitude is not None:
        try:
            n_steps = int(round(float(n_orbits) * 2000))
        except (TypeError, ValueError):
            n_steps = 0

    estimated = n_steps * SECONDS_PER_STEP.get(model, SECONDS_PER_STEP["cw"])

    if n_steps > max_steps:
        errors.append(
            f"This run needs {n_steps:,} integration steps, above the {max_steps:,} "
            "limit for the shared pilot server. Increase the timestep or shorten "
            "the duration. (The engine itself has no such limit -- run it locally "
            "for long studies.)")
    if estimated > max_seconds:
        errors.append(
            f"This run is estimated to take about {estimated:.0f} s, above the "
            f"{max_seconds:.0f} s limit for the shared pilot server. Increase the "
            "timestep or shorten the duration.")

    if dt is not None and altitude is not None and not errors:
        period = orbit.orbital_period(orbit.R_EARTH + altitude * 1000.0)
        samples_per_orbit = period / dt
        if samples_per_orbit < 200:
            warnings.append(
                f"Only {samples_per_orbit:.0f} steps per orbit. The published results "
                "use 500-2000; below about 200 the fixed-step RK4 result starts to "
                "depend visibly on the step size. Reduce the timestep to check.")

    return ValidationResult(
        ok=not errors, errors=errors, warnings=warnings,
        n_steps=n_steps, estimated_seconds=estimated,
    )


def _safe_norm(vector) -> float:
    try:
        return math.sqrt(sum(float(component) ** 2 for component in vector))
    except (TypeError, ValueError):
        return 0.0


MIN_CODE_LENGTH = 3
MAX_CODE_LENGTH = 40

_EMAIL_MARKER = "@"


def validate_participant_code(code: str) -> tuple[str, str | None]:
    normalised = (code or "").strip()
    if not normalised:
        return "", "Enter an anonymous participant code."
    if len(normalised) < MIN_CODE_LENGTH:
        return "", f"The participant code must be at least {MIN_CODE_LENGTH} characters."
    if len(normalised) > MAX_CODE_LENGTH:
        return "", f"The participant code must be at most {MAX_CODE_LENGTH} characters."
    if _EMAIL_MARKER in normalised:
        return "", ("That looks like an email address. Please use an anonymous code "
                    "that is not personal information.")
    if not all(character.isalnum() or character in "-_" for character in normalised):
        return "", ("Use letters, digits, hyphens and underscores only, so the code "
                    "cannot contain personal information.")
    return normalised.upper(), None


def validate_pilot_code(code: str) -> tuple[str, str | None]:
    normalised = (code or "").strip()
    if not normalised:
        return "", "Enter the organisation or pilot code you were given."
    if len(normalised) > MAX_CODE_LENGTH:
        return "", f"The pilot code must be at most {MAX_CODE_LENGTH} characters."
    if not all(character.isalnum() or character in "-_" for character in normalised):
        return "", "Use letters, digits, hyphens and underscores only."
    return normalised.upper(), None


__all__ = [
    "ValidationError", "ValidationResult", "validate_simulation_request",
    "validate_participant_code", "validate_pilot_code", "check_number",
    "check_vector", "MIN_ALTITUDE_KM", "MAX_ALTITUDE_KM", "MAX_DURATION_ORBITS",
    "CW_TRUSTWORTHY_SEPARATION_M", "CW_CHARACTERISED_SEPARATION_M",
]
