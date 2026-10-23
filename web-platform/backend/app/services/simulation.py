from __future__ import annotations

from typing import Any

from app import bootstrap  # noqa: F401

from pilot.analytics import ParticipantContext  # noqa: E402
from pilot.content.scenarios_catalog import Scenario  # noqa: E402
from pilot.runner import RunOutcome, run_scenario, validate_params  # noqa: E402

from app.schemas.common import ErrorResponse, FieldError  # noqa: E402
from app.schemas.scenarios import SimulationResult  # noqa: E402
from app.services.adapters import build_result  # noqa: E402
from app.services.catalogue import coerce_params  # noqa: E402

_FIELD_HINTS: tuple[tuple[str, str], ...] = (
    ("Altitude", "altitude_km"),
    ("Inclination", "inclination_deg"),
    ("Timestep", "dt_s"),
    ("Duration", "duration_s"),
    ("Number of orbits", "n_orbits"),
    ("Relative position x", "rel_x"),
    ("Relative position y", "rel_y"),
    ("Relative position z", "rel_z"),
    ("Relative velocity x", "rel_vx"),
    ("Relative velocity y", "rel_vy"),
    ("Relative velocity z", "rel_vz"),
    ("Kp", "kp"),
    ("Kd", "kd"),
    ("Max thrust acceleration", "max_accel"),
    ("Mass", "mass"),
    ("Cross-section area", "area"),
    ("Drag coefficient", "cd"),
)


def _attribute(message: str) -> FieldError:
    for prefix, field in _FIELD_HINTS:
        if message.startswith(prefix):
            return FieldError(field=field, message=message)
    return FieldError(field=None, message=message)


def preflight(scenario: Scenario, params: dict[str, Any]) -> dict[str, Any]:
    resolved = coerce_params(scenario, params)
    outcome = validate_params(scenario, resolved)
    return {
        "ok": outcome.ok,
        "errors": [_attribute(message).model_dump() for message in outcome.errors],
        "warnings": list(outcome.warnings),
        "n_steps": outcome.n_steps,
        "estimated_seconds": round(outcome.estimated_seconds, 3),
        "params": resolved,
    }


def run(
    scenario: Scenario,
    params: dict[str, Any],
    context: ParticipantContext | None = None,
) -> tuple[SimulationResult | None, ErrorResponse | None]:
    resolved = coerce_params(scenario, params)
    outcome: RunOutcome = run_scenario(scenario, resolved, context)

    if not outcome.ok:
        return None, ErrorResponse(
            error_class=outcome.error_class or "SimulationError",
            message=(outcome.errors[0] if outcome.errors
                     else "The simulation could not be completed."),
            detail=[_attribute(message) for message in outcome.errors],
            warnings=list(outcome.warnings),
            rejected=outcome.rejected,
        )

    result = build_result(
        scenario=scenario,
        params=resolved,
        result=outcome.result,
        runtime_ms=outcome.duration_ms,
        n_steps=outcome.n_steps,
        warnings=list(outcome.warnings),
    )
    return result, None
