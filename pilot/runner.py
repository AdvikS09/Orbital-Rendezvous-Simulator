"""The one path every simulation takes: validate, guard, run, record.

Keeping this in a single place means the guarantees hold everywhere:

1. inputs are validated before the engine is touched (no ``NaN`` trajectories,
   no silently truncated horizons);
2. the shared server is protected by a rate limit and a predicted-cost check;
3. exactly one ``simulation_started`` and one ``simulation_completed`` or
   ``simulation_failed`` event is recorded per attempt; and
4. analytics failures can never break a participant's run.

This module deliberately contains **no physics**. It calls the scenario
runners in ``pilot/content/scenarios_catalog.py``, which call
``src/scenarios.py``, which is the validated engine boundary and is not
modified by any of this.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from .analytics import EventType, ParticipantContext, record_event
from .config import Settings, get_settings
from .content.scenarios_catalog import Scenario
from .ratelimit import get_limiter
from .validation import ValidationResult, validate_simulation_request

logger = logging.getLogger("pilot.runner")

#: Parameter names the validator understands, mapped from catalogue names.
_VALIDATED_KEYS = (
    "altitude_km", "inclination_deg", "duration_s", "dt_s", "n_orbits",
    "kp", "kd", "max_accel", "mass", "area", "cd",
)


@dataclass
class RunOutcome:
    """Result of one attempted simulation."""

    ok: bool
    scenario_id: str
    result: object = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_ms: int = 0
    n_steps: int = 0
    error_class: str | None = None
    rejected: bool = False          # blocked before running (validation or rate limit)


def _bucket(value: float | None, edges: tuple) -> str | None:
    """Coarsen a number into a labelled band.

    Analytics stores bands, never the value the participant typed, so a
    parameter sweep cannot be reconstructed from the event log.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    for edge, label in edges:
        if number < edge:
            return label
    return edges[-1][1]


_ALTITUDE_BANDS = ((300, "<300km"), (500, "300-500km"), (800, "500-800km"),
                   (1200, "800-1200km"), (float("inf"), ">1200km"))
_SEPARATION_BANDS = ((100, "<100m"), (1_000, "100m-1km"), (10_000, "1-10km"),
                     (50_000, "10-50km"), (float("inf"), ">50km"))
_STEP_BANDS = ((500, "<500"), (2_000, "500-2k"), (10_000, "2k-10k"),
               (40_000, "10k-40k"), (float("inf"), ">40k"))


def _safe_detail(scenario: Scenario, params: dict, validation: ValidationResult) -> dict:
    """Build the non-identifying detail payload for an event."""
    separation = None
    if any(key in params for key in ("rel_x", "rel_y", "rel_z")):
        separation = math.sqrt(sum(
            float(params.get(key, 0.0) or 0.0) ** 2 for key in ("rel_x", "rel_y", "rel_z")))
    detail = {
        "model": scenario.model,
        "steps_bucket": _bucket(validation.n_steps, _STEP_BANDS),
        "altitude_bucket_km": _bucket(params.get("altitude_km"), _ALTITUDE_BANDS),
        "separation_bucket_m": _bucket(separation, _SEPARATION_BANDS),
        "warning_count": len(validation.warnings),
    }
    if "controlled" in params:
        detail["controlled"] = bool(params["controlled"])
    if "use_j2" in params or "use_drag" in params:
        j2, drag = bool(params.get("use_j2")), bool(params.get("use_drag"))
        detail["force_model"] = ("two-body + J2 + drag" if j2 and drag else
                                 "two-body + J2" if j2 else
                                 "two-body + drag" if drag else "two-body")
    if get_settings().log_raw_parameters:
        # Opt-in only, and clearly separated from the derived fields above.
        detail["raw_parameters"] = {k: v for k, v in params.items()}
    return {k: v for k, v in detail.items() if v is not None}


def validate_params(scenario: Scenario, params: dict,
                    settings: Settings | None = None) -> ValidationResult:
    """Validate a parameter dict for a scenario, without running anything."""
    settings = settings or get_settings()
    kwargs = {key: params[key] for key in _VALIDATED_KEYS if key in params}

    if any(key in params for key in ("rel_x", "rel_y", "rel_z")):
        kwargs["rel_pos"] = [params.get("rel_x", 0.0), params.get("rel_y", 0.0),
                             params.get("rel_z", 0.0)]
    if any(key in params for key in ("rel_vx", "rel_vy", "rel_vz")):
        kwargs["rel_vel"] = [params.get("rel_vx", 0.0), params.get("rel_vy", 0.0),
                             params.get("rel_vz", 0.0)]

    # Gains only constrain the run when the user actually set them.
    if not params.get("use_explicit_gains", True):
        kwargs.pop("kp", None)
        kwargs.pop("kd", None)

    # The gain sweep runs 16 rendezvous, so its predicted cost is 16x a single run.
    max_steps = settings.max_simulation_steps
    max_seconds = settings.max_estimated_seconds
    if scenario.id == "gain_tuning":
        max_steps = max(1, max_steps // 16)
        max_seconds = max_seconds / 16.0
    elif scenario.id == "integrator_convergence":
        # Fixed internal grid totalling ~7900 steps; user cannot inflate it.
        max_steps = max(max_steps, 10_000)

    return validate_simulation_request(
        model=scenario.model, max_steps=max_steps, max_seconds=max_seconds, **kwargs)


def run_scenario(scenario: Scenario, params: dict,
                 context: ParticipantContext | None = None,
                 settings: Settings | None = None) -> RunOutcome:
    """Validate, guard, execute and record one simulation.

    Never raises: an engine exception becomes ``ok=False`` with an
    ``error_class``, so the UI can show a useful message instead of a traceback.
    """
    settings = settings or get_settings()

    # 1. validate
    validation = validate_params(scenario, params, settings)
    if not validation.ok:
        if context is not None:
            record_event(context, EventType.SIMULATION_REJECTED,
                         scenario_category=scenario.category,
                         scenario_id=scenario.id,
                         feature_category=scenario.feature_category,
                         success=False, error_class="ValidationError",
                         detail={"rejected_reason": "validation",
                                 **_safe_detail(scenario, params, validation)},
                         settings=settings)
        return RunOutcome(ok=False, scenario_id=scenario.id, errors=validation.errors,
                          warnings=validation.warnings, n_steps=validation.n_steps,
                          error_class="ValidationError", rejected=True)

    # 2. rate limit
    if context is not None:
        decision = get_limiter().consume(context.session_token)
        if not decision.allowed:
            record_event(context, EventType.SIMULATION_REJECTED,
                         scenario_category=scenario.category,
                         scenario_id=scenario.id,
                         feature_category=scenario.feature_category,
                         success=False, error_class="RateLimited",
                         detail={"rejected_reason": "rate_limit"},
                         settings=settings)
            return RunOutcome(ok=False, scenario_id=scenario.id,
                              errors=[decision.reason], warnings=validation.warnings,
                              n_steps=validation.n_steps, error_class="RateLimited",
                              rejected=True)

    detail = _safe_detail(scenario, params, validation)

    # 3. run
    if context is not None:
        record_event(context, EventType.SIMULATION_STARTED,
                     scenario_category=scenario.category, scenario_id=scenario.id,
                     feature_category=scenario.feature_category,
                     detail=detail, settings=settings)

    started = time.perf_counter()
    try:
        result = scenario.runner(dict(params))
    except Exception as exc:  # noqa: BLE001 - the UI must never show a traceback
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.exception("scenario %s failed", scenario.id)
        if context is not None:
            record_event(context, EventType.SIMULATION_FAILED,
                         scenario_category=scenario.category, scenario_id=scenario.id,
                         feature_category=scenario.feature_category,
                         duration_ms=elapsed_ms, success=False,
                         error_class=type(exc).__name__, detail=detail,
                         settings=settings)
        return RunOutcome(
            ok=False, scenario_id=scenario.id, duration_ms=elapsed_ms,
            n_steps=validation.n_steps, warnings=validation.warnings,
            error_class=type(exc).__name__,
            errors=[f"The simulation could not be completed ({type(exc).__name__}). "
                    "Try adjusting the parameters, or report this if it persists."])

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # 4. record success
    if context is not None:
        record_event(context, EventType.SIMULATION_COMPLETED,
                     scenario_category=scenario.category, scenario_id=scenario.id,
                     feature_category=scenario.feature_category,
                     duration_ms=elapsed_ms, success=True, detail=detail,
                     settings=settings)
        record_event(context, EventType.SCENARIO_CATEGORY_USED,
                     scenario_category=scenario.category, scenario_id=scenario.id,
                     settings=settings)
        record_event(context, EventType.FEATURE_CATEGORY_USED,
                     feature_category=scenario.feature_category, settings=settings)

    return RunOutcome(ok=True, scenario_id=scenario.id, result=result,
                      warnings=validation.warnings, duration_ms=elapsed_ms,
                      n_steps=validation.n_steps)


__all__ = ["RunOutcome", "run_scenario", "validate_params"]
