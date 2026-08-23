"""Expose the scenario catalogue over the API.

Reads ``pilot.content.scenarios_catalog`` directly. There is no second copy of
any scenario definition, parameter bound or piece of explanatory text: the
Streamlit application and this API describe scenarios from the same objects, so
they cannot disagree.
"""

from __future__ import annotations

from typing import Any

from app import bootstrap  # noqa: F401  (side effect: repo root on sys.path)

from pilot.content.scenarios_catalog import (  # noqa: E402
    CATEGORY_LABELS,
    SCENARIOS,
    Param,
    Scenario,
    default_params,
    get_scenario,
    scenarios_for,
)
from pilot.runner import validate_params  # noqa: E402
from pilot import validation as pilot_validation  # noqa: E402

from app.schemas.common import NumberFormat  # noqa: E402
from app.schemas.scenarios import ParamSpec, ScenarioDetail, ScenarioSummary  # noqa: E402

#: Bounds enforced by pilot.validation, keyed by parameter name. These are
#: tighter than some widget ranges and are what actually rejects a run, so the
#: frontend shows them and blocks before a pointless round-trip.
_HARD_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "altitude_km": (pilot_validation.MIN_ALTITUDE_KM, pilot_validation.MAX_ALTITUDE_KM),
    "inclination_deg": (
        pilot_validation.MIN_INCLINATION_DEG,
        pilot_validation.MAX_INCLINATION_DEG,
    ),
    "dt_s": (pilot_validation.MIN_DT_S, pilot_validation.MAX_DT_S),
    "duration_s": (pilot_validation.MIN_DURATION_S, None),
    "kp": (pilot_validation.MIN_GAIN, pilot_validation.MAX_KP),
    "kd": (pilot_validation.MIN_GAIN, pilot_validation.MAX_KD),
    "max_accel": (pilot_validation.MIN_MAX_ACCEL, pilot_validation.MAX_MAX_ACCEL),
    "mass": (pilot_validation.MIN_MASS_KG, pilot_validation.MAX_MASS_KG),
    "area": (pilot_validation.MIN_AREA_M2, pilot_validation.MAX_AREA_M2),
    "cd": (pilot_validation.MIN_CD, pilot_validation.MAX_CD),
}

for _axis in ("rel_x", "rel_y", "rel_z"):
    _HARD_BOUNDS[_axis] = (
        -pilot_validation.MAX_SEPARATION_M,
        pilot_validation.MAX_SEPARATION_M,
    )
for _axis in ("rel_vx", "rel_vy", "rel_vz"):
    _HARD_BOUNDS[_axis] = (
        -pilot_validation.MAX_REL_VELOCITY_MPS,
        pilot_validation.MAX_REL_VELOCITY_MPS,
    )


def _format_for(param: Param) -> tuple[NumberFormat, int | None]:
    """Choose a display format from the parameter's unit and magnitude.

    Gains sit around 1e-5 and 1e-2, so fixed notation would render them as
    ``0.00005``. They get scientific notation; everything else is automatic.
    """
    if param.kind == "checkbox" or param.kind == "select":
        return NumberFormat.AUTO, None
    if param.name in ("kp", "kd"):
        return NumberFormat.SCIENTIFIC, 3
    if param.unit == "deg":
        return NumberFormat.ANGLE, 2
    if param.name == "max_accel":
        return NumberFormat.SIGNIFICANT, 4
    return NumberFormat.AUTO, None


def to_param_spec(param: Param) -> ParamSpec:
    fmt, precision = _format_for(param)
    hard_min, hard_max = _HARD_BOUNDS.get(param.name, (None, None))
    return ParamSpec(
        name=param.name,
        label=param.label,
        kind=param.kind,  # type: ignore[arg-type]
        default=param.default,
        minimum=param.minimum,
        maximum=param.maximum,
        step=param.step,
        unit=param.unit,
        help=param.help,
        options=list(param.options),
        advanced=param.advanced,
        fmt=fmt,
        precision=precision,
        hard_minimum=hard_min,
        hard_maximum=hard_max,
    )


def to_summary(scenario: Scenario) -> ScenarioSummary:
    return ScenarioSummary(
        id=scenario.id,
        title=scenario.title,
        category=scenario.category,
        category_label=CATEGORY_LABELS.get(scenario.category, scenario.category),
        feature_category=scenario.feature_category,
        summary=scenario.summary,
        levels=list(scenario.levels),
        result_kind=scenario.result_kind,
        model=scenario.model,
        param_count=len(scenario.params),
    )


def to_detail(scenario: Scenario) -> ScenarioDetail:
    defaults = default_params(scenario)
    estimate = validate_params(scenario, defaults)
    return ScenarioDetail(
        id=scenario.id,
        title=scenario.title,
        category=scenario.category,
        category_label=CATEGORY_LABELS.get(scenario.category, scenario.category),
        feature_category=scenario.feature_category,
        model=scenario.model,
        result_kind=scenario.result_kind,
        levels=list(scenario.levels),
        summary=scenario.summary,
        explanation=scenario.explanation,
        reading_guide=scenario.reading_guide or "",
        references=list(scenario.references),
        assumptions=list(scenario.assumptions),
        limitations=list(scenario.limitations),
        params=[to_param_spec(param) for param in scenario.params],
        defaults=defaults,
        estimated_steps=estimate.n_steps or None,
        estimated_seconds=round(estimate.estimated_seconds, 3) or None,
    )


def list_scenarios(track: str | None = None) -> list[ScenarioSummary]:
    scenarios = scenarios_for(track) if track else list(SCENARIOS)
    return [to_summary(scenario) for scenario in scenarios]


def find(scenario_id: str) -> Scenario | None:
    return get_scenario(scenario_id)


def category_labels() -> dict[str, str]:
    return dict(CATEGORY_LABELS)


def coerce_params(scenario: Scenario, supplied: dict[str, Any]) -> dict[str, Any]:
    """Merge supplied values over the defaults, dropping unknown names.

    Coercion is type-only. Range checking belongs to ``pilot.validation``, which
    produces the explanatory messages participants actually see; silently
    clamping here would recreate exactly the silent-wrong-answer problem the
    validation layer exists to fix.
    """
    resolved = default_params(scenario)
    by_name = {param.name: param for param in scenario.params}

    for name, value in supplied.items():
        param = by_name.get(name)
        if param is None:
            continue
        if param.kind == "checkbox":
            resolved[name] = bool(value)
        elif param.kind == "select":
            resolved[name] = value if value in param.options else param.default
        else:
            try:
                resolved[name] = float(value)
            except (TypeError, ValueError):
                # Leave it as supplied; the validator reports it properly.
                resolved[name] = value
    return resolved
