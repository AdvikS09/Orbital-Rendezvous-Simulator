"""Health, exports and the protected administrator reporting endpoints."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import date, datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from fastapi.responses import PlainTextResponse, Response

from app import bootstrap  # noqa: F401

import orbital_core  # noqa: E402
from pilot import analytics  # noqa: E402
from pilot.analytics import EventType, ReportFilters  # noqa: E402
from pilot.config import get_settings  # noqa: E402
from pilot.db import healthcheck  # noqa: E402
from pilot.content.scenarios_catalog import SCENARIOS  # noqa: E402
from pilot.export import config_json, results_readme  # noqa: E402

from app import __version__ as api_version  # noqa: E402
from app.api.deps import optional_participant, require_admin  # noqa: E402
from app.schemas.pilot import AdminReport, SummaryMetrics  # noqa: E402
from app.schemas.scenarios import HealthResponse  # noqa: E402
from app.services import catalogue  # noqa: E402

router = APIRouter(tags=["system"])

#: Shipped with every export so a spreadsheet cannot drift from its meaning.
METRIC_DEFINITIONS: dict[str, str] = {
    "unique_user": (
        "An anonymous participant who has completed at least one simulation. "
        "Not limited to the reporting period: being unique is a property of the "
        "participant, not of a window."
    ),
    "active_user": (
        "An anonymous participant who completed a simulation during the selected "
        "reporting period."
    ),
    "returning_user": (
        "An anonymous participant who used the platform on at least two separate "
        "dates. Two sessions on the same day is not returning."
    ),
    "completed_simulation": (
        "One simulation that finished successfully and displayed results "
        "(a simulation_completed event)."
    ),
    "failed_simulation": (
        "One simulation that raised an error during execution. Runs rejected by "
        "input validation or rate limiting are NOT failures - they are recorded "
        "separately as simulation_rejected, because the engine never ran."
    ),
    "measured_impact_participant": (
        "A participant who completed both the pre- and the post-assessment for "
        "the same instrument."
    ),
    "return_rate": "returning users divided by unique users.",
    "simulations_per_user": "completed runs divided by unique users.",
    "page_view": (
        "NOT counted as a user anywhere in this platform. A visitor who signs in "
        "but never completes a simulation is not a unique user."
    ),
}


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health() -> HealthResponse:
    """Liveness and configuration report.

    Deliberately does not require authentication: a container orchestrator has
    to be able to call it. It exposes no secret - the database is described by
    backend and host only, never by its URL.
    """
    settings = get_settings()
    report = healthcheck()
    return HealthResponse(
        status="ok" if report["ok"] else "degraded",
        engine_version=orbital_core.__version__,
        api_version=api_version,
        app_version=settings.app_version,
        environment=settings.environment,
        database_ok=bool(report["ok"]),
        database=report["database"],
        schema_version=int(report["schema_version"]),
        scenario_count=len(SCENARIOS),
        warnings=settings.deployment_warnings(),
    )


# Exports
@router.post(
    "/exports/{scenario_id}/config",
    response_class=PlainTextResponse,
    summary="Reproducible scenario configuration (JSON)",
)
def export_config(
    scenario_id: str,
    params: Annotated[dict[str, Any], Body(embed=True)],
    context=Depends(optional_participant),
) -> PlainTextResponse:
    """A file that fully determines the run, so it can be reproduced later.

    Carries no participant identifier: a configuration is about the simulation,
    not about who ran it.
    """
    scenario = catalogue.find(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"No scenario {scenario_id!r}.")
    settings = get_settings()
    resolved = catalogue.coerce_params(scenario, params)
    if context is not None:
        analytics.record_event(
            context, EventType.RESULTS_DOWNLOADED, feature_category="download",
            scenario_id=scenario_id, detail={"download_kind": "scenario_config"},
        )
    return PlainTextResponse(
        config_json(scenario, resolved, settings.app_version),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{scenario_id}_config.json"'
        },
    )


@router.post(
    "/exports/{scenario_id}/notes",
    response_class=PlainTextResponse,
    summary="Human-readable run summary with assumptions and caveats",
)
def export_notes(
    scenario_id: str,
    params: Annotated[dict[str, Any], Body(embed=True)],
    warnings: Annotated[list[str], Body(embed=True)] = [],
    context=Depends(optional_participant),
) -> PlainTextResponse:
    scenario = catalogue.find(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"No scenario {scenario_id!r}.")
    settings = get_settings()
    resolved = catalogue.coerce_params(scenario, params)
    if context is not None:
        analytics.record_event(
            context, EventType.RESULTS_DOWNLOADED, feature_category="download",
            scenario_id=scenario_id, detail={"download_kind": "notes"},
        )
    return PlainTextResponse(
        results_readme(scenario, resolved, list(warnings), settings.app_version),
        headers={
            "Content-Disposition": f'attachment; filename="{scenario_id}_notes.txt"'
        },
    )


# Administration
def _filters(
    start: str | None, end: str | None, organisation_id: int | None,
    user_type: str | None, include_demo: bool,
) -> ReportFilters:
    def parse(value: str | None) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid date {value!r}. Expected YYYY-MM-DD.",
            )

    return ReportFilters(
        start=parse(start), end=parse(end), organisation_id=organisation_id,
        user_type=user_type, include_demo=include_demo,
    )


@router.get(
    "/admin/report",
    response_model=AdminReport,
    summary="Anonymous aggregate pilot report (administrator only)",
)
def admin_report(
    start: Annotated[str | None, Query()] = None,
    end: Annotated[str | None, Query()] = None,
    organisation_id: Annotated[int | None, Query()] = None,
    user_type: Annotated[str | None, Query()] = None,
    include_demo: Annotated[bool, Query()] = False,
    _admin: str = Depends(require_admin),
) -> AdminReport:
    """Every reporting table, anonymous and aggregate throughout.

    Development and demonstration records are excluded by default, so a demo run
    can never be mistaken for pilot activity.
    """
    settings = get_settings()
    filters = _filters(start, end, organisation_id, user_type, include_demo)
    summary = analytics.summary_metrics(filters)

    return AdminReport(
        summary=SummaryMetrics(**summary),
        usage_by_date=[_jsonable(r) for r in analytics.usage_by_date(filters)],
        usage_by_organisation=[_jsonable(r) for r in analytics.usage_by_organisation(filters)],
        usage_by_pilot_code=[_jsonable(r) for r in analytics.usage_by_pilot_code(filters)],
        usage_by_scenario=[_jsonable(r) for r in analytics.usage_by_scenario(filters)],
        usage_by_user_type=[_jsonable(r) for r in analytics.usage_by_user_type(filters)],
        usage_by_feature=[_jsonable(r) for r in analytics.usage_by_feature(filters)],
        simulations_per_user=[_jsonable(r) for r in analytics.simulations_per_user(filters)],
        assessment_changes=[_jsonable(r) for r in analytics.assessment_changes(filters)],
        feedback=[_jsonable(r) for r in analytics.feedback_rows(filters)],
        feedback_ratings=_jsonable(analytics.feedback_ratings_summary(filters)),
        bugs_and_requests=[_jsonable(r) for r in analytics.bug_and_feature_rows(filters)],
        environment=settings.environment,
        is_demo_environment=settings.is_demo_data,
        include_demo=include_demo,
        metric_definitions=METRIC_DEFINITIONS,
        warnings=settings.deployment_warnings(),
    )


@router.get(
    "/admin/organisations",
    summary="Registered organisations (administrator only)",
)
def admin_organisations(
    include_demo: Annotated[bool, Query()] = False,
    _admin: str = Depends(require_admin),
) -> list[dict[str, Any]]:
    return [_jsonable(row) for row in analytics.list_organisations(include_demo)]


@router.get(
    "/admin/export",
    summary="Every aggregate table as a ZIP of CSVs (administrator only)",
)
def admin_export(
    start: Annotated[str | None, Query()] = None,
    end: Annotated[str | None, Query()] = None,
    organisation_id: Annotated[int | None, Query()] = None,
    user_type: Annotated[str | None, Query()] = None,
    include_demo: Annotated[bool, Query()] = False,
    _admin: str = Depends(require_admin),
) -> Response:
    """Anonymous aggregate export, with the metric definitions alongside."""
    filters = _filters(start, end, organisation_id, user_type, include_demo)

    tables: dict[str, list[dict[str, Any]]] = {
        "summary": [analytics.summary_metrics(filters)],
        "usage_by_date": analytics.usage_by_date(filters),
        "usage_by_organisation": analytics.usage_by_organisation(filters),
        "usage_by_pilot_code": analytics.usage_by_pilot_code(filters),
        "usage_by_scenario": analytics.usage_by_scenario(filters),
        "usage_by_user_type": analytics.usage_by_user_type(filters),
        "usage_by_feature": analytics.usage_by_feature(filters),
        "simulations_per_user": analytics.simulations_per_user(filters),
        "assessment_changes": analytics.assessment_changes(filters),
        "feedback": analytics.feedback_rows(filters),
        "bugs_and_requests": analytics.bug_and_feature_rows(filters),
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, rows in tables.items():
            if not rows:
                continue
            text = io.StringIO()
            writer = csv.DictWriter(text, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _csv_value(v) for k, v in row.items()})
            archive.writestr(f"{name}.csv", text.getvalue())
        archive.writestr(
            "METRIC_DEFINITIONS.txt",
            _definitions_text(filters),
        )

    stamp = datetime.now(timezone.utc).date().isoformat()
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="pilot_export_{stamp}.zip"'
        },
    )


def _definitions_text(filters: ReportFilters) -> str:
    lines = [
        "Pilot metric definitions",
        "========================",
        "",
        f"Exported     : {datetime.now(timezone.utc).date().isoformat()}",
        f"Period       : {filters.start or 'all time'} to {filters.end or 'today'}",
        f"Organisation : {filters.organisation_id or 'all'}",
        f"User type    : {filters.user_type or 'all'}",
        f"Demo records : "
        f"{'INCLUDED - NOT PILOT DATA' if filters.include_demo else 'excluded'}",
        "",
    ]
    for key, definition in METRIC_DEFINITIONS.items():
        lines.append(f"{key}")
        lines.append(f"    {definition}")
        lines.append("")
    lines += [
        "All figures are anonymous and aggregate. No participant can be identified",
        "from this export: the database stores only a one-way hash of the",
        "participant code, and holds no name, email address, age, IP address or",
        "browser information.",
        "",
    ]
    return "\n".join(lines)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _jsonable(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (date, datetime)):
            out[key] = value.isoformat()
        elif value is None or isinstance(value, (str, int, float, bool)):
            out[key] = value
        else:
            out[key] = json.loads(json.dumps(value, default=str))
    return out
