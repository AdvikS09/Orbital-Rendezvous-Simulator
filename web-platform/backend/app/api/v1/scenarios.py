"""Scenario catalogue and simulation endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from app.api.deps import optional_participant
from app.schemas.common import ErrorResponse
from app.schemas.scenarios import (
    ScenarioDetail,
    ScenarioSummary,
    SimulationRequest,
    SimulationResult,
)
from app.services import catalogue, simulation

router = APIRouter(tags=["scenarios"])


@router.get(
    "/scenarios",
    response_model=list[ScenarioSummary],
    summary="List every available scenario",
)
def list_scenarios(
    track: Annotated[
        str | None,
        Query(description="Filter to one track: guided, professional or free."),
    ] = None,
) -> list[ScenarioSummary]:
    if track and track not in ("guided", "professional", "free"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown track {track!r}. Expected guided, professional or free.",
        )
    return catalogue.list_scenarios(track)


@router.get(
    "/scenarios/categories",
    response_model=dict[str, str],
    summary="Category identifiers and their display labels",
)
def list_categories() -> dict[str, str]:
    return catalogue.category_labels()


@router.get(
    "/scenarios/{scenario_id}",
    response_model=ScenarioDetail,
    summary="One scenario, with its full explanatory content",
    responses={404: {"model": ErrorResponse}},
)
def get_scenario(scenario_id: str) -> ScenarioDetail:
    scenario = catalogue.find(scenario_id)
    if scenario is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No scenario with id {scenario_id!r}.",
        )
    return catalogue.to_detail(scenario)


@router.post(
    "/scenarios/{scenario_id}/preflight",
    summary="Validate parameters and estimate cost without running",
    responses={404: {"model": ErrorResponse}},
)
def preflight(
    scenario_id: str,
    payload: Annotated[SimulationRequest, Body()],
) -> dict[str, Any]:
    """Cheap check so the interface can warn before a pointless round-trip."""
    scenario = catalogue.find(scenario_id)
    if scenario is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No scenario with id {scenario_id!r}.",
        )
    return simulation.preflight(scenario, payload.params)


@router.post(
    "/simulations/{scenario_id}",
    response_model=SimulationResult,
    summary="Run a simulation and return its numerical results",
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def run_simulation(
    scenario_id: str,
    payload: Annotated[SimulationRequest, Body()],
    context=Depends(optional_participant),
) -> Any:
    """Execute one scenario.

    The response carries the inputs with units, the derived scalars, the raw
    numeric arrays, plot metadata, tables, warnings, interpretation text, solver
    information and the runtime. The frontend renders it; it does not
    recalculate anything.
    """
    scenario = catalogue.find(scenario_id)
    if scenario is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No scenario with id {scenario_id!r}.",
        )

    result, error = simulation.run(scenario, payload.params, context)

    if error is not None:
        # Returned as an explicit JSONResponse rather than by setting a status
        # code on the injected Response: `response_model=SimulationResult`
        # applies to anything returned normally, and would reject the error
        # payload as malformed. A JSONResponse bypasses that validation.
        if error.error_class == "RateLimited":
            code = status.HTTP_429_TOO_MANY_REQUESTS
        elif error.rejected:
            code = status.HTTP_422_UNPROCESSABLE_ENTITY
        else:
            code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return JSONResponse(status_code=code, content=error.model_dump())

    return result
