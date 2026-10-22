"""Scenario description and simulation request/response models.

The scenario payload carries the *whole* of a scenario's educational content --
summary, multi-paragraph explanation, reading guide, assumptions, limitations,
references -- not just its parameters. That is deliberate: in the Streamlit
version this text sat next to the controls, and a migration that dropped it
would leave students with numbers and no qualifications on them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .common import (
    Interpretation,
    Metric,
    NumberFormat,
    PlotSpec,
    ProvenanceInfo,
    SolverInfo,
    StrictModel,
    TableSpec,
)


# Parameters
class ParamKind(str):
    NUMBER = "number"
    SLIDER = "slider"
    CHECKBOX = "checkbox"
    SELECT = "select"


class ParamSpec(StrictModel):
    """One user-editable input, mirroring pilot.content.scenarios_catalog.Param."""

    name: str
    label: str
    kind: Literal["number", "slider", "checkbox", "select"]
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    unit: str = ""
    help: str = ""
    options: list[Any] = Field(default_factory=list)
    #: Hidden behind an "Advanced" disclosure so newcomers are not overwhelmed.
    advanced: bool = False
    #: How the frontend should render the current value.
    fmt: NumberFormat = NumberFormat.AUTO
    precision: int | None = None
    #: Bounds enforced by pilot.validation, which are tighter than the widget's
    #: own range for some fields. Shown as help text, enforced on both sides.
    hard_minimum: float | None = None
    hard_maximum: float | None = None


class ScenarioSummary(StrictModel):
    """Listing entry, for the scenario selector."""

    id: str
    title: str
    category: str
    category_label: str
    feature_category: str
    summary: str
    levels: list[str]
    result_kind: str
    model: str
    param_count: int


class ScenarioDetail(StrictModel):
    """Full scenario, including every word of its educational content."""

    id: str
    title: str
    category: str
    category_label: str
    feature_category: str
    model: str
    result_kind: str
    levels: list[str]

    summary: str
    explanation: str
    reading_guide: str = ""
    references: list[str] = Field(default_factory=list)

    assumptions: list[str]
    limitations: list[str]

    params: list[ParamSpec]
    defaults: dict[str, Any]

    #: Estimated cost at the default parameters, so the UI can warn before a run.
    estimated_steps: int | None = None
    estimated_seconds: float | None = None


# Simulation
class SimulationRequest(StrictModel):
    """Parameter values for one run.

    ``params`` is deliberately loose (``dict[str, Any]``) because each scenario
    declares its own parameter set; the values are validated against that
    scenario's ParamSpec list and then against pilot.validation, which owns the
    physical bounds.
    """

    params: dict[str, Any] = Field(default_factory=dict)
    #: Anonymous session token from POST /api/v1/sessions. Optional: the
    #: simulator works without analytics.
    session_token: str | None = None
    #: Set when the run comes from a guided activity step, for analytics only.
    activity_id: str | None = None
    step_id: str | None = None


class InputEcho(StrictModel):
    """The inputs as the engine actually used them, with units.

    Echoed back so a result is self-describing: a downloaded CSV or a
    screenshot can always be traced to the exact inputs that produced it.
    """

    name: str
    label: str
    value: Any
    unit: str = ""


class SimulationResult(StrictModel):
    """A completed run.

    ``series`` holds the raw numeric arrays. ``plots`` describes how to draw
    them. The frontend never recomputes a trajectory -- it plots exactly what
    the Python engine returned.
    """

    status: Literal["ok"] = "ok"
    scenario_id: str
    scenario_title: str
    result_kind: str

    inputs: list[InputEcho]
    derived: list[Metric] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)

    #: Named numeric arrays: "t_s", "x_m", "pos_error_m", and so on.
    series: dict[str, list[float | None]] = Field(default_factory=dict)
    plots: list[PlotSpec] = Field(default_factory=list)
    tables: list[TableSpec] = Field(default_factory=list)

    interpretations: list[Interpretation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    reading_guide: str = ""

    solver: SolverInfo
    provenance: ProvenanceInfo = Field(default_factory=ProvenanceInfo)

    runtime_ms: int
    n_steps: int = 0
    #: ISO-8601 UTC timestamp of completion.
    completed_at: str

    #: Free-form payload for scenario-specific extras (for example the CSV and
    #: JSON text of an external-validation export).
    extras: dict[str, Any] = Field(default_factory=dict)


# Health
class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    engine_version: str
    api_version: str
    app_version: str
    environment: str
    database_ok: bool
    database: str
    schema_version: int
    scenario_count: int
    #: Configuration problems worth surfacing on a deployment.
    warnings: list[str] = Field(default_factory=list)
