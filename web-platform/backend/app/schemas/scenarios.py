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


class ParamKind(str):
    NUMBER = "number"
    SLIDER = "slider"
    CHECKBOX = "checkbox"
    SELECT = "select"


class ParamSpec(StrictModel):
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
    advanced: bool = False
    fmt: NumberFormat = NumberFormat.AUTO
    precision: int | None = None
    hard_minimum: float | None = None
    hard_maximum: float | None = None


class ScenarioSummary(StrictModel):
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

    estimated_steps: int | None = None
    estimated_seconds: float | None = None


class SimulationRequest(StrictModel):
    params: dict[str, Any] = Field(default_factory=dict)
    session_token: str | None = None
    activity_id: str | None = None
    step_id: str | None = None


class InputEcho(StrictModel):
    name: str
    label: str
    value: Any
    unit: str = ""


class SimulationResult(StrictModel):
    status: Literal["ok"] = "ok"
    scenario_id: str
    scenario_title: str
    result_kind: str

    inputs: list[InputEcho]
    derived: list[Metric] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)

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
    completed_at: str

    extras: dict[str, Any] = Field(default_factory=dict)


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
    warnings: list[str] = Field(default_factory=list)
