from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class NumberFormat(str, Enum):
    AUTO = "auto"
    FIXED = "fixed"
    SIGNIFICANT = "significant"
    SCIENTIFIC = "scientific"
    PERCENT = "percent"
    DURATION = "duration"
    DISTANCE = "distance"
    VELOCITY = "velocity"
    ANGLE = "angle"
    INTEGER = "integer"


class Severity(str, Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class Metric(StrictModel):
    key: str
    label: str
    value: float | int | str | None
    unit: str = ""
    fmt: NumberFormat = NumberFormat.AUTO
    precision: int | None = None
    help: str | None = None
    highlight: bool = False


class Interpretation(StrictModel):
    severity: Severity = Severity.INFO
    text: str
    title: str | None = None


class PlotKind(str, Enum):
    LINE = "line"
    SCATTER_3D = "scatter3d"
    ORBIT_3D = "orbit3d"
    HEATMAP = "heatmap"
    DUAL_AXIS = "dual_axis"
    LOG_LOG = "log_log"


class SeriesSpec(StrictModel):
    key: str
    label: str
    x_key: str | None = None
    y_key: str
    z_key: str | None = None
    unit: str = ""
    color_index: int = 0
    dash: Literal["solid", "dash", "dot", "dashdot"] = "solid"
    secondary_y: bool = False
    width: float = 2.5
    mode: Literal["lines", "markers", "lines+markers"] = "lines"


class AxisSpec(StrictModel):
    title: str
    unit: str = ""
    log: bool = False


class PlotSpec(StrictModel):
    id: str
    title: str
    kind: PlotKind
    x: AxisSpec
    y: AxisSpec
    y2: AxisSpec | None = None
    z: AxisSpec | None = None
    x_key: str | None = None
    series: list[SeriesSpec] = Field(default_factory=list)
    caption: str | None = None
    show_earth: bool = False
    show_target_marker: bool = False
    keep_out_radius_m: float | None = None
    reference_line: dict[str, Any] | None = None
    matrix: dict[str, Any] | None = None
    height: int = 380


class ColumnSpec(StrictModel):
    key: str
    label: str
    unit: str = ""
    fmt: NumberFormat = NumberFormat.AUTO
    precision: int | None = None
    numeric: bool = True


class TableSpec(StrictModel):
    id: str
    label: str
    columns: list[ColumnSpec]
    rows: list[dict[str, Any]]
    caption: str | None = None
    downloadable: bool = True


class SolverInfo(StrictModel):
    integrator: str = "fixed-step classical Runge-Kutta 4"
    deterministic: bool = True
    random_seed: int | None = None
    force_model: str | None = None
    model: str | None = None
    n_steps: int | None = None
    dt_s: float | None = None
    duration_s: float | None = None
    engine_version: str
    units: str = "SI (m, m/s, s, kg, rad)"
    frames: str = (
        "ECI for absolute states; LVLH (radial, along-track, cross-track) "
        "for relative states"
    )


class ProvenanceInfo(StrictModel):
    software: str = "Orbital Rendezvous Simulator"
    external_validation: str = (
        "This engine has NOT been benchmarked against GMAT, STK or any other "
        "external tool. It is not NASA-approved, NASA-validated, "
        "flight-certified or operational mission software."
    )
    licence: str = "MIT"


class FieldError(StrictModel):
    field: str | None = None
    message: str


class ErrorResponse(StrictModel):
    status: Literal["error"] = "error"
    error_class: str
    message: str
    detail: list[FieldError] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rejected: bool = False
    retry_after_s: float | None = None
