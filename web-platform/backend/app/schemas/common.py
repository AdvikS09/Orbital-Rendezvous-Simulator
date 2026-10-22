"""Shared Pydantic models.

These types are the contract between the Python engine and the React frontend.
Two principles shape them:

1. **The backend returns numbers; the frontend decides presentation.** No
   Plotly figure is ever built in Python. Endpoints return named numeric arrays
   plus plot *metadata* (axis labels, units, which series belong on which
   axis), and React constructs the chart. That keeps presentation decisions on
   the presentation side and keeps the payload small enough to inspect.

2. **A number is never sent without its unit.** Every metric, every axis and
   every parameter carries an explicit unit string, so the interface cannot
   silently mislabel a quantity.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that rejects unknown fields, so contract drift fails loudly."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# Presentation hints
class NumberFormat(str, Enum):
    """How the frontend should render a value.

    The frontend owns the actual formatting (see src/formatting/number.ts);
    this only says which of its modes applies. ``SCIENTIFIC`` exists because
    values such as an energy drift of 1.1e-13 are meaningless in fixed
    notation, and ``AUTO`` covers the ordinary case.
    """

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
    """One scalar output value, ready to display."""

    key: str
    label: str
    value: float | int | str | None
    unit: str = ""
    fmt: NumberFormat = NumberFormat.AUTO
    #: Significant figures or decimal places, depending on ``fmt``.
    precision: int | None = None
    help: str | None = None
    #: Highlighted metrics are shown in the headline row.
    highlight: bool = False


class Interpretation(StrictModel):
    """A plain-English reading of the result.

    Carried over verbatim from the Streamlit renderers, including the branching
    logic that chooses between messages.
    """

    severity: Severity = Severity.INFO
    text: str
    title: str | None = None


# Plot metadata
class PlotKind(str, Enum):
    LINE = "line"
    SCATTER_3D = "scatter3d"
    ORBIT_3D = "orbit3d"
    HEATMAP = "heatmap"
    DUAL_AXIS = "dual_axis"
    LOG_LOG = "log_log"


class SeriesSpec(StrictModel):
    """One trace on a plot, referring to arrays by key in ``SimulationResult.series``."""

    key: str
    label: str
    #: Key into ``series`` for this trace's x values. Falls back to the plot's x_key.
    x_key: str | None = None
    y_key: str
    z_key: str | None = None
    unit: str = ""
    #: Index into the Okabe-Ito palette; keeps colours consistent across scenarios.
    color_index: int = 0
    dash: Literal["solid", "dash", "dot", "dashdot"] = "solid"
    #: Put this trace on the secondary y axis.
    secondary_y: bool = False
    width: float = 2.5
    mode: Literal["lines", "markers", "lines+markers"] = "lines"


class AxisSpec(StrictModel):
    title: str
    unit: str = ""
    log: bool = False


class PlotSpec(StrictModel):
    """Everything React needs to build one chart, with no physics involved."""

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
    #: Draw Earth to scale (orbit plots only).
    show_earth: bool = False
    #: Mark the LVLH origin as the target (relative-motion plots only).
    show_target_marker: bool = False
    #: Radius in metres of a keep-out sphere to draw, if any.
    keep_out_radius_m: float | None = None
    #: Reference line the measured data should be compared against.
    reference_line: dict[str, Any] | None = None
    #: Heatmap payload: {"z": [[...]], "x": [...], "y": [...]}.
    matrix: dict[str, Any] | None = None
    height: int = 380


# Tables
class ColumnSpec(StrictModel):
    key: str
    label: str
    unit: str = ""
    fmt: NumberFormat = NumberFormat.AUTO
    precision: int | None = None
    numeric: bool = True


class TableSpec(StrictModel):
    """A downloadable table. ``rows`` holds raw values; the frontend formats them."""

    id: str
    label: str
    columns: list[ColumnSpec]
    rows: list[dict[str, Any]]
    caption: str | None = None
    downloadable: bool = True


# Provenance
class SolverInfo(StrictModel):
    """What actually computed the answer. Shipped with every result and export."""

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
    """The honest statement about what this software is, carried into exports."""

    software: str = "Orbital Rendezvous Simulator"
    external_validation: str = (
        "This engine has NOT been benchmarked against GMAT, STK or any other "
        "external tool. It is not NASA-approved, NASA-validated, "
        "flight-certified or operational mission software."
    )
    licence: str = "MIT"


# Errors
class FieldError(StrictModel):
    """A validation failure attributable to one input, so the UI can mark it."""

    field: str | None = None
    message: str


class ErrorResponse(StrictModel):
    """Structured failure. Never a traceback."""

    status: Literal["error"] = "error"
    error_class: str
    message: str
    detail: list[FieldError] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    #: True when the request was blocked before the engine ran.
    rejected: bool = False
    retry_after_s: float | None = None
