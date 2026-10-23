from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

import orbital_core
from app.schemas.common import (
    AxisSpec,
    ColumnSpec,
    Interpretation,
    Metric,
    NumberFormat,
    PlotKind,
    PlotSpec,
    SeriesSpec,
    Severity,
    SolverInfo,
    TableSpec,
)
from app.schemas.scenarios import InputEcho, SimulationResult
from app.services.formatting import format_scientific

BLUE, ORANGE, GREEN, PINK, AMBER, SKY, BLACK = range(7)


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if (math.isnan(number) or math.isinf(number)) else number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.str_, str)):
        return str(value)
    if pd.isna(value) if np.isscalar(value) else False:
        return None
    return value


def _array(values: Any) -> list[float | None]:
    return [_clean(v) for v in np.asarray(values).ravel()]


def _row_to_dict(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or len(frame) == 0:
        return {}
    return {key: _clean(value) for key, value in frame.iloc[0].items()}


def _table_from_frame(
    table_id: str,
    label: str,
    frame: pd.DataFrame,
    units: dict[str, str] | None = None,
    caption: str | None = None,
) -> TableSpec:
    units = units or {}
    columns = [
        ColumnSpec(
            key=str(name),
            label=str(name),
            unit=units.get(str(name), _unit_from_column(str(name))),
            fmt=NumberFormat.AUTO,
            numeric=pd.api.types.is_numeric_dtype(frame[name]),
        )
        for name in frame.columns
    ]
    rows = [{str(k): _clean(v) for k, v in record.items()} for record in frame.to_dict("records")]
    return TableSpec(id=table_id, label=label, columns=columns, rows=rows, caption=caption)


def _unit_from_column(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith("_mps2"):
        return "m/s²"
    if lowered.endswith("_mps"):
        return "m/s"
    if lowered.endswith("_km"):
        return "km"
    if lowered.endswith("_m"):
        return "m"
    if lowered.endswith("_s"):
        return "s"
    if lowered.endswith("_rel") or lowered.endswith("_ratio"):
        return ""
    if lowered.endswith("_deg"):
        return "deg"
    return ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_orbit(result: dict, series: dict, plots: list, metrics: list,
                 tables: list, interpretations: list) -> None:
    validation = _row_to_dict(result["validation"])
    period_s = float(result["period_s"])

    series["t_s"] = _array(result["times"])
    states = np.asarray(result["states"])
    for index, key in enumerate(("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")):
        series[key] = _array(states[:, index])
    series["altitude_km"] = _array(result["altitude_km"])

    metrics.extend([
        Metric(key="period_s", label="Orbital period", value=period_s, unit="s",
               fmt=NumberFormat.DURATION, highlight=True,
               help="Time for one complete revolution."),
        Metric(key="max_energy_drift_rel", label="Energy drift (relative)",
               value=validation.get("max_energy_drift_rel"), unit="",
               fmt=NumberFormat.SCIENTIFIC, precision=3, highlight=True,
               help="How much the total orbital energy wandered. Physics says it "
                    "should be exactly zero, so this number is a measure of the "
                    "integrator's accuracy."),
        Metric(key="period_rel_err", label="Period error (relative)",
               value=validation.get("period_rel_err"), unit="",
               fmt=NumberFormat.SCIENTIFIC, precision=3, highlight=True,
               help="Measured period against the analytic formula."),
        Metric(key="max_ang_mom_drift_rel", label="Angular-momentum drift (relative)",
               value=validation.get("max_ang_mom_drift_rel"), unit="",
               fmt=NumberFormat.SCIENTIFIC, precision=3),
        Metric(key="measured_period_s", label="Measured period",
               value=validation.get("measured_period_s"), unit="s",
               fmt=NumberFormat.FIXED, precision=2),
        Metric(key="analytic_period_s", label="Analytic period",
               value=validation.get("analytic_period_s"), unit="s",
               fmt=NumberFormat.FIXED, precision=2),
        Metric(key="radius_m", label="Orbit radius", value=_clean(result["radius_m"]),
               unit="m", fmt=NumberFormat.DISTANCE),
    ])

    plots.append(PlotSpec(
        id="orbit_3d", title="Orbit (Earth-centred inertial frame)",
        kind=PlotKind.ORBIT_3D, show_earth=True, height=520,
        x=AxisSpec(title="x", unit="km"), y=AxisSpec(title="y", unit="km"),
        z=AxisSpec(title="z", unit="km"),
        series=[SeriesSpec(key="orbit", label="orbit", x_key="x_m", y_key="y_m",
                           z_key="z_m", unit="m", color_index=ORANGE, width=4)],
        caption="The translucent sphere is Earth, drawn to scale.",
    ))
    plots.append(PlotSpec(
        id="altitude", title="Altitude against time", kind=PlotKind.LINE,
        x=AxisSpec(title="time", unit="s"), y=AxisSpec(title="altitude", unit="km"),
        x_key="t_s",
        series=[SeriesSpec(key="altitude", label="altitude", y_key="altitude_km",
                           unit="km", color_index=BLUE)],
    ))

    tables.append(_table_from_frame("orbit_validation", "Orbit validation",
                                    result["validation"]))
    tables.append(_table_from_frame("orbit_timeseries", "Orbit time series", pd.DataFrame({
        "t_s": result["times"], "x_m": states[:, 0], "y_m": states[:, 1],
        "z_m": states[:, 2], "vx_mps": states[:, 3], "vy_mps": states[:, 4],
        "vz_mps": states[:, 5], "altitude_km": result["altitude_km"],
    })))

    drift = validation.get("max_energy_drift_rel")
    if drift and drift < 1e-8:
        digits = abs(int(math.log10(drift)))
        interpretations.append(Interpretation(
            severity=Severity.INFO,
            text=(f"The integrator conserved energy to about "
                  f"{format_scientific(drift, 2)} — roughly {digits} digits of "
                  "agreement with theory. It is not a meaningful source of error "
                  "here."),
        ))
    elif drift:
        interpretations.append(Interpretation(
            severity=Severity.WARNING,
            text=(f"Energy drifted by about {format_scientific(drift, 2)}, which is "
                  f"larger than the {format_scientific(1e-13, 1)} seen at the "
                  "published step sizes. Consider a finer timestep."),
        ))
    else:
        interpretations.append(Interpretation(
            severity=Severity.WARNING,
            text="Energy drift could not be evaluated for this run.",
        ))


def _build_orbit_comparison(result: dict, series: dict, plots: list, metrics: list,
                            tables: list, interpretations: list) -> None:
    separation = np.asarray(result["separation_m"])
    perturbed_altitude = np.asarray(result["perturbed"]["altitude_km"])
    baseline_altitude = np.asarray(result["baseline"]["altitude_km"])
    altitude_change = float(perturbed_altitude[0] - perturbed_altitude[-1])

    series["t_s"] = _array(result["times"])
    series["separation_m"] = _array(separation)
    series["separation_km"] = _array(separation / 1000.0)
    series["perturbed_altitude_km"] = _array(perturbed_altitude)
    series["baseline_altitude_km"] = _array(baseline_altitude)

    perturbed_states = np.asarray(result["perturbed"]["states"])
    for index, key in enumerate(("x_m", "y_m", "z_m")):
        series[key] = _array(perturbed_states[:, index])

    metrics.extend([
        Metric(key="max_separation_m", label="Maximum divergence from the ideal orbit",
               value=float(separation.max()), unit="m", fmt=NumberFormat.DISTANCE,
               highlight=True,
               help="Greatest distance between the perturbed trajectory and the "
                    "unperturbed one."),
        Metric(key="final_separation_m", label="Divergence at the end",
               value=float(separation[-1]), unit="m", fmt=NumberFormat.DISTANCE,
               highlight=True),
        Metric(key="altitude_change_m", label="Altitude change",
               value=altitude_change * 1000.0, unit="m", fmt=NumberFormat.FIXED,
               precision=0, highlight=True,
               help="Positive means the orbit lost altitude over the run."),
        Metric(key="period_s", label="Orbital period",
               value=float(result["perturbed"]["period_s"]), unit="s",
               fmt=NumberFormat.DURATION),
    ])

    plots.append(PlotSpec(
        id="separation", title="How far the perturbed orbit drifts from the ideal one",
        kind=PlotKind.LINE, x_key="t_s",
        x=AxisSpec(title="time", unit="s"), y=AxisSpec(title="separation", unit="km"),
        series=[SeriesSpec(key="sep", label="separation", y_key="separation_km",
                           unit="km", color_index=PINK)],
    ))
    plots.append(PlotSpec(
        id="altitudes", title="Altitude against time", kind=PlotKind.LINE, x_key="t_s",
        x=AxisSpec(title="time", unit="s"), y=AxisSpec(title="altitude", unit="km"),
        series=[
            SeriesSpec(key="baseline", label="ideal two-body",
                       y_key="baseline_altitude_km", unit="km", color_index=BLUE),
            SeriesSpec(key="perturbed", label="perturbed",
                       y_key="perturbed_altitude_km", unit="km",
                       color_index=ORANGE, dash="dash"),
        ],
    ))
    plots.append(PlotSpec(
        id="orbit_3d", title="Perturbed orbit", kind=PlotKind.ORBIT_3D,
        show_earth=True, height=520,
        x=AxisSpec(title="x", unit="km"), y=AxisSpec(title="y", unit="km"),
        z=AxisSpec(title="z", unit="km"),
        series=[SeriesSpec(key="orbit", label="perturbed orbit", x_key="x_m",
                           y_key="y_m", z_key="z_m", unit="m",
                           color_index=ORANGE, width=4)],
    ))

    tables.append(_table_from_frame("perturbation_effect", "Perturbation effect",
                                    pd.DataFrame({
                                        "t_s": result["times"],
                                        "separation_m": separation,
                                        "baseline_altitude_km": baseline_altitude,
                                        "perturbed_altitude_km": perturbed_altitude,
                                    })))

    if separation.max() > 1.0 and np.all(np.diff(separation) >= -1e-6):
        interpretations.append(Interpretation(
            severity=Severity.INFO,
            text=("The divergence grows steadily rather than oscillating about zero. "
                  "That steady growth is what makes a perturbation *secular*: it "
                  "accumulates orbit after orbit instead of averaging out."),
        ))


def _relative_series(prefix: str, bundle: dict, series: dict) -> None:
    states = np.asarray(bundle["states"])
    controls = np.asarray(bundle["controls"])
    for index, key in enumerate(("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")):
        series[f"{prefix}{key}"] = _array(states[:, index])
    for index, key in enumerate(("ax_mps2", "ay_mps2", "az_mps2")):
        series[f"{prefix}{key}"] = _array(controls[:, index])
    series[f"{prefix}ctrl_mag_mps2"] = _array(bundle["control_mag"])
    series[f"{prefix}pos_err_m"] = _array(bundle["pos_error"])
    series[f"{prefix}vel_err_mps"] = _array(bundle["vel_error"])


def _relative_metrics(bundle: dict) -> list[Metric]:
    row = _row_to_dict(bundle["metrics"])
    return [
        Metric(key="initial_pos_error_m", label="Starting distance",
               value=row.get("initial_pos_error_m"), unit="m",
               fmt=NumberFormat.DISTANCE, highlight=True),
        Metric(key="final_pos_error_m", label="Final distance",
               value=row.get("final_pos_error_m"), unit="m",
               fmt=NumberFormat.DISTANCE, highlight=True),
        Metric(key="convergence_time_s", label="Time to converge",
               value=row.get("convergence_time_s"), unit="s",
               fmt=NumberFormat.DURATION, highlight=True,
               help="Time after which the distance stayed within 1 m for the rest "
                    "of the run. A dash means it never did."),
        Metric(key="delta_v_mps", label="Delta-v (propellant cost)",
               value=row.get("delta_v_mps"), unit="m/s",
               fmt=NumberFormat.VELOCITY, highlight=True,
               help="The integral of thrust acceleration over time. This is what "
                    "converts into propellant mass."),
        Metric(key="final_vel_error_mps", label="Final velocity error",
               value=row.get("final_vel_error_mps"), unit="m/s",
               fmt=NumberFormat.VELOCITY),
        Metric(key="max_control_accel_mps2", label="Peak thrust acceleration",
               value=row.get("max_control_accel_mps2"), unit="m/s²",
               fmt=NumberFormat.AUTO),
    ]


def _relative_plots(prefix: str, label: str, controlled: bool) -> list[PlotSpec]:
    plots = [
        PlotSpec(
            id=f"{prefix}trajectory",
            title="Relative trajectory (target-centred LVLH frame)",
            kind=PlotKind.SCATTER_3D, show_target_marker=True, height=520,
            x=AxisSpec(title="radial x", unit="m"),
            y=AxisSpec(title="along-track y", unit="m"),
            z=AxisSpec(title="cross-track z", unit="m"),
            series=[SeriesSpec(key=f"{prefix}traj", label=label,
                               x_key=f"{prefix}x_m", y_key=f"{prefix}y_m",
                               z_key=f"{prefix}z_m", unit="m",
                               color_index=BLUE, width=5)],
            caption="The green diamond at the origin is the target. Axes are the "
                    "target-centred LVLH frame.",
        ),
        PlotSpec(
            id=f"{prefix}position", title="Relative position", kind=PlotKind.LINE,
            x_key="t_s", x=AxisSpec(title="time", unit="s"),
            y=AxisSpec(title="position", unit="m"),
            series=[
                SeriesSpec(key="rx", label="radial x", y_key=f"{prefix}x_m",
                           unit="m", color_index=BLUE),
                SeriesSpec(key="ry", label="along-track y", y_key=f"{prefix}y_m",
                           unit="m", color_index=ORANGE, dash="dash"),
                SeriesSpec(key="rz", label="cross-track z", y_key=f"{prefix}z_m",
                           unit="m", color_index=GREEN, dash="dot"),
            ],
        ),
        PlotSpec(
            id=f"{prefix}velocity", title="Relative velocity", kind=PlotKind.LINE,
            x_key="t_s", x=AxisSpec(title="time", unit="s"),
            y=AxisSpec(title="velocity", unit="m/s"),
            series=[
                SeriesSpec(key="vx", label="radial", y_key=f"{prefix}vx_mps",
                           unit="m/s", color_index=BLUE),
                SeriesSpec(key="vy", label="along-track", y_key=f"{prefix}vy_mps",
                           unit="m/s", color_index=ORANGE, dash="dash"),
                SeriesSpec(key="vz", label="cross-track", y_key=f"{prefix}vz_mps",
                           unit="m/s", color_index=GREEN, dash="dot"),
            ],
        ),
    ]
    if controlled:
        plots.append(PlotSpec(
            id=f"{prefix}control", title="Commanded thrust acceleration",
            kind=PlotKind.LINE, x_key="t_s",
            x=AxisSpec(title="time", unit="s"),
            y=AxisSpec(title="acceleration", unit="m/s²"),
            series=[
                SeriesSpec(key="ax", label="radial a_x", y_key=f"{prefix}ax_mps2",
                           unit="m/s²", color_index=BLUE),
                SeriesSpec(key="ay", label="along-track a_y", y_key=f"{prefix}ay_mps2",
                           unit="m/s²", color_index=ORANGE, dash="dash"),
                SeriesSpec(key="az", label="cross-track a_z", y_key=f"{prefix}az_mps2",
                           unit="m/s²", color_index=GREEN, dash="dot"),
                SeriesSpec(key="mag", label="magnitude |u|",
                           y_key=f"{prefix}ctrl_mag_mps2", unit="m/s²",
                           color_index=BLACK, width=3.5),
            ],
            caption="The area under the |u| curve is the delta-v.",
        ))
    return plots


def _relative_interpretation(bundle: dict, controlled: bool) -> list[Interpretation]:
    from app.services.formatting import display_metric

    row = _row_to_dict(bundle["metrics"])
    converged = row.get("convergence_time_s") is not None

    if controlled:
        peak = row.get("max_control_accel_mps2") or 0.0
        if converged:
            return [Interpretation(
                severity=Severity.SUCCESS,
                text=(f"The chaser converged after {row['convergence_time_s']:.0f} s, "
                      f"finishing {display_metric(row.get('final_pos_error_m'), 'm')} "
                      f"from the target for "
                      f"{display_metric(row.get('delta_v_mps'), 'm/s')} of delta-v. "
                      f"Peak thrust was {peak:.4g} m/s²."),
            )]
        return [Interpretation(
            severity=Severity.WARNING,
            text=("The chaser never settled within 1 m and stayed there. That is a "
                  "real result, not an error: the gains may be too weak for this "
                  "starting offset, or the run may simply be too short. Try a "
                  "higher bandwidth or a longer duration."),
        )]

    states = np.asarray(bundle["states"])
    start, end = states[0, :3], states[-1, :3]
    drift = float(np.linalg.norm(end - start))
    along_track = float(end[1] - start[1])
    radial = float(end[0] - start[0])

    message = (f"With no thrust at all, the chaser moved {drift:.0f} m over the run "
               f"({along_track:+.0f} m of that along-track).")
    if drift < 1.0:
        message += (" It effectively stayed put — you have found one of the natural "
                    "equilibria of relative motion.")
    elif abs(along_track) > 5 * abs(radial) + 1.0:
        message += (" The motion is dominated by along-track drift, which is the "
                    "signature of a period mismatch between the two orbits.")
    return [Interpretation(severity=Severity.INFO, text=message)]


def _build_relative(result: dict, series: dict, plots: list, metrics: list,
                    tables: list, interpretations: list) -> None:
    series["t_s"] = _array(result["times"])
    _relative_series("", result, series)
    controlled = float(np.max(np.abs(np.asarray(result["controls"])))) > 0.0

    metrics.extend(_relative_metrics(result))
    plots.extend(_relative_plots("", "chaser", controlled))
    tables.append(_table_from_frame("relative_timeseries", "Relative time series",
                                    result["history"]))
    tables.append(_table_from_frame("metrics", "Rendezvous metrics", result["metrics"]))
    interpretations.extend(_relative_interpretation(result, controlled))


def _build_gain_sweep(result: dict, series: dict, plots: list, metrics: list,
                      tables: list, interpretations: list) -> None:
    table: pd.DataFrame = result["table"]
    kp_values = [float(v) for v in result["kp_values"]]
    kd_values = [float(v) for v in result["kd_values"]]

    for metric_name, plot_id, title, colourbar in (
        ("convergence_time_s", "heat_convergence", "Time to converge", "seconds"),
        ("delta_v_mps", "heat_delta_v", "Propellant cost (delta-v)", "m/s"),
    ):
        grid = table.pivot(index="kd", columns="kp", values=metric_name)
        plots.append(PlotSpec(
            id=plot_id, title=title, kind=PlotKind.HEATMAP, height=400,
            x=AxisSpec(title="Kp (proportional gain)", unit="1/s²"),
            y=AxisSpec(title="Kd (derivative gain)", unit="1/s"),
            matrix={
                "z": [[_clean(v) for v in row] for row in grid.to_numpy()],
                "x": [f"{v:.1e}" for v in grid.columns],
                "y": [f"{v:.1e}" for v in grid.index],
                "colorbar_title": colourbar,
            },
        ))

    tables.append(_table_from_frame("gain_sweep", "Gain sweep", table))

    converged = table.dropna(subset=["convergence_time_s"])
    metrics.append(Metric(key="cells", label="Gain combinations tested",
                          value=int(len(table)), fmt=NumberFormat.INTEGER,
                          highlight=True))
    metrics.append(Metric(key="converged_cells", label="Combinations that converged",
                          value=int(len(converged)), fmt=NumberFormat.INTEGER,
                          highlight=True,
                          help="Blank convergence times are combinations that never "
                               "reached the 1 m tolerance within the run."))

    if converged.empty:
        interpretations.append(Interpretation(
            severity=Severity.WARNING,
            text=("No gain combination in this grid reached the 1 m tolerance within "
                  "the run. Try a longer duration or a larger starting offset."),
        ))
        return

    fastest = converged.loc[converged["convergence_time_s"].idxmin()]
    cheapest = converged.loc[converged["delta_v_mps"].idxmin()]
    non_converging = len(table) - len(converged)

    metrics.extend([
        Metric(key="fastest_time_s", label="Fastest convergence",
               value=float(fastest.convergence_time_s), unit="s",
               fmt=NumberFormat.DURATION, highlight=True),
        Metric(key="cheapest_delta_v_mps", label="Lowest delta-v",
               value=float(cheapest.delta_v_mps), unit="m/s",
               fmt=NumberFormat.VELOCITY, highlight=True),
    ])

    def gains(row) -> str:
        return (f"Kp = {format_scientific(float(row.kp), 2)}, "
                f"Kd = {format_scientific(float(row.kd), 3)}")

    text = (f"**Fastest:** {gains(fastest)} → "
            f"{fastest.convergence_time_s:.0f} s at {fastest.delta_v_mps:.2f} m/s of "
            f"delta-v.\n\n"
            f"**Cheapest:** {gains(cheapest)} → "
            f"{cheapest.convergence_time_s:.0f} s at {cheapest.delta_v_mps:.2f} m/s "
            f"of delta-v.\n\n")
    if fastest.name != cheapest.name:
        text += (f"They are not the same cell — that gap *is* the trade-off. The "
                 f"fastest option costs "
                 f"{fastest.delta_v_mps / cheapest.delta_v_mps:.1f}x the delta-v of "
                 f"the most efficient one.")
    else:
        text += "On this grid the same cell happens to be both fastest and cheapest."
    if non_converging:
        text += (f"\n\n{non_converging} of {len(table)} combinations never reached "
                 "the 1 m tolerance and are shown blank. They are reported, not "
                 "hidden.")
    interpretations.append(Interpretation(severity=Severity.INFO, text=text))

    series["kp_values"] = [_clean(v) for v in kp_values]
    series["kd_values"] = [_clean(v) for v in kd_values]


def _build_comparison(result: dict, series: dict, plots: list, metrics: list,
                      tables: list, interpretations: list) -> None:
    series["t_s"] = _array(result["times"])
    controlled = bool(result.get("controlled"))

    if "cw" in result:
        _relative_series("cw_", result["cw"], series)
    if "high_fidelity" in result:
        _relative_series("hf_", result["high_fidelity"], series)

    primary = result.get("primary")
    if primary is not None:
        metrics.extend(_relative_metrics(primary))
        tables.append(_table_from_frame("relative_timeseries",
                                        "Relative time series (reference model)",
                                        primary["history"]))
        tables.append(_table_from_frame("metrics", "Rendezvous metrics",
                                        primary["metrics"]))

    if "comparison" not in result:
        prefix = "cw_" if "cw" in result else "hf_"
        label = "CW (linear model)" if prefix == "cw_" else "non-linear reference"
        plots.extend(_relative_plots(prefix, label, controlled))
        if primary is not None:
            interpretations.extend(_relative_interpretation(primary, controlled))
        return

    comparison = result["comparison"]
    summary = _row_to_dict(comparison["summary"])
    series["cmp_pos_error_m"] = _array(comparison["pos_error"])
    series["cmp_vel_error_mps"] = _array(comparison["vel_error"])

    metrics.insert(0, Metric(
        key="model_gap_max_m", label="Largest gap between the models",
        value=summary.get("max_pos_error_m"), unit="m", fmt=NumberFormat.DISTANCE,
        highlight=True,
        help="Greatest distance between the linear CW prediction and the non-linear "
             "reference at any point in the run."))
    metrics.insert(1, Metric(
        key="model_gap_final_m", label="Gap at the end",
        value=summary.get("final_pos_error_m"), unit="m", fmt=NumberFormat.DISTANCE,
        highlight=True,
        help="Distance between the two models' predictions at the end of the run."))
    metrics.insert(2, Metric(
        key="model_gap_max_vel_mps", label="Largest velocity gap",
        value=summary.get("max_vel_error_mps"), unit="m/s", fmt=NumberFormat.VELOCITY,
        highlight=True))

    plots.append(PlotSpec(
        id="model_trajectories",
        title="Relative trajectory (target-centred LVLH frame)",
        kind=PlotKind.SCATTER_3D, show_target_marker=True, height=520,
        x=AxisSpec(title="radial x", unit="m"),
        y=AxisSpec(title="along-track y", unit="m"),
        z=AxisSpec(title="cross-track z", unit="m"),
        series=[
            SeriesSpec(key="cw", label="CW (linear model)", x_key="cw_x_m",
                       y_key="cw_y_m", z_key="cw_z_m", unit="m",
                       color_index=BLUE, width=5),
            SeriesSpec(key="hf", label="non-linear reference", x_key="hf_x_m",
                       y_key="hf_y_m", z_key="hf_z_m", unit="m",
                       color_index=ORANGE, dash="dash", width=5),
        ],
    ))
    plots.append(PlotSpec(
        id="model_divergence", title="Difference between the two models",
        kind=PlotKind.DUAL_AXIS, x_key="t_s",
        x=AxisSpec(title="time", unit="s"),
        y=AxisSpec(title="position difference", unit="m"),
        y2=AxisSpec(title="velocity difference", unit="m/s"),
        series=[
            SeriesSpec(key="pos", label="position", y_key="cmp_pos_error_m",
                       unit="m", color_index=ORANGE),
            SeriesSpec(key="vel", label="velocity", y_key="cmp_vel_error_mps",
                       unit="m/s", color_index=BLUE, dash="dash", secondary_y=True),
        ],
    ))
    plots.extend(_relative_plots("hf_", "non-linear reference", controlled))

    tables.append(_table_from_frame("model_divergence_summary",
                                    "Model divergence summary", comparison["summary"]))
    tables.append(_table_from_frame("model_divergence_timeseries",
                                    "Model divergence time series", pd.DataFrame({
                                        "t_s": result["times"],
                                        "pos_error_m": comparison["pos_error"],
                                        "vel_error_mps": comparison["vel_error"],
                                    })))

    cw_states = np.asarray(result["cw"]["states"])
    initial_separation = float(np.linalg.norm(cw_states[0, :3]))
    gap = float(summary.get("max_pos_error_m") or 0.0)
    text = (f"Over this run the linear model and the non-linear reference differ by "
            f"at most **{gap:.3g} m** with the force model set to "
            f"*{result['force_model']}*.")
    if initial_separation:
        text += (f" That is {gap / initial_separation:.2%} of the "
                 f"{initial_separation:.0f} m starting separation.")
    if controlled:
        text += ("\n\nThe controller is on, which suppresses the difference: feedback "
                 "keeps correcting the model error before it can accumulate. That is "
                 "reassuring for flying, but it means a closed-loop test is a weak "
                 "way to judge how good the linear model actually is. Turn control "
                 "off to see the unmasked error.")
    interpretations.append(Interpretation(severity=Severity.INFO, text=text))


def _build_collision(result: dict, series: dict, plots: list, metrics: list,
                     tables: list, interpretations: list) -> None:
    analysis = result["analysis"]
    series["t_s"] = _array(result["times"])
    _relative_series("", result["chaser"], series)
    series["dist_to_target_m"] = _array(analysis["dist_to_target"])

    debris_states = result.get("debris_states")
    has_debris = debris_states is not None
    if has_debris:
        debris = np.asarray(debris_states)
        for index, key in enumerate(("debris_x_m", "debris_y_m", "debris_z_m")):
            series[key] = _array(debris[:, index])
    if "dist_to_debris" in analysis:
        series["dist_to_debris_m"] = _array(analysis["dist_to_debris"])

    metrics.extend([
        Metric(key="min_dist_to_target_m", label="Closest approach to target",
               value=_clean(analysis["min_dist_to_target_m"]), unit="m",
               fmt=NumberFormat.DISTANCE, highlight=True),
        Metric(key="keep_out_radius_m", label="Keep-out radius",
               value=_clean(analysis["keep_out_radius_m"]), unit="m",
               fmt=NumberFormat.DISTANCE, highlight=True),
        Metric(key="closest_target_time_s", label="Time of closest approach",
               value=_clean(analysis["closest_target_time_s"]), unit="s",
               fmt=NumberFormat.DURATION),
    ])
    if "min_dist_to_debris_m" in analysis:
        metrics.insert(2, Metric(
            key="min_dist_to_debris_m", label="Closest approach to debris",
            value=_clean(analysis["min_dist_to_debris_m"]), unit="m",
            fmt=NumberFormat.DISTANCE, highlight=True))

    series_specs = [SeriesSpec(key="chaser", label="chaser", x_key="x_m", y_key="y_m",
                               z_key="z_m", unit="m", color_index=BLUE, width=5)]
    if has_debris:
        series_specs.append(SeriesSpec(
            key="debris", label="debris", x_key="debris_x_m", y_key="debris_y_m",
            z_key="debris_z_m", unit="m", color_index=AMBER, dash="dot", width=4))

    plots.append(PlotSpec(
        id="proximity_3d", title="Chaser path, keep-out zone and debris (LVLH)",
        kind=PlotKind.SCATTER_3D, show_target_marker=True, height=520,
        keep_out_radius_m=float(analysis["keep_out_radius_m"]),
        x=AxisSpec(title="radial x", unit="m"),
        y=AxisSpec(title="along-track y", unit="m"),
        z=AxisSpec(title="cross-track z", unit="m"),
        series=series_specs,
    ))

    distance_series = [SeriesSpec(key="target", label="chaser to target",
                                  y_key="dist_to_target_m", unit="m",
                                  color_index=BLUE)]
    if "dist_to_debris" in analysis:
        distance_series.append(SeriesSpec(
            key="debris", label="chaser to debris", y_key="dist_to_debris_m",
            unit="m", color_index=ORANGE, dash="dash"))
    plots.append(PlotSpec(
        id="distances", title="Separation over time", kind=PlotKind.LINE, x_key="t_s",
        x=AxisSpec(title="time", unit="s"), y=AxisSpec(title="distance", unit="m"),
        series=distance_series,
    ))

    tables.append(_table_from_frame("proximity_summary", "Proximity summary",
                                    analysis["summary"]))
    proximity = {"t_s": result["times"], "dist_to_target_m": analysis["dist_to_target"]}
    if "dist_to_debris" in analysis:
        proximity["dist_to_debris_m"] = analysis["dist_to_debris"]
    tables.append(_table_from_frame("proximity_timeseries", "Proximity time series",
                                    pd.DataFrame(proximity)))
    tables.append(_table_from_frame("chaser_timeseries", "Chaser time series",
                                    result["chaser"]["history"]))

    if analysis["keep_out_violation"]:
        interpretations.append(Interpretation(
            severity=Severity.ERROR, title="Keep-out zone entered",
            text=(f"The chaser came within {analysis['min_dist_to_target_m']:.1f} m of "
                  f"the target, inside the {analysis['keep_out_radius_m']:.0f} m "
                  "protected sphere."),
        ))
    else:
        interpretations.append(Interpretation(
            severity=Severity.SUCCESS, title="Keep-out zone respected",
            text=(f"Closest approach was {analysis['min_dist_to_target_m']:.1f} m, "
                  f"outside the {analysis['keep_out_radius_m']:.0f} m sphere."),
        ))

    if "collision_risk" in analysis:
        if analysis["collision_risk"]:
            interpretations.append(Interpretation(
                severity=Severity.ERROR, title="Debris collision risk",
                text=(f"Minimum separation {analysis['min_dist_to_debris_m']:.1f} m at "
                      f"t = {analysis['closest_debris_time_s']:.0f} s, below the "
                      f"{analysis['collision_radius_m']:.0f} m threshold."),
            ))
        else:
            interpretations.append(Interpretation(
                severity=Severity.SUCCESS, title="Debris clearance acceptable",
                text=(f"Minimum separation {analysis['min_dist_to_debris_m']:.1f} m, "
                      f"above the {analysis['collision_radius_m']:.0f} m threshold."),
            ))

    interpretations.append(Interpretation(
        severity=Severity.WARNING,
        text=("This is **detection only**. The controller has no avoidance term, so "
              "it will fly straight through a zone rather than around it. No "
              "position uncertainty is modelled either, so this is not a "
              "conjunction-assessment tool and must not be used as one."),
    ))


def _build_convergence(result: dict, series: dict, plots: list, metrics: list,
                       tables: list, interpretations: list) -> None:
    rows = result["rows"]
    frame = pd.DataFrame(rows)
    orders = [row["observed_order"] for row in rows if "observed_order" in row]

    series["dt_s"] = [_clean(row["dt_s"]) for row in rows]
    series["max_position_error_m"] = [_clean(row["max_position_error_m"]) for row in rows]
    finest = rows[-1]
    series["ideal_fourth_order_m"] = [
        _clean(finest["max_position_error_m"] * (row["dt_s"] / finest["dt_s"]) ** 4)
        for row in rows
    ]

    metrics.extend([
        Metric(key="finest_dt_s", label="Finest step tested",
               value=_clean(finest["dt_s"]), unit="s", fmt=NumberFormat.FIXED,
               precision=2, highlight=True),
        Metric(key="finest_error_m", label="Error at the finest step",
               value=_clean(finest["max_position_error_m"]), unit="m",
               fmt=NumberFormat.SCIENTIFIC, precision=3, highlight=True),
    ])
    if orders:
        mean_order = float(np.mean(orders))
        metrics.append(Metric(
            key="mean_observed_order", label="Mean observed order",
            value=mean_order, fmt=NumberFormat.FIXED, precision=2, highlight=True,
            help="A fourth-order method should give about 4."))
        if 3.6 <= mean_order <= 4.4:
            interpretations.append(Interpretation(
                severity=Severity.SUCCESS,
                text=(f"The measured order is {mean_order:.2f}, which is the ~4 "
                      "expected of classical RK4: halving the step cuts the error by "
                      "roughly sixteen. The published results, run at 500-2000 steps "
                      "per orbit, are well converged."),
            ))
        else:
            interpretations.append(Interpretation(
                severity=Severity.WARNING,
                text=(f"The measured order is {mean_order:.2f} rather than about 4. "
                      "At very fine steps this usually means floating-point round-off "
                      "has become larger than the truncation error, which is a limit "
                      "of arithmetic rather than of the method."),
            ))

    plots.append(PlotSpec(
        id="convergence", title="Integration error against step size",
        kind=PlotKind.LOG_LOG, x_key="dt_s", height=400,
        x=AxisSpec(title="timestep", unit="s", log=True),
        y=AxisSpec(title="max position error vs exact solution", unit="m", log=True),
        series=[
            SeriesSpec(key="measured", label="measured error",
                       y_key="max_position_error_m", unit="m", color_index=BLUE,
                       width=3, mode="lines+markers"),
            SeriesSpec(key="ideal", label="ideal 4th-order slope",
                       y_key="ideal_fourth_order_m", unit="m", color_index=BLACK,
                       dash="dash", width=2),
        ],
    ))

    tables.append(_table_from_frame("convergence_study", "Convergence study", frame))
    interpretations.append(Interpretation(
        severity=Severity.INFO,
        text=("Note: this engine has only one integrator (RK4), so this measures how "
              "that integrator converges as the step shrinks. It is not a comparison "
              "between different integration schemes."),
    ))


def _build_external_export(result: dict, series: dict, plots: list, metrics: list,
                           tables: list, interpretations: list) -> None:
    metadata = result["metadata"]
    states = np.asarray(result["states"])

    series["t_s"] = _array(result["times"])
    for index, key in enumerate(("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")):
        series[key] = _array(states[:, index])

    metrics.extend([
        Metric(key="n_samples", label="Ephemeris samples",
               value=int(metadata["n_samples"]), fmt=NumberFormat.INTEGER,
               highlight=True),
        Metric(key="dt_s", label="Timestep", value=float(metadata["dt_s"]), unit="s",
               fmt=NumberFormat.FIXED, precision=2, highlight=True),
        Metric(key="duration_s", label="Duration",
               value=float(metadata["duration_s"]), unit="s",
               fmt=NumberFormat.DURATION, highlight=True),
    ])

    plots.append(PlotSpec(
        id="orbit_3d", title="Exported trajectory", kind=PlotKind.ORBIT_3D,
        show_earth=True, height=520,
        x=AxisSpec(title="x", unit="km"), y=AxisSpec(title="y", unit="km"),
        z=AxisSpec(title="z", unit="km"),
        series=[SeriesSpec(key="orbit", label="trajectory", x_key="x_m", y_key="y_m",
                           z_key="z_m", unit="m", color_index=ORANGE, width=4)],
    ))

    tables.append(_table_from_frame("ephemeris", "Ephemeris", result["ephemeris"]))
    tables.append(TableSpec(
        id="scenario_metadata", label="Scenario as exported",
        columns=[ColumnSpec(key="field", label="field", numeric=False),
                 ColumnSpec(key="value", label="value", numeric=False)],
        rows=[{"field": key, "value": str(value)} for key, value in metadata.items()],
        downloadable=False,
    ))

    interpretations.append(Interpretation(
        severity=Severity.WARNING,
        text=("**No external comparison is shipped with this software.** To complete "
              "the validation you would reproduce this exact scenario in GMAT or STK "
              "yourself, export that tool's ephemeris, and compare. Use the initial "
              "state to full precision from the JSON — a rounded copy measures your "
              "rounding, not the propagators. This project is not affiliated with "
              "NASA or Ansys/AGI."),
    ))


_BUILDERS = {
    "orbit": _build_orbit,
    "orbit_comparison": _build_orbit_comparison,
    "relative": _build_relative,
    "gain_sweep": _build_gain_sweep,
    "comparison": _build_comparison,
    "collision": _build_collision,
    "convergence": _build_convergence,
    "external_export": _build_external_export,
}


def build_result(
    scenario: Any,
    params: dict[str, Any],
    result: Any,
    runtime_ms: int,
    n_steps: int,
    warnings: list[str],
) -> SimulationResult:
    builder = _BUILDERS.get(scenario.result_kind)
    if builder is None:
        raise ValueError(f"no adapter for result kind {scenario.result_kind!r}")

    series: dict[str, list[float | None]] = {}
    plots: list[PlotSpec] = []
    metrics: list[Metric] = []
    tables: list[TableSpec] = []
    interpretations: list[Interpretation] = []

    builder(result, series, plots, metrics, tables, interpretations)

    by_name = {param.name: param for param in scenario.params}
    inputs = [
        InputEcho(
            name=name,
            label=by_name[name].label if name in by_name else name,
            value=_clean(value),
            unit=by_name[name].unit if name in by_name else "",
        )
        for name, value in params.items()
    ]

    force_model = None
    if isinstance(result, dict):
        force_model = result.get("force_model") or result.get("label")

    solver = SolverInfo(
        engine_version=orbital_core.__version__,
        model=scenario.model,
        force_model=force_model,
        n_steps=n_steps or None,
        dt_s=_clean(params.get("dt_s")),
        duration_s=_clean(params.get("duration_s")),
    )

    extras: dict[str, Any] = {}
    if scenario.result_kind == "external_export" and isinstance(result, dict):
        extras["csv_text"] = result.get("csv_text")
        extras["json_text"] = result.get("json_text")
        extras["metadata"] = {k: _clean(v) for k, v in result.get("metadata", {}).items()}

    return SimulationResult(
        scenario_id=scenario.id,
        scenario_title=scenario.title,
        result_kind=scenario.result_kind,
        inputs=inputs,
        metrics=metrics,
        series=series,
        plots=plots,
        tables=tables,
        interpretations=interpretations,
        warnings=list(warnings),
        assumptions=list(scenario.assumptions),
        limitations=list(scenario.limitations),
        reading_guide=scenario.reading_guide or "",
        solver=solver,
        runtime_ms=runtime_ms,
        n_steps=n_steps,
        completed_at=_now_iso(),
        extras=extras,
    )
