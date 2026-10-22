"""Rendering of simulation results: plots, numbers, interpretation, downloads.

Step 6 of the participant journey. Every result view shows four things, in this
order:

1. the headline numbers,
2. the plots,
3. a plain-English reading of what the numbers mean, and
4. the assumptions and limitations that qualify them.

Point 4 is not optional decoration. A number without its assumptions is how a
student comes away believing something false.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from ..export import config_json, result_tables, results_readme, tables_to_csv
from . import figures
from .common import format_number, show_assumptions_and_limits


# Per-kind renderers
def _render_orbit(result) -> None:
    row = result["validation"].iloc[0]
    columns = st.columns(3)
    columns[0].metric("Orbital period", f"{result['period_s'] / 60:.1f} min",
                      help="Time for one complete revolution.")
    columns[1].metric("Energy drift (relative)",
                      format_number(row["max_energy_drift_rel"]),
                      help="How much the total orbital energy wandered. Physics says "
                           "it should be exactly zero, so this number is a measure of "
                           "the integrator's accuracy.")
    columns[2].metric("Period error (relative)",
                      format_number(row["period_rel_err"]),
                      help="Measured period against the analytic formula.")

    st.plotly_chart(figures.orbit_3d(result["states"]), width="stretch")
    st.caption("The translucent sphere is Earth, drawn to scale.")
    st.plotly_chart(
        figures.series(result["times"], [("altitude", result["altitude_km"])],
                       "altitude [km]", "Altitude against time"),
        width="stretch")

    drift = float(row["max_energy_drift_rel"])
    verdict = ("The integrator conserved energy to about "
               f"{drift:.0e} — roughly {abs(int(np.log10(drift))) if drift else 0} "
               "digits of agreement with theory. It is not a meaningful source of "
               "error here." if drift and drift < 1e-8 else
               f"Energy drifted by about {drift:.0e}, which is larger than the "
               "~1e-13 seen at the published step sizes. Consider a finer timestep.")
    st.info(verdict, icon="📐")


def _render_orbit_comparison(result) -> None:
    separation = result["separation_m"]
    perturbed_altitude = result["perturbed"]["altitude_km"]
    altitude_change = float(perturbed_altitude[0] - perturbed_altitude[-1])

    columns = st.columns(3)
    columns[0].metric("Maximum divergence from the ideal orbit",
                      f"{separation.max() / 1000:.2f} km",
                      help="Greatest distance between the perturbed trajectory and the "
                           "unperturbed one.")
    columns[1].metric("Divergence at the end", f"{separation[-1] / 1000:.2f} km")
    columns[2].metric("Altitude change", f"{altitude_change * 1000:.0f} m",
                      help="Positive means the orbit lost altitude over the run.")

    st.plotly_chart(
        figures.series(result["times"], [("separation", separation / 1000.0)],
                       "separation [km]",
                       "How far the perturbed orbit drifts from the ideal one"),
        width="stretch")
    st.plotly_chart(
        figures.series(result["times"],
                       [("ideal two-body", result["baseline"]["altitude_km"]),
                        ("perturbed", perturbed_altitude)],
                       "altitude [km]", "Altitude against time"),
        width="stretch")
    st.plotly_chart(figures.orbit_3d(result["perturbed"]["states"],
                                     "Perturbed orbit"), width="stretch")

    if separation.max() > 1.0 and np.all(np.diff(separation) >= -1e-6):
        st.info(
            "The divergence grows steadily rather than oscillating about zero. That "
            "steady growth is what makes a perturbation *secular*: it accumulates "
            "orbit after orbit instead of averaging out.", icon="📈")


def _render_relative(result) -> None:
    metrics = result["metrics"].iloc[0]
    converged = np.isfinite(metrics["convergence_time_s"])

    columns = st.columns(4)
    columns[0].metric("Starting distance",
                      format_number(metrics["initial_pos_error_m"], "m"))
    columns[1].metric("Final distance",
                      format_number(metrics["final_pos_error_m"], "m"))
    columns[2].metric("Time to converge",
                      f"{metrics['convergence_time_s']:.0f} s" if converged else "—",
                      help="Time after which the distance stayed within 1 m for the "
                           "rest of the run. A dash means it never did.")
    columns[3].metric("Δv (propellant cost)",
                      format_number(metrics["delta_v_mps"], "m/s"),
                      help="The integral of thrust acceleration over time. This is what "
                           "converts into propellant mass.")

    st.plotly_chart(figures.lvlh_trajectory([("chaser", result["states"])]),
                    width="stretch")
    st.caption("The green diamond at the origin is the target. Axes are the "
               "target-centred LVLH frame.")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            figures.components(result["times"], result["states"], [0, 1, 2],
                               ["radial x", "along-track y", "cross-track z"],
                               "position [m]", "Relative position"),
            width="stretch")
    with right:
        st.plotly_chart(
            figures.components(result["times"], result["states"], [3, 4, 5],
                               ["radial", "along-track", "cross-track"],
                               "velocity [m/s]", "Relative velocity"),
            width="stretch")

    if float(np.max(np.abs(result["controls"]))) > 0.0:
        st.plotly_chart(figures.control_history(result["times"], result["controls"]),
                        width="stretch")
        st.caption("The area under the |u| curve is the Δv.")
        _controlled_interpretation(metrics, converged)
    else:
        _free_drift_interpretation(result)


def _controlled_interpretation(metrics, converged) -> None:
    peak = float(metrics["max_control_accel_mps2"])
    if converged:
        st.success(
            f"The chaser converged after {metrics['convergence_time_s']:.0f} s, "
            f"finishing {format_number(metrics['final_pos_error_m'], 'm')} from the "
            f"target for {format_number(metrics['delta_v_mps'], 'm/s')} of Δv. Peak "
            f"thrust was {peak:.4g} m/s².", icon="✅")
    else:
        st.warning(
            "The chaser never settled within 1 m and stayed there. That is a real "
            "result, not an error: the gains may be too weak for this starting "
            "offset, or the run may simply be too short. Try a higher bandwidth or "
            "a longer duration.", icon="⚠️")


def _free_drift_interpretation(result) -> None:
    states = result["states"]
    start, end = states[0, :3], states[-1, :3]
    drift = float(np.linalg.norm(end - start))
    along_track = float(end[1] - start[1])

    message = (
        f"With no thrust at all, the chaser moved {drift:.0f} m over the run "
        f"({along_track:+.0f} m of that along-track).")
    if drift < 1.0:
        message += (" It effectively stayed put — you have found one of the natural "
                    "equilibria of relative motion.")
    elif abs(along_track) > 5 * abs(float(end[0] - start[0])) + 1.0:
        message += (" The motion is dominated by along-track drift, which is the "
                    "signature of a period mismatch between the two orbits.")
    st.info(message, icon="🛰️")


def _render_gain_sweep(result) -> None:
    table = result["table"]
    converged = table.dropna(subset=["convergence_time_s"])

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            figures.gain_heatmap(table, "convergence_time_s",
                                 "Time to converge", "seconds"),
            width="stretch")
    with right:
        st.plotly_chart(
            figures.gain_heatmap(table, "delta_v_mps", "Propellant cost (Δv)", "m/s"),
            width="stretch")

    st.dataframe(table, width="stretch", hide_index=True)

    if converged.empty:
        st.warning(
            "No gain combination in this grid reached the 1 m tolerance within the "
            "run. Try a longer duration or a larger starting offset.", icon="⚠️")
        return

    fastest = converged.loc[converged["convergence_time_s"].idxmin()]
    cheapest = converged.loc[converged["delta_v_mps"].idxmin()]
    non_converging = len(table) - len(converged)

    st.info(
        f"**Fastest:** Kp = {fastest.kp:.1e}, Kd = {fastest.kd:.1e} → "
        f"{fastest.convergence_time_s:.0f} s at {fastest.delta_v_mps:.2f} m/s of Δv.\n\n"
        f"**Cheapest:** Kp = {cheapest.kp:.1e}, Kd = {cheapest.kd:.1e} → "
        f"{cheapest.convergence_time_s:.0f} s at {cheapest.delta_v_mps:.2f} m/s of Δv.\n\n"
        + (f"They are not the same cell — that gap *is* the trade-off. The fastest "
           f"option costs {fastest.delta_v_mps / cheapest.delta_v_mps:.1f}× the Δv of "
           f"the most efficient one."
           if fastest.name != cheapest.name else
           "On this grid the same cell happens to be both fastest and cheapest.")
        + (f"\n\n{non_converging} of {len(table)} combinations never reached the 1 m "
           "tolerance and are shown blank. They are reported, not hidden."
           if non_converging else ""),
        icon="⚖️")


def _render_comparison(result) -> None:
    if "comparison" not in result:
        primary = result.get("primary")
        if primary is not None:
            _render_relative(primary)
        return

    comparison = result["comparison"]
    summary = comparison["summary"].iloc[0]

    columns = st.columns(3)
    columns[0].metric("Largest gap between the models",
                      format_number(summary["max_pos_error_m"], "m"),
                      help="Greatest distance between the linear CW prediction and the "
                           "non-linear reference at any point in the run.")
    columns[1].metric("Gap at the end",
                      format_number(summary["final_pos_error_m"], "m"))
    columns[2].metric("Largest velocity gap",
                      format_number(summary["max_vel_error_mps"], "m/s"))

    st.plotly_chart(
        figures.lvlh_trajectory([("CW (linear model)", result["cw"]["states"]),
                                 ("non-linear reference",
                                  result["high_fidelity"]["states"])]),
        width="stretch")
    st.plotly_chart(
        figures.dual_error(result["times"], comparison["pos_error"],
                           comparison["vel_error"]),
        width="stretch")
    st.dataframe(comparison["summary"], width="stretch", hide_index=True)

    initial_separation = float(np.linalg.norm(result["cw"]["states"][0, :3]))
    gap = float(summary["max_pos_error_m"])
    relative = gap / initial_separation if initial_separation else float("nan")

    message = (
        f"Over this run the linear model and the non-linear reference differ by at "
        f"most **{gap:.3g} m** with the force model set to *{result['force_model']}*.")
    if np.isfinite(relative):
        message += (f" That is {relative:.2%} of the {initial_separation:.0f} m "
                    "starting separation.")
    if result.get("controlled"):
        message += (
            "\n\nThe controller is on, which suppresses the difference: feedback keeps "
            "correcting the model error before it can accumulate. That is reassuring "
            "for flying, but it means a closed-loop test is a weak way to judge how "
            "good the linear model actually is. Turn control off to see the "
            "unmasked error.")
    st.info(message, icon="🔬")

    primary = result.get("primary")
    if primary is not None:
        with st.expander("Full trajectory detail for the reference model"):
            _render_relative(primary)


def _render_collision(result) -> None:
    analysis = result["analysis"]

    columns = st.columns(3)
    columns[0].metric("Closest approach to target",
                      f"{analysis['min_dist_to_target_m']:.1f} m")
    columns[1].metric("Keep-out radius", f"{analysis['keep_out_radius_m']:.0f} m")
    if "min_dist_to_debris_m" in analysis:
        columns[2].metric("Closest approach to debris",
                          f"{analysis['min_dist_to_debris_m']:.1f} m")

    if analysis["keep_out_violation"]:
        st.error(
            f"**Keep-out zone entered.** The chaser came within "
            f"{analysis['min_dist_to_target_m']:.1f} m of the target, inside the "
            f"{analysis['keep_out_radius_m']:.0f} m protected sphere.", icon="🚨")
    else:
        st.success(
            f"**Keep-out zone respected.** Closest approach was "
            f"{analysis['min_dist_to_target_m']:.1f} m, outside the "
            f"{analysis['keep_out_radius_m']:.0f} m sphere.", icon="✅")

    if "collision_risk" in analysis:
        if analysis["collision_risk"]:
            st.error(
                f"**Debris collision risk.** Minimum separation "
                f"{analysis['min_dist_to_debris_m']:.1f} m at "
                f"t = {analysis['closest_debris_time_s']:.0f} s, below the "
                f"{analysis['collision_radius_m']:.0f} m threshold.", icon="🚨")
        else:
            st.success(
                f"**Debris clearance acceptable.** Minimum separation "
                f"{analysis['min_dist_to_debris_m']:.1f} m, above the "
                f"{analysis['collision_radius_m']:.0f} m threshold.", icon="✅")

    st.plotly_chart(
        figures.proximity_3d(result["chaser"]["states"], result["debris_states"],
                             analysis["keep_out_radius_m"], analysis),
        width="stretch")

    entries = [("chaser to target", analysis["dist_to_target"])]
    if "dist_to_debris" in analysis:
        entries.append(("chaser to debris", analysis["dist_to_debris"]))
    st.plotly_chart(
        figures.series(result["times"], entries, "distance [m]", "Separation over time"),
        width="stretch")

    st.warning(
        "This is **detection only**. The controller has no avoidance term, so it "
        "will fly straight through a zone rather than around it. No position "
        "uncertainty is modelled either, so this is not a conjunction-assessment "
        "tool and must not be used as one.", icon="⚠️")


def _render_convergence(result) -> None:
    import pandas as pd

    rows = result["rows"]
    table = pd.DataFrame(rows)
    orders = [row["observed_order"] for row in rows if "observed_order" in row]

    columns = st.columns(3)
    columns[0].metric("Finest step tested", f"{rows[-1]['dt_s']:.2f} s")
    columns[1].metric("Error at the finest step",
                      format_number(rows[-1]["max_position_error_m"], "m"))
    if orders:
        columns[2].metric("Mean observed order", f"{np.mean(orders):.2f}",
                          help="A fourth-order method should give about 4.")

    st.plotly_chart(figures.convergence_plot(rows), width="stretch")
    st.dataframe(table, width="stretch", hide_index=True)

    if orders:
        mean_order = float(np.mean(orders))
        if 3.6 <= mean_order <= 4.4:
            st.success(
                f"The measured order is {mean_order:.2f}, which is the ~4 expected of "
                "classical RK4: halving the step cuts the error by roughly sixteen. "
                "The published results, run at 500–2000 steps per orbit, are well "
                "converged.", icon="✅")
        else:
            st.warning(
                f"The measured order is {mean_order:.2f} rather than about 4. At very "
                "fine steps this usually means floating-point round-off has become "
                "larger than the truncation error, which is a limit of arithmetic "
                "rather than of the method.", icon="⚠️")

    st.caption(
        "Note: this engine has only one integrator (RK4), so this measures how that "
        "integrator converges as the step shrinks. It is not a comparison between "
        "different integration schemes.")


def _render_external_export(result) -> None:
    import pandas as pd

    metadata = result["metadata"]
    st.markdown("**Scenario as exported**")
    st.dataframe(
        pd.DataFrame({"field": list(metadata),
                      "value": [str(value) for value in metadata.values()]}),
        width="stretch", hide_index=True)

    left, right = st.columns(2)
    left.download_button("Download ephemeris + metadata (CSV)",
                         result["csv_text"].encode("utf-8"),
                         "validation_scenario.csv", "text/csv",
                         width="stretch")
    right.download_button("Download scenario (JSON)",
                          result["json_text"].encode("utf-8"),
                          "validation_scenario.json", "application/json",
                          width="stretch")

    st.plotly_chart(figures.orbit_3d(result["states"], "Exported trajectory"),
                    width="stretch")

    st.warning(
        "**No external comparison is shipped with this software.** To complete the "
        "validation you would reproduce this exact scenario in GMAT or STK "
        "yourself, export that tool's ephemeris, and compare. Use the initial "
        "state to full precision from the JSON — a rounded copy measures your "
        "rounding, not the propagators. This project is not affiliated with NASA "
        "or Ansys/AGI.", icon="⚠️")


_RENDERERS = {
    "orbit": _render_orbit,
    "orbit_comparison": _render_orbit_comparison,
    "relative": _render_relative,
    "gain_sweep": _render_gain_sweep,
    "comparison": _render_comparison,
    "collision": _render_collision,
    "convergence": _render_convergence,
    "external_export": _render_external_export,
}


# Entry point
def render_result(scenario, outcome, params: dict, app_version: str = "1.0.0",
                  on_download=None) -> None:
    """Render one completed run: numbers, plots, reading, caveats, downloads."""
    st.subheader("Results")

    renderer = _RENDERERS.get(scenario.result_kind)
    if renderer is None:
        st.error("This result type cannot be displayed. Please report this.")
        return

    try:
        renderer(outcome.result)
    except Exception:  # noqa: BLE001 - a plotting fault must not lose the run
        st.error(
            "The results could not be drawn, although the simulation itself "
            "completed. You can still download the numbers below.", icon="🚫")

    if scenario.reading_guide:
        with st.expander("How to read this", expanded=False):
            st.markdown(scenario.reading_guide)

    st.divider()
    show_assumptions_and_limits(scenario)

    st.divider()
    _downloads(scenario, outcome, params, app_version, on_download)


def _downloads(scenario, outcome, params, app_version, on_download) -> None:
    st.subheader("Download")
    st.caption(
        "Results are numbers only — nothing that identifies you is included in "
        "any download.")

    tables = result_tables(scenario.result_kind, outcome.result)
    files = tables_to_csv(tables)

    columns = st.columns(max(2, min(3, len(files) + 2)))
    position = 0

    for name, content in files.items():
        with columns[position % len(columns)]:
            if st.download_button(f"{name}", content, name, "text/csv",
                                  key=f"dl_{scenario.id}_{name}",
                                  width="stretch") and on_download:
                on_download("results_csv")
        position += 1

    with columns[position % len(columns)]:
        if st.download_button(
                "Scenario configuration (JSON)",
                config_json(scenario, params, app_version).encode("utf-8"),
                f"{scenario.id}_config.json", "application/json",
                key=f"dl_{scenario.id}_config", width="stretch",
                help="Everything needed to reproduce this exact run.") and on_download:
            on_download("scenario_config")
    position += 1

    with columns[position % len(columns)]:
        if st.download_button(
                "Notes and caveats (TXT)",
                results_readme(scenario, params, outcome.warnings,
                               app_version).encode("utf-8"),
                f"{scenario.id}_notes.txt", "text/plain",
                key=f"dl_{scenario.id}_notes", width="stretch",
                help="Assumptions, limitations and any warnings raised for this "
                     "run — keep it with the CSVs.") and on_download:
            on_download("notes")


__all__ = ["render_result"]
