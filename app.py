"""Streamlit MVP interface for the orbital rendezvous research engine.

This is a *thin* frontend. It contains **no physics or simulation logic** -- all
computation goes through ``src/scenarios.py`` (the frontend-agnostic backend
API), which returns plain dicts / pandas DataFrames. This file only handles
inputs, buttons, Plotly plots, explanations, metrics, and downloads, so the
backend stays reusable if this UI is later replaced by a polished web frontend.

Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import scenarios as sc
from src import orbit

st.set_page_config(page_title="Orbital Rendezvous Engine", page_icon="🛰️", layout="wide")

EARTH_R_KM = orbit.R_EARTH / 1000.0


# ==========================================================================
# Plotly figure builders (pure presentation: data in -> go.Figure out)
# ==========================================================================
def fig_orbit_3d(states, title="Inertial orbit"):
    p = np.asarray(states)[:, :3] / 1000.0  # km
    fig = go.Figure()
    # Light translucent Earth.
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:16j]
    fig.add_surface(
        x=EARTH_R_KM * np.cos(u) * np.sin(v),
        y=EARTH_R_KM * np.sin(u) * np.sin(v),
        z=EARTH_R_KM * np.cos(v),
        colorscale="Blues", opacity=0.3, showscale=False, hoverinfo="skip",
    )
    fig.add_scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
                      line=dict(color="orange", width=4), name="orbit")
    fig.add_scatter3d(x=[p[0, 0]], y=[p[0, 1]], z=[p[0, 2]], mode="markers",
                      marker=dict(size=4, color="black"), name="start")
    fig.update_layout(title=title, height=520, scene=dict(
        xaxis_title="x [km]", yaxis_title="y [km]", zaxis_title="z [km]",
        aspectmode="data"), margin=dict(l=0, r=0, t=40, b=0))
    return fig


def fig_lvlh_trajectory(trajectories, title="Relative trajectory (LVLH)"):
    """``trajectories`` = list of (label, states)."""
    fig = go.Figure()
    for label, states in trajectories:
        p = np.asarray(states)[:, :3]
        fig.add_scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
                          line=dict(width=4), name=label)
        fig.add_scatter3d(x=[p[0, 0]], y=[p[0, 1]], z=[p[0, 2]], mode="markers",
                          marker=dict(size=3, color="black"), showlegend=False)
    fig.add_scatter3d(x=[0], y=[0], z=[0], mode="markers",
                      marker=dict(size=6, color="green", symbol="diamond"), name="target")
    fig.update_layout(title=title, height=520, scene=dict(
        xaxis_title="radial x [m]", yaxis_title="along-track y [m]",
        zaxis_title="cross-track z [m]", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0))
    return fig


def fig_components(times, states, cols, names, ylabel, title):
    fig = go.Figure()
    states = np.asarray(states)
    for c, n in zip(cols, names):
        fig.add_scatter(x=times, y=states[:, c], mode="lines", name=n)
    fig.update_layout(title=title, xaxis_title="time [s]", yaxis_title=ylabel,
                      height=360, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def fig_control(times, controls, title="Control acceleration vs time"):
    controls = np.asarray(controls)
    fig = go.Figure()
    for c, n in zip(range(3), ("ax radial", "ay along-track", "az cross-track")):
        fig.add_scatter(x=times, y=controls[:, c], mode="lines", name=n)
    fig.add_scatter(x=times, y=np.linalg.norm(controls, axis=1), mode="lines",
                    name="|u|", line=dict(color="black", dash="dash"))
    fig.update_layout(title=title, xaxis_title="time [s]",
                      yaxis_title="control accel [m/s²]", height=360,
                      margin=dict(l=0, r=0, t=40, b=0))
    return fig


def fig_single(times, y, ylabel, title, color="crimson", logy=False):
    fig = go.Figure()
    fig.add_scatter(x=times, y=y, mode="lines", line=dict(color=color), name=ylabel)
    fig.update_layout(title=title, xaxis_title="time [s]", yaxis_title=ylabel,
                      height=340, margin=dict(l=0, r=0, t=40, b=0))
    if logy:
        fig.update_yaxes(type="log")
    return fig


def fig_error_over_time(times, pos_err, vel_err, title="CW vs high-fidelity error over time"):
    fig = go.Figure()
    fig.add_scatter(x=times, y=pos_err, mode="lines", name="position error [m]",
                    line=dict(color="crimson"))
    fig.add_scatter(x=times, y=vel_err, mode="lines", name="velocity error [m/s]",
                    line=dict(color="royalblue"), yaxis="y2")
    fig.update_layout(
        title=title, xaxis_title="time [s]",
        yaxis=dict(title="position error [m]"),
        yaxis2=dict(title="velocity error [m/s]", overlaying="y", side="right"),
        height=380, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.12))
    return fig


def fig_gain_heatmap(df, metric, title):
    grid = df.pivot(index="kd", columns="kp", values=metric)
    fig = go.Figure(go.Heatmap(
        z=grid.to_numpy(),
        x=[f"{v:.1e}" for v in grid.columns],
        y=[f"{v:.1e}" for v in grid.index],
        colorscale="Viridis", colorbar=dict(title=metric)))
    fig.update_layout(title=title, xaxis_title="Kp", yaxis_title="Kd",
                      height=380, margin=dict(l=0, r=0, t=40, b=0))
    return fig


# ==========================================================================
# Small UI helpers
# ==========================================================================
def download_csv(df, filename, key):
    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv", key=key)


def reset_prefix(prefix):
    for k in list(st.session_state):
        if k.startswith(prefix):
            del st.session_state[k]
    st.rerun()


def show_metrics_row(metrics_df):
    row = metrics_df.iloc[0]
    cols = st.columns(len(row))
    for col, (name, val) in zip(cols, row.items()):
        col.metric(name, f"{val:.4g}" if isinstance(val, (int, float, np.floating)) else str(val))


def rel_inputs(prefix, defaults=(0.0, 1000.0, 0.0, 0.0, 0.0, 0.0), with_velocity=True):
    """Initial relative state inputs; returns (pos, vel)."""
    c = st.columns(3)
    x = c[0].number_input("x radial [m]", value=defaults[0], step=10.0, key=prefix + "x")
    y = c[1].number_input("y along-track [m]", value=defaults[1], step=10.0, key=prefix + "y")
    z = c[2].number_input("z cross-track [m]", value=defaults[2], step=10.0, key=prefix + "z")
    if with_velocity:
        d = st.columns(3)
        vx = d[0].number_input("vx [m/s]", value=defaults[3], step=0.01, format="%.3f", key=prefix + "vx")
        vy = d[1].number_input("vy [m/s]", value=defaults[4], step=0.01, format="%.3f", key=prefix + "vy")
        vz = d[2].number_input("vz [m/s]", value=defaults[5], step=0.01, format="%.3f", key=prefix + "vz")
    else:
        vx = vy = vz = 0.0
    return [x, y, z], [vx, vy, vz]


def run_reset_buttons(prefix):
    c1, c2 = st.columns([1, 1])
    run = c1.button("▶️ Run Simulation", key=prefix + "run", type="primary")
    if c2.button("↺ Reset to Default", key=prefix + "reset"):
        reset_prefix(prefix)
    return run


# ==========================================================================
# LEARNING MODE presets
# ==========================================================================
def preset_basic_orbit():
    p = "basic_"
    st.subheader("Basic circular orbit")
    st.markdown(
        "A satellite on a **circular orbit** stays at constant altitude, balancing "
        "gravity against its orbital speed. With only point-mass gravity, energy and "
        "the orbital period are conserved — a good check that the simulator is accurate.")
    alt = st.slider("Altitude [km]", 200, 1200, 400, 50, key=p + "alt")
    inc = st.slider("Inclination [deg]", 0, 90, 0, 5, key=p + "inc")
    norb = st.slider("Number of orbits", 1, 5, 1, key=p + "norb")
    if run_reset_buttons(p):
        st.session_state[p + "res"] = sc.run_orbit(alt, inc, norb, False, False, steps_per_orbit=1500)
    res = st.session_state.get(p + "res")
    if res:
        v = res["validation"].iloc[0]
        c = st.columns(3)
        c[0].metric("Orbital period", f"{res['period_s']/60:.1f} min")
        c[1].metric("Energy drift (rel)", f"{v['max_energy_drift_rel']:.1e}")
        c[2].metric("Period error (rel)", f"{v['period_rel_err']:.1e}")
        st.plotly_chart(fig_orbit_3d(res["states"], "Inertial circular orbit"), use_container_width=True)
        st.plotly_chart(fig_single(res["times"], res["altitude_km"], "altitude [km]",
                                   "Altitude vs time (should be flat)", color="seagreen"),
                        use_container_width=True)
        st.success(
            f"At {alt} km the period is {res['period_s']/60:.1f} min. Energy is conserved to "
            f"~{v['max_energy_drift_rel']:.0e} and the measured period matches theory to "
            f"~{v['period_rel_err']:.0e} — the RK4 integrator is trustworthy at this step.")


def preset_j2():
    p = "j2_"
    st.subheader("J2 perturbation effect")
    st.markdown(
        "Earth is not a perfect sphere — its **equatorial bulge (J2)** tugs on the orbit, "
        "causing it to slowly precess and drift away from the ideal two-body path. This "
        "compares an orbit *with* J2 against the ideal orbit and shows how far they separate.")
    alt = st.slider("Altitude [km]", 200, 1200, 400, 50, key=p + "alt")
    inc = st.slider("Inclination [deg]", 0, 90, 51, 5, key=p + "inc")
    norb = st.slider("Number of orbits", 1, 15, 5, key=p + "norb")
    if run_reset_buttons(p):
        st.session_state[p + "res"] = sc.compare_orbit_perturbation(alt, inc, norb, True, False, steps_per_orbit=800)
    res = st.session_state.get(p + "res")
    if res:
        sep = res["separation_m"]
        st.metric("Max divergence from ideal orbit", f"{sep.max()/1000:.1f} km")
        st.plotly_chart(fig_single(res["times"], sep / 1000.0, "separation [km]",
                                   "Position difference from ideal two-body orbit", color="purple"),
                        use_container_width=True)
        st.plotly_chart(fig_orbit_3d(res["perturbed"]["states"], "Orbit with J2"),
                        use_container_width=True)
        st.info(
            f"Over {norb} orbits, J2 pulls the satellite up to **{sep.max()/1000:.1f} km** away "
            f"from where ideal two-body gravity predicts. J2 is the dominant secular gravitational "
            f"perturbation in LEO; its effect grows with each orbit and with inclination.")


def preset_drag():
    p = "drag_"
    st.subheader("Atmospheric drag effect")
    st.markdown(
        "Even in LEO there is thin atmosphere. **Drag** slowly removes energy, lowering the "
        "altitude — the orbit *decays*. Lower altitude, larger area, or smaller mass all "
        "increase the decay rate.")
    alt = st.slider("Altitude [km]", 200, 600, 300, 25, key=p + "alt")
    norb = st.slider("Number of orbits", 1, 30, 10, key=p + "norb")
    area = st.slider("Cross-section area [m²]", 0.5, 10.0, 2.0, 0.5, key=p + "area")
    mass = st.slider("Mass [kg]", 10, 500, 50, 10, key=p + "mass")
    if run_reset_buttons(p):
        st.session_state[p + "res"] = sc.compare_orbit_perturbation(
            alt, 51.6, norb, False, True, steps_per_orbit=600, area=area, mass=mass)
    res = st.session_state.get(p + "res")
    if res:
        a = res["perturbed"]["altitude_km"]
        drop = a[0] - a[-1]
        st.metric("Altitude lost", f"{drop*1000:.0f} m")
        st.plotly_chart(fig_single(res["times"], a, "altitude [km]",
                                   "Altitude vs time (decaying)", color="firebrick"),
                        use_container_width=True)
        st.info(
            f"Drag lowers the orbit by **{drop*1000:.0f} m** over {norb} orbits for this "
            f"ballistic configuration (area {area} m², mass {mass} kg). Thinner, lighter, "
            f"lower spacecraft decay faster.")


def preset_cw_relative():
    p = "cwr_"
    st.subheader("CW relative motion")
    st.markdown(
        "In the target's local frame (radial / along-track / cross-track), the chaser's "
        "**relative motion** follows the Clohessy–Wiltshire equations. Try the classics: a "
        "pure along-track offset stays put, while a radial offset drifts along-track each orbit.")
    pos, vel = rel_inputs(p, defaults=(100.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    alt = st.slider("Altitude [km]", 200, 1200, 400, 50, key=p + "alt")
    norb = st.slider("Number of orbits", 1, 5, 2, key=p + "norb")
    if run_reset_buttons(p):
        period = orbit.orbital_period(sc.radius_from_altitude(alt))
        st.session_state[p + "res"] = sc.run_cw_relative(pos, vel, alt, duration_s=norb * period, dt_s=10.0)
    res = st.session_state.get(p + "res")
    if res:
        st.plotly_chart(fig_lvlh_trajectory([("chaser", res["states"])]), use_container_width=True)
        st.plotly_chart(fig_components(res["times"], res["states"], [0, 1, 2],
                                       ["x radial", "y along-track", "z cross-track"],
                                       "position [m]", "Relative position vs time"),
                        use_container_width=True)
        drift = res["pos_error"][-1] - res["pos_error"][0]
        st.info(
            f"Starting at {pos} m, the chaser ends {res['pos_error'][-1]:.0f} m from the target "
            f"(drift of {drift:+.0f} m). A radial offset induces along-track drift each orbit; a "
            f"pure along-track offset is an equilibrium and barely moves.")


def preset_pd_rendezvous():
    p = "pd_"
    st.subheader("CW rendezvous with PD controller")
    st.markdown(
        "Add a **PD controller** that thrusts to cancel position and velocity error, driving "
        "the chaser to the target. Higher gains converge faster but cost more fuel (Δv).")
    pos, vel = rel_inputs(p, defaults=(200.0, -300.0, 80.0, 0.0, 0.0, 0.0))
    alt = st.slider("Altitude [km]", 200, 1200, 400, 50, key=p + "alt")
    wn = st.slider("Controller bandwidth (× orbital rate n)", 2.0, 15.0, 6.0, 0.5, key=p + "wn")
    amax = st.slider("Max thrust accel [m/s²]", 0.005, 0.2, 0.05, 0.005, key=p + "amax")
    if run_reset_buttons(p):
        st.session_state[p + "res"] = sc.run_cw_rendezvous(pos, vel, alt, dt_s=5.0,
                                                           max_accel=amax, omega_n_factor=wn)
    res = st.session_state.get(p + "res")
    if res:
        show_metrics_row(res["metrics"])
        c = st.columns(2)
        c[0].plotly_chart(fig_lvlh_trajectory([("chaser", res["states"])]), use_container_width=True)
        c[1].plotly_chart(fig_components(res["times"], res["states"], [0, 1, 2],
                                         ["x", "y", "z"], "position [m]", "Position vs time"),
                          use_container_width=True)
        st.plotly_chart(fig_control(res["times"], res["controls"]), use_container_width=True)
        m = res["metrics"].iloc[0]
        conv = m["convergence_time_s"]
        conv_txt = "did not converge within the run" if not np.isfinite(conv) else f"converged in {conv/60:.1f} min"
        st.success(
            f"The chaser {conv_txt}, ending {m['final_pos_error_m']:.2g} m from the target, "
            f"using Δv ≈ {m['delta_v_mps']:.2f} m/s (peak thrust {m['max_control_accel_mps2']:.3f} m/s²).")
        download_csv(res["history"], "cw_rendezvous.csv", p + "dl")


def preset_gain_tuning():
    p = "gain_"
    st.subheader("Gain tuning demonstration")
    st.markdown(
        "Sweep proportional (**Kp**) and derivative (**Kd**) gains and see the trade-off: "
        "aggressive gains converge fast but burn more Δv, while weak gains may never reach "
        "the target within one orbit.")
    pos, vel = rel_inputs(p, defaults=(200.0, -300.0, 80.0, 0.0, 0.0, 0.0))
    alt = st.slider("Altitude [km]", 200, 1200, 400, 50, key=p + "alt")
    if run_reset_buttons(p):
        st.session_state[p + "res"] = sc.run_gain_sweep(pos, vel, alt, dt_s=10.0)
    res = st.session_state.get(p + "res")
    if res:
        df = res["table"]
        c = st.columns(2)
        c[0].plotly_chart(fig_gain_heatmap(df, "convergence_time_s", "Convergence time [s]"),
                          use_container_width=True)
        c[1].plotly_chart(fig_gain_heatmap(df, "delta_v_mps", "Delta-v [m/s]"),
                          use_container_width=True)
        st.dataframe(df, use_container_width=True)
        conv = df.dropna(subset=["convergence_time_s"])
        if not conv.empty:
            fast = conv.loc[conv["convergence_time_s"].idxmin()]
            cheap = conv.loc[conv["delta_v_mps"].idxmin()]
            st.info(
                f"**Fastest:** Kp={fast.kp:.1e}, Kd={fast.kd:.1e} → {fast.convergence_time_s/60:.1f} min, "
                f"Δv {fast.delta_v_mps:.2f} m/s. **Cheapest:** Kp={cheap.kp:.1e}, Kd={cheap.kd:.1e} → "
                f"{cheap.convergence_time_s/60:.1f} min, Δv {cheap.delta_v_mps:.2f} m/s. "
                f"Blank cells never reached the 1 m tolerance within the orbit.")
        download_csv(df, "gain_sweep.csv", p + "dl")


def preset_cw_vs_hf():
    p = "cmp_"
    st.subheader("CW vs high-fidelity comparison")
    st.markdown(
        "The CW equations are a **linear approximation**. Here we let the chaser drift freely "
        "and compare CW against a high-fidelity non-linear model (with J2 / drag). The gap "
        "shows where the simple model stops being trustworthy.")
    pos, vel = rel_inputs(p, defaults=(0.0, 1000.0, 0.0, 0.0, 0.0, 0.0))
    alt = st.slider("Altitude [km]", 200, 1200, 400, 50, key=p + "alt")
    cc = st.columns(2)
    j2 = cc[0].checkbox("Include J2", True, key=p + "j2")
    drag = cc[1].checkbox("Include drag", False, key=p + "drag")
    if run_reset_buttons(p):
        st.session_state[p + "res"] = sc.run_relative_scenario(
            pos, vel, alt, inclination_deg=51.6, dt_s=10.0, model="comparison",
            controlled=False, use_j2=j2, use_drag=drag)
    res = st.session_state.get(p + "res")
    if res:
        comp = res["comparison"]
        st.dataframe(comp["summary"], use_container_width=True)
        st.plotly_chart(fig_lvlh_trajectory([("CW (linear)", res["cw"]["states"]),
                                             ("high-fidelity", res["high_fidelity"]["states"])]),
                        use_container_width=True)
        st.plotly_chart(fig_error_over_time(res["times"], comp["pos_error"], comp["vel_error"]),
                        use_container_width=True)
        s = comp["summary"].iloc[0]
        st.info(
            f"Over one orbit the CW prediction drifts up to **{s['max_pos_error_m']:.1f} m** from the "
            f"high-fidelity model ({res['force_model']}). The error grows with separation and with "
            f"added perturbations — CW is best trusted for close-proximity operations.")


LEARNING_PRESETS = {
    "Basic circular orbit": preset_basic_orbit,
    "J2 perturbation effect": preset_j2,
    "Atmospheric drag effect": preset_drag,
    "CW relative motion": preset_cw_relative,
    "CW rendezvous with PD controller": preset_pd_rendezvous,
    "Gain tuning demonstration": preset_gain_tuning,
    "CW vs high-fidelity comparison": preset_cw_vs_hf,
}


def learning_mode():
    st.header("🎓 Learning Mode")
    st.caption("Guided presets with plain-English explanations for students and newcomers.")
    choice = st.selectbox("Choose a scenario", list(LEARNING_PRESETS), key="learn_choice")
    st.divider()
    LEARNING_PRESETS[choice]()


# ==========================================================================
# RESEARCH MODE
# ==========================================================================
def research_mode():
    st.header("🔬 Research Mode")
    st.caption("Full control over the initial relative state, orbit, controller, and model.")
    p = "rs_"

    with st.sidebar:
        st.subheader("Initial relative state")
        x = st.number_input("x radial [m]", value=200.0, step=10.0, key=p + "x")
        y = st.number_input("y along-track [m]", value=-300.0, step=10.0, key=p + "y")
        z = st.number_input("z cross-track [m]", value=80.0, step=10.0, key=p + "z")
        vx = st.number_input("vx [m/s]", value=0.0, step=0.01, format="%.3f", key=p + "vx")
        vy = st.number_input("vy [m/s]", value=0.0, step=0.01, format="%.3f", key=p + "vy")
        vz = st.number_input("vz [m/s]", value=0.0, step=0.01, format="%.3f", key=p + "vz")

        st.subheader("Orbit & simulation")
        alt = st.number_input("Target altitude [km]", value=400.0, step=25.0, key=p + "alt")
        inc = st.number_input("Inclination [deg]", value=51.6, step=1.0, key=p + "inc")
        period = orbit.orbital_period(sc.radius_from_altitude(alt))
        dur = st.number_input("Duration [s]", value=float(round(period)), step=100.0, key=p + "dur",
                              help=f"One orbit ≈ {period:.0f} s")
        dt = st.number_input("Timestep [s]", value=5.0, step=1.0, min_value=0.1, key=p + "dt")

        st.subheader("Controller")
        controlled = st.checkbox("Apply PD control", True, key=p + "ctl")
        kp = st.number_input("Kp", value=4.6e-5, format="%.2e", key=p + "kp")
        kd = st.number_input("Kd", value=1.36e-2, format="%.2e", key=p + "kd")
        amax = st.number_input("Max thrust accel [m/s²]", value=0.05, step=0.005, format="%.3f", key=p + "amax")

        st.subheader("Force model & comparison")
        j2 = st.checkbox("J2", True, key=p + "j2")
        drag = st.checkbox("Drag", True, key=p + "drag")
        model = st.selectbox("Model", ["comparison", "cw", "high_fidelity"], key=p + "model")

        run = st.button("▶️ Run Simulation", type="primary", key=p + "run")
        if st.button("↺ Reset to Default", key=p + "reset"):
            reset_prefix(p)

    if run:
        st.session_state[p + "res"] = sc.run_relative_scenario(
            [x, y, z], [vx, vy, vz], altitude_km=alt, inclination_deg=inc,
            duration_s=dur, dt_s=dt, model=model, controlled=controlled,
            kp=kp, kd=kd, max_accel=amax, use_j2=j2, use_drag=drag)

    res = st.session_state.get(p + "res")
    if not res:
        st.info("Set parameters in the sidebar and press **Run Simulation**.")
        return

    primary = res["primary"]
    m = primary["metrics"].iloc[0]
    conv = m["convergence_time_s"]
    c = st.columns(4)
    c[0].metric("Final position error", f"{m['final_pos_error_m']:.3g} m")
    c[1].metric("Final velocity error", f"{m['final_vel_error_mps']:.3g} m/s")
    c[2].metric("Convergence time", "—" if not np.isfinite(conv) else f"{conv:.0f} s")
    c[3].metric("Delta-v (control effort)", f"{m['delta_v_mps']:.3f} m/s")

    trajectories = []
    if "cw" in res:
        trajectories.append(("CW", res["cw"]["states"]))
    if "high_fidelity" in res:
        trajectories.append(("high-fidelity", res["high_fidelity"]["states"]))
    st.plotly_chart(fig_lvlh_trajectory(trajectories, "Relative trajectory (LVLH)"),
                    use_container_width=True)

    c = st.columns(2)
    c[0].plotly_chart(fig_components(primary["times"], primary["states"], [0, 1, 2],
                                     ["x radial", "y along-track", "z cross-track"],
                                     "position [m]", "Position vs time"), use_container_width=True)
    c[1].plotly_chart(fig_components(primary["times"], primary["states"], [3, 4, 5],
                                     ["vx", "vy", "vz"], "velocity [m/s]", "Velocity vs time"),
                      use_container_width=True)
    st.plotly_chart(fig_control(primary["times"], primary["controls"]), use_container_width=True)

    if "comparison" in res:
        st.subheader("CW vs high-fidelity divergence")
        st.dataframe(res["comparison"]["summary"], use_container_width=True)
        st.plotly_chart(fig_error_over_time(res["times"], res["comparison"]["pos_error"],
                                            res["comparison"]["vel_error"]), use_container_width=True)

    st.subheader("Data export")
    download_csv(primary["history"], "research_run.csv", p + "dl")


# ==========================================================================
# Main
# ==========================================================================
def main():
    st.title("🛰️ Orbital Rendezvous Research Engine")
    st.caption("CW-based rendezvous guidance with a PD controller, validated against a "
               "higher-fidelity orbital model. This is a thin interface over the `src/` engine.")
    mode = st.sidebar.radio("Mode", ["Learning Mode", "Research Mode"], key="mode")
    st.sidebar.divider()
    if mode == "Learning Mode":
        learning_mode()
    else:
        research_mode()
    st.sidebar.caption("All physics lives in `src/`; this app only renders results.")


if __name__ == "__main__":
    main()
