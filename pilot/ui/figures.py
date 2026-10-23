from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from orbital_core import orbit

EARTH_R_KM = orbit.R_EARTH / 1000.0

PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
DASHES = ("solid", "dash", "dot", "dashdot")

_LAYOUT = dict(
    margin=dict(l=10, r=10, t=48, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
)


def _style(fig, title, height=380):
    fig.update_layout(title=title, height=height, **_LAYOUT)
    return fig


def _scene(fig, title, x, y, z, height=520):
    fig.update_layout(
        title=title, height=height,
        scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z, aspectmode="data"),
        margin=dict(l=0, r=0, t=48, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


def orbit_3d(states, title="Orbit (Earth-centred inertial frame)"):
    positions = np.asarray(states)[:, :3] / 1000.0
    fig = go.Figure()
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:16j]
    fig.add_surface(
        x=EARTH_R_KM * np.cos(u) * np.sin(v),
        y=EARTH_R_KM * np.sin(u) * np.sin(v),
        z=EARTH_R_KM * np.cos(v),
        colorscale="Blues", opacity=0.28, showscale=False, hoverinfo="skip",
        name="Earth")
    fig.add_scatter3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                      mode="lines", line=dict(color=PALETTE[1], width=4), name="orbit")
    fig.add_scatter3d(x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
                      mode="markers", marker=dict(size=5, color=PALETTE[6]),
                      name="start")
    return _scene(fig, title, "x [km]", "y [km]", "z [km]")


def lvlh_trajectory(trajectories, title="Relative trajectory (target-centred LVLH frame)"):
    fig = go.Figure()
    for index, (label, states) in enumerate(trajectories):
        positions = np.asarray(states)[:, :3]
        fig.add_scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], mode="lines",
            line=dict(width=5, color=PALETTE[index % len(PALETTE)],
                      dash=DASHES[index % len(DASHES)]),
            name=label)
        fig.add_scatter3d(x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
                          mode="markers", marker=dict(size=4, color=PALETTE[6]),
                          name=f"{label} start", showlegend=False)
    fig.add_scatter3d(x=[0], y=[0], z=[0], mode="markers",
                      marker=dict(size=7, color=PALETTE[2], symbol="diamond"),
                      name="target")
    return _scene(fig, title, "radial x [m]", "along-track y [m]", "cross-track z [m]")


def series(times, entries, ylabel, title, log_y=False, height=340):
    fig = go.Figure()
    for index, (label, values) in enumerate(entries):
        fig.add_scatter(x=times, y=values, mode="lines", name=label,
                        line=dict(color=PALETTE[index % len(PALETTE)],
                                  dash=DASHES[index % len(DASHES)], width=2.5))
    fig.update_xaxes(title_text="time [s]")
    fig.update_yaxes(title_text=ylabel, type="log" if log_y else "linear")
    return _style(fig, title, height)


def components(times, states, indices, labels, ylabel, title):
    states = np.asarray(states)
    return series(times, [(label, states[:, index])
                          for index, label in zip(indices, labels)], ylabel, title)


def control_history(times, controls, title="Commanded thrust acceleration"):
    controls = np.asarray(controls)
    entries = [("radial a_x", controls[:, 0]),
               ("along-track a_y", controls[:, 1]),
               ("cross-track a_z", controls[:, 2]),
               ("magnitude |u|", np.linalg.norm(controls, axis=1))]
    fig = series(times, entries, "acceleration [m/s²]", title)
    fig.data[3].line.width = 3.5
    fig.data[3].line.color = PALETTE[6]
    return fig


def dual_error(times, position_error, velocity_error,
               title="Difference between the two models"):
    fig = go.Figure()
    fig.add_scatter(x=times, y=position_error, mode="lines", name="position [m]",
                    line=dict(color=PALETTE[1], width=2.5))
    fig.add_scatter(x=times, y=velocity_error, mode="lines", name="velocity [m/s]",
                    line=dict(color=PALETTE[0], width=2.5, dash="dash"), yaxis="y2")
    fig.update_layout(
        title=title, height=380,
        xaxis=dict(title="time [s]"),
        yaxis=dict(title="position difference [m]"),
        yaxis2=dict(title="velocity difference [m/s]", overlaying="y", side="right"),
        **{k: v for k, v in _LAYOUT.items() if k != "hovermode"})
    return fig


def gain_heatmap(table, metric, title, colourbar_title=""):
    grid = table.pivot(index="kd", columns="kp", values=metric)
    fig = go.Figure(go.Heatmap(
        z=grid.to_numpy(),
        x=[f"{value:.1e}" for value in grid.columns],
        y=[f"{value:.1e}" for value in grid.index],
        colorscale="Viridis",
        colorbar=dict(title=colourbar_title or metric),
        hovertemplate="Kp=%{x}<br>Kd=%{y}<br>value=%{z:.4g}<extra></extra>"))
    fig.update_xaxes(title_text="Kp (proportional gain) [1/s²]")
    fig.update_yaxes(title_text="Kd (derivative gain) [1/s]")
    return _style(fig, title, height=400)


def _sphere(radius, colour, opacity, centre=(0.0, 0.0, 0.0), name=""):
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:12j]
    return go.Surface(
        x=centre[0] + radius * np.cos(u) * np.sin(v),
        y=centre[1] + radius * np.sin(u) * np.sin(v),
        z=centre[2] + radius * np.cos(v),
        colorscale=[[0, colour], [1, colour]], opacity=opacity, showscale=False,
        hoverinfo="skip", name=name)


def proximity_3d(chaser_states, debris_states, keep_out_radius, analysis,
                 title="Chaser path, keep-out zone and debris (LVLH)"):
    chaser = np.asarray(chaser_states)[:, :3]
    fig = go.Figure()
    fig.add_trace(_sphere(keep_out_radius, PALETTE[1], 0.16, name="keep-out zone"))
    fig.add_scatter3d(x=chaser[:, 0], y=chaser[:, 1], z=chaser[:, 2], mode="lines",
                      line=dict(color=PALETTE[0], width=5), name="chaser")
    fig.add_scatter3d(x=[chaser[0, 0]], y=[chaser[0, 1]], z=[chaser[0, 2]],
                      mode="markers", marker=dict(size=5, color=PALETTE[6]),
                      name="chaser start")
    fig.add_scatter3d(x=[0], y=[0], z=[0], mode="markers",
                      marker=dict(size=7, color=PALETTE[2], symbol="diamond"),
                      name="target")
    if debris_states is not None:
        debris = np.asarray(debris_states)[:, :3]
        fig.add_scatter3d(x=debris[:, 0], y=debris[:, 1], z=debris[:, 2], mode="lines",
                          line=dict(color=PALETTE[4], width=4, dash="dot"),
                          name="debris")
        if "dist_to_debris" in analysis:
            index = int(np.argmin(analysis["dist_to_debris"]))
            fig.add_scatter3d(
                x=[chaser[index, 0], debris[index, 0]],
                y=[chaser[index, 1], debris[index, 1]],
                z=[chaser[index, 2], debris[index, 2]],
                mode="lines+markers", line=dict(color=PALETTE[3], width=5),
                marker=dict(size=4, color=PALETTE[3]), name="closest approach")
    return _scene(fig, title, "radial x [m]", "along-track y [m]", "cross-track z [m]")


def convergence_plot(rows, title="Integration error against step size"):
    steps = np.array([row["dt_s"] for row in rows], dtype=float)
    errors = np.array([row["max_position_error_m"] for row in rows], dtype=float)

    fig = go.Figure()
    fig.add_scatter(x=steps, y=errors, mode="lines+markers", name="measured error",
                    line=dict(color=PALETTE[0], width=3),
                    marker=dict(size=9, color=PALETTE[0]))
    reference = errors[-1] * (steps / steps[-1]) ** 4
    fig.add_scatter(x=steps, y=reference, mode="lines",
                    name="ideal 4th-order slope",
                    line=dict(color=PALETTE[6], width=2, dash="dash"))
    fig.update_xaxes(title_text="timestep [s]", type="log")
    fig.update_yaxes(title_text="max position error vs exact solution [m]", type="log")
    return _style(fig, title, height=400)


__all__ = ["orbit_3d", "lvlh_trajectory", "series", "components", "control_history",
           "dual_error", "gain_heatmap", "proximity_3d", "convergence_plot",
           "PALETTE", "EARTH_R_KM"]
