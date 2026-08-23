"""Headless matplotlib plotting helpers for the rendezvous engine.

The Agg backend is selected on import so experiments render and save figures
without any display. Every helper saves a PNG to ``path`` and returns it.

Relative states are LVLH 6-vectors ``[x, y, z, vx, vy, vz]`` (x=radial,
y=along-track, z=cross-track).
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  (enables 3d projection)


def _save(fig, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lvlh_trajectory(trajectories, path, title="Relative trajectory (LVLH)"):
    """3D LVLH relative trajectory. ``trajectories`` = list of (label, states)."""
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    for label, states in trajectories:
        p = np.asarray(states)[:, :3]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], lw=1.3, label=label)
        ax.scatter(p[0, 0], p[0, 1], p[0, 2], color="black", s=25)
    ax.scatter(0, 0, 0, color="green", marker="*", s=140, label="target")
    ax.set_xlabel("radial x [m]")
    ax.set_ylabel("along-track y [m]")
    ax.set_zlabel("cross-track z [m]")
    ax.set_title(title)
    ax.legend(loc="upper right")
    return _save(fig, path)


def plot_position_components(t, states, path, title="Relative position vs time"):
    """x / y / z relative position components vs time."""
    states = np.asarray(states)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for j, name in enumerate(("x radial", "y along-track", "z cross-track")):
        ax.plot(t, states[:, j], lw=1.3, label=name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position [m]")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_velocity_components(t, states, path, title="Relative velocity vs time"):
    """vx / vy / vz relative velocity components vs time."""
    states = np.asarray(states)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for j, name in zip((3, 4, 5), ("vx radial", "vy along-track", "vz cross-track")):
        ax.plot(t, states[:, j], lw=1.3, label=name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("velocity [m/s]")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_control_acceleration(t, controls, path, title="Control acceleration vs time"):
    """Control acceleration components and magnitude vs time."""
    controls = np.asarray(controls)
    mag = np.linalg.norm(controls, axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for j, name in enumerate(("ax radial", "ay along-track", "az cross-track")):
        ax.plot(t, controls[:, j], lw=1.0, label=name)
    ax.plot(t, mag, lw=1.6, color="black", ls="--", label="|u|")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("control acceleration [m/s$^2$]")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_error_growth(t, pos_error, vel_error, path,
                      title="CW vs high-fidelity error over time"):
    """Position and velocity error between two models vs time (twin axes)."""
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    l1, = ax1.plot(t, pos_error, lw=1.4, color="tab:red", label="position error")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("position error [m]", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    l2, = ax2.plot(t, vel_error, lw=1.4, color="tab:blue", label="velocity error")
    ax2.set_ylabel("velocity error [m/s]", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_title(title)
    ax1.legend(handles=[l1, l2], loc="upper left")
    return _save(fig, path)


def plot_error_curves(t, series, path, ylabel="error", title="", logy=False):
    """Plot multiple error curves vs time. ``series`` = list of (label, values)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, values in series:
        ax.plot(np.asarray(t), np.asarray(values), lw=1.3, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_gain_comparison(df, path, metrics=("convergence_time_s", "delta_v_mps"),
                         title="PD gain sweep"):
    """Heatmaps of sweep metrics over the Kp x Kd grid.

    ``df`` is the output of :func:`src.validation.gain_sweep` (columns kp, kd,
    plus the metric columns).
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 5.0))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        grid = df.pivot(index="kd", columns="kp", values=metric)
        kp_vals = grid.columns.to_numpy()
        kd_vals = grid.index.to_numpy()
        data = grid.to_numpy()

        im = ax.imshow(data, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(kp_vals)))
        ax.set_xticklabels([f"{v:.1e}" for v in kp_vals], rotation=45, ha="right")
        ax.set_yticks(range(len(kd_vals)))
        ax.set_yticklabels([f"{v:.1e}" for v in kd_vals])
        ax.set_xlabel("Kp")
        ax.set_ylabel("Kd")
        ax.set_title(metric)
        for i in range(data.shape[0]):
            for k in range(data.shape[1]):
                val = data[i, k]
                if np.isfinite(val):
                    ax.text(k, i, f"{val:.2g}", ha="center", va="center",
                            color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)


__all__ = [
    "plot_lvlh_trajectory",
    "plot_position_components",
    "plot_velocity_components",
    "plot_control_acceleration",
    "plot_error_growth",
    "plot_error_curves",
    "plot_gain_comparison",
]
