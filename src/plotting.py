"""Matplotlib plotting helpers (headless / non-interactive).

The Agg backend is selected on import so the scripts can run and save figures
without any display or GUI. Every helper saves a PNG to ``path`` and returns it.
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .constants import R_EARTH  # noqa: E402


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _save(fig, path):
    _ensure_parent(path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_orbit_trajectory(states, path, earth_radius=R_EARTH, title="Orbit trajectory"):
    """Plot an inertial-frame orbit (km) with the Earth disc for scale."""
    pos = np.asarray(states)[:, :2] / 1000.0  # m -> km
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    earth = plt.Circle((0, 0), earth_radius / 1000.0, color="steelblue", alpha=0.5, zorder=0)
    ax.add_patch(earth)
    ax.plot(pos[:, 0], pos[:, 1], lw=1.0, color="darkorange", label="trajectory")
    ax.plot(pos[0, 0], pos[0, 1], "o", color="black", label="start")
    ax.set_aspect("equal")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_relative_trajectory(
    trajectories, path, obstacles=None, title="Relative trajectory (LVLH)"
):
    """Plot one or more LVLH relative trajectories.

    ``trajectories`` is a list of ``(label, states)`` tuples. The target sits at
    the origin; optional ``obstacles`` (objects with ``center``/``radius``) are
    drawn as keep-out discs.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, states in trajectories:
        pos = np.asarray(states)[:, :2]
        ax.plot(pos[:, 0], pos[:, 1], lw=1.3, label=label)
        ax.plot(pos[0, 0], pos[0, 1], "o", ms=5, color="black")
    ax.plot(0, 0, "*", ms=14, color="green", label="target")

    if obstacles:
        for i, ob in enumerate(obstacles):
            koz = plt.Circle(
                tuple(ob.center), ob.radius, color="red", alpha=0.25,
                label="keep-out zone" if i == 0 else None,
            )
            ax.add_patch(koz)

    ax.set_aspect("equal")
    ax.set_xlabel("radial  x [m]")
    ax.set_ylabel("along-track  y [m]")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_time_series(t, series, path, ylabel="", title="", xlabel="time [s]", logy=False):
    """Plot one or more time series. ``series`` is a list of ``(label, values)``."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, values in series:
        ax.plot(np.asarray(t), np.asarray(values), lw=1.2, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_conservation(t, drift_series, path, title="Conservation drift"):
    """Plot relative drift of conserved quantities vs time.

    ``drift_series`` is a list of ``(label, relative_drift_values)``.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, values in drift_series:
        ax.plot(np.asarray(t), np.asarray(values), lw=1.2, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("relative drift  (q - q0) / |q0|")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


__all__ = [
    "plot_orbit_trajectory",
    "plot_relative_trajectory",
    "plot_time_series",
    "plot_conservation",
]
