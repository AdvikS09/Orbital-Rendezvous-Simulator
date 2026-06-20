"""Diagnostics: conservation tracking, error metrics and validation tables.

All tabular outputs are returned as :class:`pandas.DataFrame` objects so they
can be printed, saved to CSV, or further processed by the calling scripts.
"""

import numpy as np
import pandas as pd

from .constants import MU_EARTH
from .physics import specific_angular_momentum, specific_orbital_energy


def _trapezoid(y, x):
    """Trapezoidal integral, compatible with both old and new NumPy names."""
    func = getattr(np, "trapezoid", None)
    if func is None:  # NumPy < 2.0
        func = np.trapz
    return func(y, x)


# --- Conservation tracking ------------------------------------------------


def energy_series(states, mu=MU_EARTH):
    """Specific orbital energy at each sample [J/kg]."""
    return np.array([specific_orbital_energy(s, mu) for s in states])


def angular_momentum_series(states):
    """Specific angular momentum (z component) at each sample [m^2/s]."""
    return np.array([specific_angular_momentum(s) for s in states])


def relative_drift(series):
    """Drift of a series relative to its initial value: (s - s0) / |s0|."""
    series = np.asarray(series, dtype=float)
    ref = series[0]
    if ref == 0.0:
        return series - ref
    return (series - ref) / abs(ref)


def max_abs_relative_drift(series):
    """Maximum absolute relative drift over the series."""
    return float(np.max(np.abs(relative_drift(series))))


def measure_period(t, states):
    """Estimate the orbital period from the unwrapped polar angle.

    Returns ``np.nan`` if the trajectory does not complete a full revolution.
    """
    t = np.asarray(t, dtype=float)
    pos = np.asarray(states)[:, :2]
    theta = np.unwrap(np.arctan2(pos[:, 1], pos[:, 0]))
    if theta[-1] < theta[0]:
        theta = -theta  # treat clockwise orbits as increasing angle
    target = theta[0] + 2.0 * np.pi
    if theta[-1] < target:
        return np.nan
    return float(np.interp(target, theta, t))


# --- Integrator validation table ------------------------------------------


def integrator_comparison_table(results, mu=MU_EARTH, true_period=None):
    """Build a table of conservation / period accuracy for several integrators.

    ``results`` maps a method name to a :class:`~src.simulation.SimulationResult`.
    """
    rows = []
    for name, res in results.items():
        energy = energy_series(res.states, mu)
        ang_mom = angular_momentum_series(res.states)
        period = measure_period(res.t, res.states)
        row = {
            "method": name,
            "max_energy_drift_rel": max_abs_relative_drift(energy),
            "final_energy_drift_rel": abs(relative_drift(energy)[-1]),
            "max_ang_mom_drift_rel": max_abs_relative_drift(ang_mom),
            "measured_period_s": period,
        }
        if true_period is not None and np.isfinite(period):
            row["period_rel_err"] = abs(period - true_period) / true_period
        else:
            row["period_rel_err"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --- Rendezvous error metrics ---------------------------------------------


def position_error_series(states, target=None):
    """Norm of the position error at each sample [m]."""
    pos = np.asarray(states)[:, :2]
    tgt = np.zeros(2) if target is None else np.asarray(target, dtype=float)[:2]
    return np.linalg.norm(pos - tgt, axis=1)


def velocity_error_series(states, target=None):
    """Norm of the velocity error at each sample [m/s]."""
    vel = np.asarray(states)[:, 2:]
    tgt = np.zeros(2) if target is None else np.asarray(target, dtype=float)[2:]
    return np.linalg.norm(vel - tgt, axis=1)


def control_acceleration_series(controls):
    """Norm of the commanded acceleration at each sample [m/s^2]."""
    return np.linalg.norm(np.asarray(controls, dtype=float), axis=1)


def cumulative_delta_v(t, controls):
    """Total delta-v expended: integral of |u| dt [m/s]."""
    accel = control_acceleration_series(controls)
    return float(_trapezoid(accel, np.asarray(t, dtype=float)))


def settling_time(t, error, tol):
    """Time after which ``error`` stays within ``tol`` for the rest of the run.

    Returns ``np.nan`` if it never settles.
    """
    t = np.asarray(t, dtype=float)
    error = np.asarray(error, dtype=float)
    above = np.where(error > tol)[0]
    if above.size == 0:
        return 0.0
    last = above[-1]
    if last >= len(t) - 1:
        return np.nan
    return float(t[last + 1])


def rendezvous_metrics_table(result, target=None, pos_tol=1.0, vel_tol=0.01):
    """Summarise a controlled rendezvous as a one-row table."""
    perr = position_error_series(result.states, target)
    verr = velocity_error_series(result.states, target)
    controls = (
        result.controls
        if result.controls is not None
        else np.zeros((len(result.t), 2))
    )
    accel = control_acceleration_series(controls)
    row = {
        "initial_pos_error_m": float(perr[0]),
        "final_pos_error_m": float(perr[-1]),
        "final_vel_error_mps": float(verr[-1]),
        "settling_time_s": settling_time(result.t, perr, pos_tol),
        "peak_accel_mps2": float(np.max(accel)),
        "delta_v_mps": cumulative_delta_v(result.t, controls),
        "pos_tol_m": pos_tol,
        "vel_tol_mps": vel_tol,
    }
    return pd.DataFrame([row])


__all__ = [
    "energy_series",
    "angular_momentum_series",
    "relative_drift",
    "max_abs_relative_drift",
    "measure_period",
    "integrator_comparison_table",
    "position_error_series",
    "velocity_error_series",
    "control_acceleration_series",
    "cumulative_delta_v",
    "settling_time",
    "rendezvous_metrics_table",
]
