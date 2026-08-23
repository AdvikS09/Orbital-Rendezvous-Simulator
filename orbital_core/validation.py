"""Validation and analysis: conservation checks, error metrics, studies.

Provides:
* two-body energy / angular-momentum / period validation,
* rendezvous error metrics (position, velocity, control effort, delta-v),
* a high-fidelity relative-motion propagator (full non-linear orbits expressed
  in the target LVLH frame, with optional control on the deputy),
* a PD gain-tuning sweep,
* a CW vs high-fidelity comparison.

Tabular results are returned as pandas DataFrames.
"""

import numpy as np
import pandas as pd

from . import cw, lvlh, orbit
from .controller import PDController


def _trapezoid(y, x):
    func = getattr(np, "trapezoid", None) or np.trapz
    return func(y, x)


# --- Conservation / orbit validation --------------------------------------
def energy_series(states, mu=orbit.MU_EARTH):
    return np.array([orbit.specific_energy(s, mu) for s in states])


def angular_momentum_series(states):
    return np.array([orbit.specific_angular_momentum(s) for s in states])


def relative_drift(series):
    series = np.asarray(series, dtype=float)
    ref = series[0]
    return series - ref if ref == 0.0 else (series - ref) / abs(ref)


def verify_orbit(times, states, mu=orbit.MU_EARTH, analytic_period=None):
    """Return a one-row DataFrame of energy/momentum drift and period accuracy."""
    energy = energy_series(states, mu)
    ang_mom = angular_momentum_series(states)
    measured = orbit.measure_period(times, states)
    row = {
        "max_energy_drift_rel": float(np.max(np.abs(relative_drift(energy)))),
        "max_ang_mom_drift_rel": float(np.max(np.abs(relative_drift(ang_mom)))),
        "measured_period_s": measured,
        "analytic_period_s": analytic_period if analytic_period is not None else np.nan,
    }
    if analytic_period is not None and np.isfinite(measured):
        row["period_rel_err"] = abs(measured - analytic_period) / analytic_period
    else:
        row["period_rel_err"] = np.nan
    return pd.DataFrame([row])


# --- Rendezvous error metrics ---------------------------------------------
def position_error_series(states, target=None):
    pos = np.asarray(states)[:, :3]
    tgt = np.zeros(3) if target is None else np.asarray(target, dtype=float)[:3]
    return np.linalg.norm(pos - tgt, axis=1)


def velocity_error_series(states, target=None):
    vel = np.asarray(states)[:, 3:]
    tgt = np.zeros(3) if target is None else np.asarray(target, dtype=float)[3:]
    return np.linalg.norm(vel - tgt, axis=1)


def control_acceleration_series(controls):
    return np.linalg.norm(np.asarray(controls, dtype=float), axis=1)


def cumulative_delta_v(times, controls):
    """Total delta-v = integral of |u| dt [m/s]."""
    return float(_trapezoid(control_acceleration_series(controls), np.asarray(times, float)))


def convergence_time(times, pos_error, tol):
    """Time after which the position error stays within ``tol`` for good.

    Returns 0.0 if always within tolerance, ``np.nan`` if it never converges.
    """
    times = np.asarray(times, dtype=float)
    pos_error = np.asarray(pos_error, dtype=float)
    above = np.where(pos_error > tol)[0]
    if above.size == 0:
        return 0.0
    last = above[-1]
    if last >= times.size - 1:
        return np.nan
    return float(times[last + 1])


def rendezvous_metrics(times, states, controls, target=None, pos_tol=1.0):
    """One-row DataFrame summarising a controlled rendezvous run."""
    perr = position_error_series(states, target)
    verr = velocity_error_series(states, target)
    accel = control_acceleration_series(controls)
    return pd.DataFrame([{
        "initial_pos_error_m": float(perr[0]),
        "final_pos_error_m": float(perr[-1]),
        "final_vel_error_mps": float(verr[-1]),
        "convergence_time_s": convergence_time(times, perr, pos_tol),
        "max_control_accel_mps2": float(np.max(accel)),
        "delta_v_mps": cumulative_delta_v(times, controls),
    }])


# --- High-fidelity relative propagation -----------------------------------
def high_fidelity_relative(target_state0, rel_state0, times, force_model, controller=None):
    """Propagate target + deputy under ``force_model`` and express in LVLH.

    The target is uncontrolled; if ``controller`` is given its LVLH command is
    rotated into the inertial frame and applied to the deputy, mirroring the CW
    rendezvous scenario. Returns ``(rel_states, controls, target_states,
    deputy_states)``.
    """
    target_state0 = np.asarray(target_state0, dtype=float)
    deputy_state0 = lvlh.lvlh_to_inertial(target_state0, rel_state0)
    times = np.asarray(times, dtype=float)

    def combined_deriv(z, t):
        ts, ds = z[:6], z[6:]
        a_t = force_model.acceleration(ts, t)
        a_d = force_model.acceleration(ds, t)
        if controller is not None:
            rel = lvlh.inertial_to_lvlh(ts, ds)
            u_lvlh = np.asarray(controller(rel, t), dtype=float)
            a_d = a_d + lvlh.lvlh_rotation_matrix(ts).T @ u_lvlh
        return np.concatenate((ts[3:], a_t, ds[3:], a_d))

    z0 = np.concatenate((target_state0, deputy_state0))
    z = orbit.propagate(combined_deriv, z0, times)
    target_states, deputy_states = z[:, :6], z[:, 6:]

    rel_states = np.empty((times.size, 6), dtype=float)
    controls = np.zeros((times.size, 3), dtype=float)
    for k in range(times.size):
        rel_states[k] = lvlh.inertial_to_lvlh(target_states[k], deputy_states[k])
        if controller is not None:
            controls[k] = np.asarray(controller(rel_states[k], times[k]), dtype=float)
    return rel_states, controls, target_states, deputy_states


# --- PD gain-tuning sweep -------------------------------------------------
def gain_sweep(initial_rel_state, times, n, kp_values, kd_values,
               target=None, max_accel=None, pos_tol=1.0):
    """Sweep Kp x Kd, running a CW rendezvous for each combination.

    Returns a DataFrame with one row per (kp, kd): final position error, final
    velocity error, convergence time, max control acceleration and delta-v.
    """
    rows = []
    for kp in kp_values:
        for kd in kd_values:
            pd_ctrl = PDController(kp, kd, target=target, max_accel=max_accel)
            states, controls = cw.propagate_cw(initial_rel_state, times, n, pd_ctrl)
            perr = position_error_series(states, target)
            verr = velocity_error_series(states, target)
            accel = control_acceleration_series(controls)
            rows.append({
                "kp": kp,
                "kd": kd,
                "final_pos_error_m": float(perr[-1]),
                "final_vel_error_mps": float(verr[-1]),
                "convergence_time_s": convergence_time(times, perr, pos_tol),
                "max_control_accel_mps2": float(np.max(accel)),
                "delta_v_mps": cumulative_delta_v(times, controls),
            })
    return pd.DataFrame(rows)


# --- CW vs high-fidelity comparison ---------------------------------------
def compare_cw_high_fidelity(target_state0, rel_state0, times, force_model,
                             n=None, controller=None):
    """Run the same scenario in CW and high-fidelity; quantify the divergence.

    Returns a dict with the CW and high-fidelity relative state histories, the
    position / velocity error-over-time series, and a one-row summary DataFrame.
    """
    target_state0 = np.asarray(target_state0, dtype=float)
    if n is None:
        n = cw.cw_mean_motion(np.linalg.norm(target_state0[:3]), force_model.mu)

    cw_states, cw_controls = cw.propagate_cw(rel_state0, times, n, controller)
    hf_states, hf_controls, _, _ = high_fidelity_relative(
        target_state0, rel_state0, times, force_model, controller
    )

    pos_err = np.linalg.norm(cw_states[:, :3] - hf_states[:, :3], axis=1)
    vel_err = np.linalg.norm(cw_states[:, 3:] - hf_states[:, 3:], axis=1)

    summary = pd.DataFrame([{
        "force_model": force_model.label,
        "initial_sep_m": float(np.linalg.norm(np.asarray(rel_state0)[:3])),
        "max_pos_error_m": float(pos_err.max()),
        "final_pos_error_m": float(pos_err[-1]),
        "max_vel_error_mps": float(vel_err.max()),
        "final_vel_error_mps": float(vel_err[-1]),
    }])

    return {
        "cw_states": cw_states,
        "hf_states": hf_states,
        "cw_controls": cw_controls,
        "hf_controls": hf_controls,
        "pos_error": pos_err,
        "vel_error": vel_err,
        "summary": summary,
    }


__all__ = [
    "energy_series", "angular_momentum_series", "relative_drift", "verify_orbit",
    "position_error_series", "velocity_error_series", "control_acceleration_series",
    "cumulative_delta_v", "convergence_time", "rendezvous_metrics",
    "high_fidelity_relative", "gain_sweep", "compare_cw_high_fidelity",
]
