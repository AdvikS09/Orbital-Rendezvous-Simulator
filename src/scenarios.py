"""Frontend-agnostic scenario API (the backend boundary for any UI).

This is a thin *orchestration* layer: it composes the simulation engine
(``orbit``, ``perturbations``, ``lvlh``, ``cw``, ``controller``, ``validation``)
into a small set of "run a scenario -> plain data" functions. Every function
returns only built-in / scientific-Python data types -- dicts of NumPy arrays and
pandas DataFrames -- with **no plotting, no Streamlit, and no file I/O**.

Why this module exists: it keeps all physics inside ``src/`` and gives the
frontend (the current Streamlit MVP, or any future web UI) a single function per
scenario to call. The frontend only renders the returned data, so the backend
stays reusable and frontend-independent. It does not duplicate any physics -- it
calls the existing engine functions.
"""

import numpy as np
import pandas as pd

from . import cw, orbit, validation
from .controller import PDController
from .perturbations import ForceModel

# Reasonable bounds so UI inputs cannot create runaway/empty simulations.
MAX_STEPS = 200_000


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def radius_from_altitude(altitude_km):
    """Circular-orbit radius [m] from an altitude in km."""
    return orbit.R_EARTH + float(altitude_km) * 1000.0


def _time_grid(duration_s, dt_s):
    """Uniform time grid [0, duration] with step dt (clamped to MAX_STEPS)."""
    duration_s = float(duration_s)
    dt_s = float(dt_s)
    n_steps = int(round(duration_s / dt_s))
    n_steps = max(1, min(n_steps, MAX_STEPS))
    return np.linspace(0.0, n_steps * dt_s, n_steps + 1)


def _rel_state(rel_pos, rel_vel):
    return np.array([*map(float, rel_pos), *map(float, rel_vel)], dtype=float)


def _relative_result(times, states, controls, model, mean_motion, period_s, target=None):
    """Standardised relative-motion result bundle (used by every relative scenario)."""
    pos_err = validation.position_error_series(states, target)
    vel_err = validation.velocity_error_series(states, target)
    ctrl_mag = validation.control_acceleration_series(controls)
    metrics = validation.rendezvous_metrics(times, states, controls, target=target, pos_tol=1.0)
    history = pd.DataFrame({
        "t_s": times,
        "x_m": states[:, 0], "y_m": states[:, 1], "z_m": states[:, 2],
        "vx_mps": states[:, 3], "vy_mps": states[:, 4], "vz_mps": states[:, 5],
        "ax_mps2": controls[:, 0], "ay_mps2": controls[:, 1], "az_mps2": controls[:, 2],
        "pos_err_m": pos_err, "vel_err_mps": vel_err, "ctrl_mag_mps2": ctrl_mag,
    })
    return {
        "model": model,
        "times": times,
        "states": states,
        "controls": controls,
        "pos_error": pos_err,
        "vel_error": vel_err,
        "control_mag": ctrl_mag,
        "metrics": metrics,
        "history": history,
        "mean_motion": mean_motion,
        "period_s": period_s,
    }


# --------------------------------------------------------------------------
# Absolute-orbit scenarios (basic circular / J2 / drag)
# --------------------------------------------------------------------------
def run_orbit(altitude_km=400.0, inclination_deg=51.6, n_orbits=2.0,
              use_j2=False, use_drag=False, steps_per_orbit=2000,
              cd=2.2, area=1.0, mass=100.0):
    """Propagate one circular orbit under a selectable force model.

    Returns the inertial trajectory, altitude history, energy drift and an
    orbit-validation table.
    """
    radius = radius_from_altitude(altitude_km)
    mu = orbit.MU_EARTH
    period = orbit.orbital_period(radius, mu)
    state0 = orbit.circular_orbit_state(radius, mu, np.radians(float(inclination_deg)))

    n_steps = max(1, min(int(round(n_orbits * steps_per_orbit)), MAX_STEPS))
    times = np.linspace(0.0, float(n_orbits) * period, n_steps + 1)

    fm = ForceModel(use_j2=use_j2, use_drag=use_drag, cd=cd, area=area, mass=mass)
    states = orbit.propagate(fm.derivatives, state0, times)

    altitude_series = (np.linalg.norm(states[:, :3], axis=1) - orbit.R_EARTH) / 1000.0
    return {
        "label": fm.label,
        "times": times,
        "states": states,                 # ECI, N x 6
        "period_s": period,
        "radius_m": radius,
        "altitude_km": altitude_series,
        "energy_drift": validation.relative_drift(validation.energy_series(states, mu)),
        "validation": validation.verify_orbit(times, states, mu, analytic_period=period),
    }


def compare_orbit_perturbation(altitude_km=400.0, inclination_deg=51.6, n_orbits=5.0,
                               use_j2=False, use_drag=True, steps_per_orbit=1000,
                               cd=2.2, area=1.0, mass=100.0):
    """Two-body baseline vs a perturbed orbit, with their position separation.

    Used by the J2-effect and drag-effect learning presets to show how far the
    perturbed trajectory diverges from the ideal two-body orbit over time.
    """
    base = run_orbit(altitude_km, inclination_deg, n_orbits, False, False, steps_per_orbit)
    pert = run_orbit(altitude_km, inclination_deg, n_orbits, use_j2, use_drag,
                     steps_per_orbit, cd=cd, area=area, mass=mass)
    sep = np.linalg.norm(pert["states"][:, :3] - base["states"][:, :3], axis=1)
    return {
        "times": base["times"],
        "baseline": base,
        "perturbed": pert,
        "separation_m": sep,
    }


# --------------------------------------------------------------------------
# Relative-motion scenarios (CW, PD rendezvous, gain sweep, comparison)
# --------------------------------------------------------------------------
def run_cw_relative(rel_pos, rel_vel, altitude_km=400.0, duration_s=None, dt_s=10.0):
    """Uncontrolled (natural) CW relative motion from an initial relative state."""
    radius = radius_from_altitude(altitude_km)
    n = cw.cw_mean_motion(radius, orbit.MU_EARTH)
    period = orbit.orbital_period(radius)
    times = _time_grid(period if duration_s is None else duration_s, dt_s)
    states, controls = cw.propagate_cw(_rel_state(rel_pos, rel_vel), times, n, None)
    return _relative_result(times, states, controls, "CW (uncontrolled)", n, period)


def run_cw_rendezvous(rel_pos, rel_vel, altitude_km=400.0, duration_s=None, dt_s=5.0,
                      kp=None, kd=None, max_accel=0.05, omega_n_factor=6.0):
    """PD-controlled CW rendezvous to the LVLH origin.

    If ``kp``/``kd`` are omitted a critically damped controller at
    ``omega_n_factor * n`` is used.
    """
    radius = radius_from_altitude(altitude_km)
    n = cw.cw_mean_motion(radius, orbit.MU_EARTH)
    period = orbit.orbital_period(radius)
    times = _time_grid(period if duration_s is None else duration_s, dt_s)
    if kp is None or kd is None:
        pd_ctrl = PDController.critically_damped(omega_n_factor * n, max_accel=max_accel)
    else:
        pd_ctrl = PDController(float(kp), float(kd), max_accel=max_accel)
    states, controls = cw.propagate_cw(_rel_state(rel_pos, rel_vel), times, n, pd_ctrl)
    return _relative_result(times, states, controls, "CW + PD", n, period)


def run_gain_sweep(rel_pos, rel_vel, altitude_km=400.0, duration_s=None, dt_s=5.0,
                   kp_values=None, kd_values=None, max_accel=None):
    """Sweep Kp x Kd over a CW rendezvous; returns the metrics table."""
    radius = radius_from_altitude(altitude_km)
    n = cw.cw_mean_motion(radius, orbit.MU_EARTH)
    period = orbit.orbital_period(radius)
    times = _time_grid(period if duration_s is None else duration_s, dt_s)
    if kp_values is None:
        kp_values = [1.0e-5, 4.6e-5, 1.0e-4, 2.0e-4]
    if kd_values is None:
        kd_values = [5.0e-3, 1.0e-2, 1.36e-2, 3.0e-2]
    table = validation.gain_sweep(
        _rel_state(rel_pos, rel_vel), times, n,
        list(kp_values), list(kd_values), max_accel=max_accel, pos_tol=1.0,
    )
    return {
        "table": table,
        "kp_values": [float(v) for v in kp_values],
        "kd_values": [float(v) for v in kd_values],
        "mean_motion": n,
        "period_s": period,
    }


def run_relative_scenario(rel_pos, rel_vel, altitude_km=400.0, inclination_deg=51.6,
                          duration_s=None, dt_s=5.0, model="comparison",
                          controlled=True, kp=None, kd=None, max_accel=0.05,
                          omega_n_factor=6.0, use_j2=True, use_drag=True,
                          cd=2.2, area=1.0, mass=100.0):
    """Unified relative-motion runner for the CW-vs-high-fidelity study and
    Research Mode.

    ``model`` selects ``"cw"``, ``"high_fidelity"``, or ``"comparison"``. When
    ``controlled`` is True the same PD controller drives both models. The result
    contains a ``"cw"`` and/or ``"high_fidelity"`` bundle (each shaped like
    :func:`_relative_result`), a ``"comparison"`` divergence block when both are
    present, and a ``"primary"`` bundle for metrics/CSV.
    """
    radius = radius_from_altitude(altitude_km)
    mu = orbit.MU_EARTH
    n = cw.cw_mean_motion(radius, mu)
    period = orbit.orbital_period(radius, mu)
    times = _time_grid(period if duration_s is None else duration_s, dt_s)
    target0 = orbit.circular_orbit_state(radius, mu, np.radians(float(inclination_deg)))
    rel0 = _rel_state(rel_pos, rel_vel)
    fm = ForceModel(use_j2=use_j2, use_drag=use_drag, cd=cd, area=area, mass=mass)

    if controlled:
        if kp is None or kd is None:
            ctrl = PDController.critically_damped(omega_n_factor * n, max_accel=max_accel)
        else:
            ctrl = PDController(float(kp), float(kd), max_accel=max_accel)
    else:
        ctrl = None

    result = {"model": model, "times": times, "mean_motion": n, "period_s": period,
              "force_model": fm.label, "controlled": controlled}

    if model in ("cw", "comparison"):
        cw_states, cw_controls = cw.propagate_cw(rel0, times, n, ctrl)
        result["cw"] = _relative_result(times, cw_states, cw_controls, "CW", n, period)
    if model in ("high_fidelity", "comparison"):
        hf_states, hf_controls, _, _ = validation.high_fidelity_relative(target0, rel0, times, fm, ctrl)
        result["high_fidelity"] = _relative_result(
            times, hf_states, hf_controls, f"high-fidelity ({fm.label})", n, period)
    if model == "comparison":
        cw_s, hf_s = result["cw"]["states"], result["high_fidelity"]["states"]
        pos_err = np.linalg.norm(cw_s[:, :3] - hf_s[:, :3], axis=1)
        vel_err = np.linalg.norm(cw_s[:, 3:] - hf_s[:, 3:], axis=1)
        result["comparison"] = {
            "pos_error": pos_err,
            "vel_error": vel_err,
            "summary": pd.DataFrame([{
                "force_model": fm.label,
                "max_pos_error_m": float(pos_err.max()),
                "final_pos_error_m": float(pos_err[-1]),
                "max_vel_error_mps": float(vel_err.max()),
                "final_vel_error_mps": float(vel_err[-1]),
            }]),
        }

    result["primary"] = result.get("high_fidelity") or result.get("cw")
    return result


__all__ = [
    "radius_from_altitude",
    "run_orbit",
    "compare_orbit_perturbation",
    "run_cw_relative",
    "run_cw_rendezvous",
    "run_gain_sweep",
    "run_relative_scenario",
]
