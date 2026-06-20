"""Experiment 1 -- closed-loop CW rendezvous with a PD controller.

A chaser starting from a configurable relative state is driven to the target
(LVLH origin, zero relative velocity) by a critically damped PD controller with
thrust saturation. Generates the relative-trajectory, position, velocity and
control-effort plots and saves the full time history plus a metrics table.

Run:  python experiments/run_cw_rendezvous.py
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "outputs", "figures")
DAT = os.path.join(ROOT, "outputs", "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DAT, exist_ok=True)

from src import controller, cw, orbit, plotting, validation


def main():
    # --- Scenario configuration (SI units) ---
    radius = orbit.DEFAULT_RADIUS               # 400 km circular LEO
    n = cw.cw_mean_motion(radius)
    period = orbit.orbital_period(radius)

    # Configurable initial relative state [x, y, z, vx, vy, vz] in the LVLH frame
    # (radial, along-track, cross-track).
    rel0 = np.array([200.0, -300.0, 80.0, 0.0, 0.0, 0.0])

    omega_n = 6.0 * n          # closed-loop bandwidth (>> orbital rate)
    max_accel = 0.05           # m/s^2 thrust-acceleration limit
    pd_ctrl = controller.PDController.critically_damped(omega_n, max_accel=max_accel)

    times = np.linspace(0.0, period, 4001)
    states, controls = cw.propagate_cw(rel0, times, n, pd_ctrl)

    pos_err = validation.position_error_series(states)
    vel_err = validation.velocity_error_series(states)
    metrics = validation.rendezvous_metrics(times, states, controls, pos_tol=1.0)

    # --- Save data ---
    history = pd.DataFrame({
        "t_s": times,
        "x_m": states[:, 0], "y_m": states[:, 1], "z_m": states[:, 2],
        "vx_mps": states[:, 3], "vy_mps": states[:, 4], "vz_mps": states[:, 5],
        "ax_mps2": controls[:, 0], "ay_mps2": controls[:, 1], "az_mps2": controls[:, 2],
        "pos_err_m": pos_err, "vel_err_mps": vel_err,
    })
    hist_path = os.path.join(DAT, "cw_rendezvous_timeseries.csv")
    metrics_path = os.path.join(DAT, "cw_rendezvous_metrics.csv")
    history.to_csv(hist_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    print(f"CW rendezvous: n = {n:.6e} rad/s, period = {period:.1f} s")
    print(f"omega_n = {omega_n:.4e} rad/s (= {omega_n/n:.1f} n), max_accel = {max_accel} m/s^2\n")
    print(metrics.T.to_string(header=False))

    # --- Figures ---
    figs = [
        plotting.plot_lvlh_trajectory(
            [("chaser", states)], os.path.join(FIG, "cw_rendezvous_lvlh_trajectory.png"),
            title="CW rendezvous: relative trajectory (LVLH)"),
        plotting.plot_position_components(
            times, states, os.path.join(FIG, "cw_rendezvous_position.png"),
            title="CW rendezvous: relative position vs time"),
        plotting.plot_velocity_components(
            times, states, os.path.join(FIG, "cw_rendezvous_velocity.png"),
            title="CW rendezvous: relative velocity vs time"),
        plotting.plot_control_acceleration(
            times, controls, os.path.join(FIG, "cw_rendezvous_control.png"),
            title="CW rendezvous: control acceleration vs time"),
    ]
    print()
    for p in (hist_path, metrics_path, *figs):
        print(f"Wrote: {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
