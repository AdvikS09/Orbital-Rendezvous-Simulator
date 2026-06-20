"""Closed-loop PD rendezvous in the CW frame.

A chaser starting from an offset relative state is driven to the target (LVLH
origin, zero relative velocity) by a critically damped PD controller with thrust
(acceleration) saturation. Produces the required diagnostic plots and a metrics
CSV:

* relative trajectory (LVLH),
* position error vs time,
* velocity error vs time,
* control effort (commanded acceleration magnitude) vs time,
* rendezvous metrics table (final errors, settling time, delta-v, peak accel).

Run:  python scripts/run_pd_rendezvous.py
"""

import numpy as np

import _bootstrap  # noqa: F401

from src import analysis, controllers, cw, plotting, simulation
from src.constants import DEFAULT_RADIUS, MU_EARTH


def main():
    n = cw.mean_motion_circular(DEFAULT_RADIUS, MU_EARTH)
    period = 2.0 * np.pi / n

    # Critically damped PD tuned an order of magnitude faster than the orbital
    # rate so the rendezvous completes well within one period.
    omega_n = 6.0 * n
    max_accel = 0.05  # m/s^2 thrust-acceleration limit
    controller = controllers.PDController.critically_damped(omega_n, max_accel=max_accel)

    x0 = np.array([150.0, -250.0, 0.0, 0.0])  # 150 m radial, -250 m along-track
    dt = period / 4000.0
    res = simulation.simulate_cw(x0, (0.0, period), dt, n, controller=controller, method="rk4")

    pos_err = analysis.position_error_series(res.states)
    vel_err = analysis.velocity_error_series(res.states)
    ctrl_mag = analysis.control_acceleration_series(res.controls)

    table = analysis.rendezvous_metrics_table(res, pos_tol=1.0, vel_tol=0.01)
    csv_path = _bootstrap.result_path("pd_rendezvous_metrics.csv")
    table.to_csv(csv_path, index=False)

    print(f"PD rendezvous: omega_n = {omega_n:.4e} rad/s (= {omega_n/n:.1f} n), "
          f"max_accel = {max_accel} m/s^2\n")
    print(table.T.to_string(header=False))

    traj_fig = plotting.plot_relative_trajectory(
        [("chaser", res.states)],
        _bootstrap.figure_path("pd_relative_trajectory.png"),
        title="PD rendezvous trajectory (LVLH)",
    )
    perr_fig = plotting.plot_time_series(
        res.t, [("position error", pos_err)],
        _bootstrap.figure_path("pd_position_error.png"),
        ylabel="position error [m]", title="Rendezvous position error vs time", logy=True,
    )
    verr_fig = plotting.plot_time_series(
        res.t, [("velocity error", vel_err)],
        _bootstrap.figure_path("pd_velocity_error.png"),
        ylabel="velocity error [m/s]", title="Rendezvous velocity error vs time", logy=True,
    )
    ctrl_fig = plotting.plot_time_series(
        res.t, [("|control accel|", ctrl_mag)],
        _bootstrap.figure_path("pd_control_effort.png"),
        ylabel="control acceleration [m/s^2]", title="Control effort vs time",
    )

    for p in (csv_path, traj_fig, perr_fig, verr_fig, ctrl_fig):
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
