"""PD rendezvous with keep-out-zone avoidance (artificial potential field).

The chaser must reach the target (LVLH origin) while avoiding a circular
keep-out zone placed near its direct approach path. The controller combines PD
attraction to the target with an artificial-potential-field repulsion from the
zone. Compares the avoidance run against a plain-PD run (which cuts the corner)
and reports the minimum clearance achieved.

Run:  python scripts/run_obstacle_avoidance_demo.py
"""

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from src import analysis, controllers, cw, plotting, simulation
from src.constants import DEFAULT_RADIUS, MU_EARTH


def min_clearance(states, obstacle):
    return float(min(obstacle.clearance(s[:2]) for s in states))


def main():
    n = cw.mean_motion_circular(DEFAULT_RADIUS, MU_EARTH)
    period = 2.0 * np.pi / n

    omega_n = 6.0 * n
    max_accel = 0.05
    x0 = np.array([0.0, 300.0, 0.0, 0.0])  # 300 m along-track from the target

    # Keep-out zone placed on the chaser's natural PD approach path (which bows
    # to negative radial), so plain PD would fly straight through it.
    obstacle = controllers.Obstacle(center=[-15.0, 140.0], radius=30.0, influence=90.0)

    dt = period / 4000.0

    # Baseline: plain PD (no avoidance).
    pd_plain = controllers.PDController.critically_damped(omega_n, max_accel=max_accel)
    res_pd = simulation.simulate_cw(x0, (0.0, period), dt, n, controller=pd_plain, method="rk4")

    # With APF obstacle avoidance.
    pd2 = controllers.PDController.critically_damped(omega_n, max_accel=max_accel)
    avoider = controllers.ObstacleAvoidanceController(
        pd2, [obstacle], eta=50.0, max_accel=max_accel
    )
    res_av = simulation.simulate_cw(x0, (0.0, period), dt, n, controller=avoider, method="rk4")

    rows = [
        {
            "run": "plain PD",
            "min_clearance_m": min_clearance(res_pd.states, obstacle),
            "final_pos_error_m": float(np.linalg.norm(res_pd.states[-1, :2])),
            "delta_v_mps": analysis.cumulative_delta_v(res_pd.t, res_pd.controls),
        },
        {
            "run": "PD + APF avoidance",
            "min_clearance_m": min_clearance(res_av.states, obstacle),
            "final_pos_error_m": float(np.linalg.norm(res_av.states[-1, :2])),
            "delta_v_mps": analysis.cumulative_delta_v(res_av.t, res_av.controls),
        },
    ]
    table = pd.DataFrame(rows)
    csv_path = _bootstrap.result_path("obstacle_avoidance_metrics.csv")
    table.to_csv(csv_path, index=False)

    print(f"Keep-out zone: centre {obstacle.center.tolist()} m, radius {obstacle.radius} m, "
          f"influence {obstacle.influence} m")
    print("(positive clearance = outside the zone)\n")
    print(table.to_string(index=False))

    fig = plotting.plot_relative_trajectory(
        [("plain PD", res_pd.states), ("PD + APF avoidance", res_av.states)],
        _bootstrap.figure_path("obstacle_avoidance_trajectory.png"),
        obstacles=[obstacle],
        title="Rendezvous with keep-out-zone avoidance (LVLH)",
    )

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {fig}")


if __name__ == "__main__":
    main()
