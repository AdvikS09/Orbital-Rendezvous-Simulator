"""Demonstrate and validate Clohessy-Wiltshire relative motion (uncontrolled).

Propagates the natural (control-free) CW relative motion of a chaser about the
target for one orbital period using RK4, and overlays the closed-form CW
state-transition-matrix solution. The maximum numeric-vs-analytic discrepancy is
written to a CSV; the classic "football"/drift relative trajectory is plotted.

This isolates the *dynamics* model (no controller); see run_pd_rendezvous.py for
the closed-loop case.

Run:  python scripts/run_cw_rendezvous.py
"""

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from src import analysis, cw, plotting, simulation
from src.constants import DEFAULT_RADIUS, MU_EARTH


def main():
    n = cw.mean_motion_circular(DEFAULT_RADIUS, MU_EARTH)
    period = 2.0 * np.pi / n

    # A few characteristic initial relative states (radial, along-track offsets
    # and small relative velocities), all in metres / metres-per-second.
    cases = {
        "radial offset 100 m": np.array([100.0, 0.0, 0.0, 0.0]),
        "along-track offset 200 m": np.array([0.0, 200.0, 0.0, 0.0]),
        "drift 50 m + 0.1 m/s": np.array([50.0, 0.0, 0.0, 0.1]),
    }

    dt = period / 4000.0
    rows = []
    trajectories = []
    for label, x0 in cases.items():
        res = simulation.simulate_cw(x0, (0.0, period), dt, n, controller=None, method="rk4")
        analytic = cw.propagate_cw_analytic(x0, res.t, n)
        pos_err = np.linalg.norm(res.states[:, :2] - analytic[:, :2], axis=1)
        rows.append(
            {
                "case": label,
                "x0_radial_m": x0[0],
                "y0_alongtrack_m": x0[1],
                "max_numeric_vs_stm_pos_err_m": float(pos_err.max()),
                "final_pos_error_m": float(np.linalg.norm(res.states[-1, :2])),
            }
        )
        trajectories.append((label, res.states))

    table = pd.DataFrame(rows)
    csv_path = _bootstrap.result_path("cw_stm_validation.csv")
    table.to_csv(csv_path, index=False)

    print(f"Mean motion n = {n:.6e} rad/s, period = {period:.3f} s\n")
    print(table.to_string(index=False))

    fig = plotting.plot_relative_trajectory(
        trajectories,
        _bootstrap.figure_path("cw_relative_trajectory.png"),
        title="Natural CW relative motion (1 orbit)",
    )
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {fig}")


if __name__ == "__main__":
    main()
