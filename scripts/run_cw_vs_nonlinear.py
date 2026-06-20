"""Compare linear CW against high-fidelity non-linear two-body relative motion.

For a set of initial separations the chaser's relative motion is propagated two
ways over one orbit:

* linear CW (closed-form state-transition matrix), and
* "truth": two independent non-linear two-body orbits (target and deputy),
  differenced in the target's rotating LVLH frame (adaptive RK45, tight
  tolerances).

The position discrepancy quantifies where the CW linearisation breaks down as
separation grows. Writes an error-growth CSV and two figures.

Run:  python scripts/run_cw_vs_nonlinear.py
"""

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from src import cw, physics, plotting, simulation
from src.constants import DEFAULT_RADIUS, MU_EARTH


def main():
    n = cw.mean_motion_circular(DEFAULT_RADIUS, MU_EARTH)
    period = 2.0 * np.pi / n
    target0 = physics.circular_orbit_state(DEFAULT_RADIUS, MU_EARTH)
    t_eval = np.linspace(0.0, period, 600)

    # Pure radial initial offsets of increasing magnitude (m).
    separations = [100.0, 1_000.0, 10_000.0, 50_000.0]

    rows = []
    error_series = []
    overlay = None
    for sep in separations:
        rel0 = np.array([sep, 0.0, 0.0, 0.0])
        nl = simulation.propagate_nonlinear_relative(target0, rel0, t_eval, MU_EARTH)
        lin = cw.propagate_cw_analytic(rel0, t_eval, n)
        pos_err = np.linalg.norm(nl.states[:, :2] - lin[:, :2], axis=1)
        rows.append(
            {
                "initial_sep_m": sep,
                "max_cw_error_m": float(pos_err.max()),
                "final_cw_error_m": float(pos_err[-1]),
                "max_error_pct_of_sep": float(100.0 * pos_err.max() / sep),
            }
        )
        error_series.append((f"{sep:.0f} m", pos_err))
        if sep == 1_000.0:
            overlay = [("CW (linear)", lin), ("non-linear truth", nl.states)]

    table = pd.DataFrame(rows)
    csv_path = _bootstrap.result_path("cw_vs_nonlinear_error.csv")
    table.to_csv(csv_path, index=False)

    print(f"Mean motion n = {n:.6e} rad/s, period = {period:.3f} s")
    print("CW vs non-linear two-body, pure radial offsets, 1 orbit:\n")
    print(table.to_string(index=False))

    err_fig = plotting.plot_time_series(
        t_eval, error_series,
        _bootstrap.figure_path("cw_vs_nonlinear_error.png"),
        ylabel="|CW - non-linear| position [m]",
        title="CW linearisation error growth vs separation", logy=True,
    )
    traj_fig = plotting.plot_relative_trajectory(
        overlay,
        _bootstrap.figure_path("cw_vs_nonlinear_trajectory.png"),
        title="CW vs non-linear relative trajectory (1 km radial offset)",
    )

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {err_fig}")
    print(f"Wrote: {traj_fig}")


if __name__ == "__main__":
    main()
