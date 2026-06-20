"""Experiment 2 -- PD gain-tuning study.

Sweeps a grid of proportional (Kp) and derivative (Kd) gains, running the same
CW rendezvous for each combination, and reports final position error, final
velocity error, convergence time, maximum control acceleration and approximate
delta-v. Saves the sweep table and a heatmap comparison of the metrics over the
Kp x Kd grid.

Gains are applied without thrust saturation so that each combination's control
effort reflects the gains directly.

Run:  python experiments/run_gain_sweep.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "outputs", "figures")
DAT = os.path.join(ROOT, "outputs", "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DAT, exist_ok=True)

from src import cw, orbit, plotting, validation


def main():
    radius = orbit.DEFAULT_RADIUS
    n = cw.cw_mean_motion(radius)
    period = orbit.orbital_period(radius)

    rel0 = np.array([200.0, -300.0, 80.0, 0.0, 0.0, 0.0])  # same scenario as exp. 1
    times = np.linspace(0.0, period, 4001)

    # Gain grids bracketing the critically damped point (omega_n = 6 n):
    #   kp ~ omega_n^2 ~ 4.6e-5,  kd ~ 2 omega_n ~ 1.36e-2.
    kp_values = [1.0e-5, 4.6e-5, 1.0e-4, 2.0e-4]
    kd_values = [5.0e-3, 1.0e-2, 1.36e-2, 3.0e-2]

    df = validation.gain_sweep(
        rel0, times, n, kp_values, kd_values, max_accel=None, pos_tol=1.0
    )
    csv_path = os.path.join(DAT, "gain_sweep.csv")
    df.to_csv(csv_path, index=False)

    print(f"PD gain sweep ({len(kp_values)} Kp x {len(kd_values)} Kd = {len(df)} runs), "
          f"n = {n:.4e} rad/s\n")
    print(df.round(6).to_string(index=False))

    # Best by convergence time among runs that actually converged.
    converged = df.dropna(subset=["convergence_time_s"])
    if not converged.empty:
        best = converged.loc[converged["convergence_time_s"].idxmin()]
        print(f"\nFastest converging: Kp={best.kp:.2e}, Kd={best.kd:.2e}, "
              f"t_conv={best.convergence_time_s:.1f} s, delta_v={best.delta_v_mps:.3f} m/s")

    fig = plotting.plot_gain_comparison(
        df, os.path.join(FIG, "gain_sweep_comparison.png"),
        metrics=("final_pos_error_m", "convergence_time_s", "delta_v_mps"),
        title="PD gain sweep: metrics over Kp x Kd grid",
    )
    print(f"\nWrote: {os.path.relpath(csv_path, ROOT)}")
    print(f"Wrote: {os.path.relpath(fig, ROOT)}")


if __name__ == "__main__":
    main()
