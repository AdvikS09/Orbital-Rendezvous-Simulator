"""Experiment 5 -- time-step convergence study.

Demonstrates that the fixed RK4 step used for the reported experiments does not
materially affect the conclusions. Two checks are run over a range of samples
per orbit (i.e. step sizes):

1. Pure CW integrator: unforced CW propagated by RK4 versus the exact closed-form
   state-transition matrix. The maximum position error is the true integration
   error and should fall as ~dt^4 (classical RK4 order).
2. Full closed-loop pipeline: the reference-model rendezvous divergence metric
   (peak CW-to-reference trajectory difference) and delivered delta-v for a
   representative case (1 km along-track, PD control, two-body reference),
   showing the reported metrics settle as the step is refined.

Run:  python experiments/run_convergence_study.py
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
DAT = os.path.join(ROOT, "outputs", "data")
os.makedirs(DAT, exist_ok=True)

from src import controller, cw, orbit, perturbations, validation


def main():
    radius = orbit.DEFAULT_RADIUS
    mu = orbit.MU_EARTH
    inclination = np.radians(51.6)
    n = cw.cw_mean_motion(radius)
    period = orbit.orbital_period(radius)
    target0 = orbit.circular_orbit_state(radius, mu, inclination)
    omega_n = 6.0 * n
    a_max = 0.05

    sample_counts = [251, 501, 1001, 2001, 4001, 8001, 16001]
    # Representative states.
    unforced0 = np.array([1000.0, 0.0, 500.0, 0.0, 0.0, 0.0])   # radial + cross-track
    rel0_cl = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0])       # 1 km along-track
    fm = perturbations.ForceModel.two_body()

    rows = []
    for N in sample_counts:
        times = np.linspace(0.0, period, N)
        dt = period / (N - 1)

        # 1. Pure CW RK4 vs exact STM (unforced).
        rk4, _ = cw.propagate_cw(unforced0, times, n, None)
        exact = cw.propagate_cw_analytic(unforced0, times, n)
        cw_int_err = float(np.max(np.linalg.norm(rk4[:, :3] - exact[:, :3], axis=1)))

        # 2. Full closed-loop pipeline metric (representative case).
        ctrl = controller.PDController.critically_damped(omega_n, max_accel=a_max)
        cmp = validation.compare_cw_high_fidelity(target0, rel0_cl, times, fm, n, controller=ctrl)
        hfm = validation.rendezvous_metrics(times, cmp["hf_states"], cmp["hf_controls"]).iloc[0]

        rows.append({
            "samples_per_orbit": N,
            "dt_s": dt,
            "cw_rk4_vs_stm_max_err_m": cw_int_err,
            "max_traj_diff_m": float(cmp["summary"].iloc[0]["max_pos_error_m"]),
            "hf_delta_v_mps": float(hfm["delta_v_mps"]),
            "hf_final_pos_err_m": float(hfm["final_pos_error_m"]),
        })

    df = pd.DataFrame(rows)
    # Relative change of the closed-loop divergence metric vs the finest grid.
    finest = df["max_traj_diff_m"].iloc[-1]
    df["max_traj_diff_rel_change"] = (df["max_traj_diff_m"] - finest).abs() / finest
    out = os.path.join(DAT, "convergence_study.csv")
    df.to_csv(out, index=False)
    with pd.option_context("display.float_format", lambda v: f"{v:.6g}"):
        print(df.to_string(index=False))
    print(f"\nWrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
