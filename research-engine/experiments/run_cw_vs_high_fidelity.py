"""Experiment 3 -- CW vs high-fidelity validation.

Three parts:

1. Two-body orbit validation: propagate a circular LEO with RK4 and confirm
   energy / angular-momentum conservation and the orbital period.
2. Same controlled rendezvous scenario run in CW and in a high-fidelity
   (two-body + J2 + drag) model; compare position and velocity error over time.
3. Perturbation-toggle study: uncontrolled free-drift divergence of CW from the
   high-fidelity model for each force-model toggle (two-body, J2, drag, J2+drag).

Run:  python experiments/run_cw_vs_high_fidelity.py
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

from src import controller, cw, orbit, perturbations, plotting, validation


def main():
    radius = orbit.DEFAULT_RADIUS
    inclination = np.radians(51.6)              # ISS-like, exercises 3D + J2
    mu = orbit.MU_EARTH
    n = cw.cw_mean_motion(radius)
    period = orbit.orbital_period(radius)
    target0 = orbit.circular_orbit_state(radius, mu, inclination)
    times = np.linspace(0.0, period, 4001)

    # --- Part 1: two-body energy / period validation (2 orbits) ---
    t2 = np.linspace(0.0, 2.0 * period, 8001)
    tb_states = orbit.propagate(perturbations.ForceModel.two_body().derivatives, target0, t2)
    verify = validation.verify_orbit(t2, tb_states, mu, analytic_period=period)
    verify_path = os.path.join(DAT, "orbit_validation.csv")
    verify.to_csv(verify_path, index=False)
    print("Two-body orbit validation (RK4, 2 orbits):")
    print(verify.to_string(index=False))

    # --- Part 2: controlled rendezvous, CW vs high-fidelity (J2 + drag) ---
    rel0 = np.array([200.0, -300.0, 80.0, 0.0, 0.0, 0.0])
    pd_ctrl = controller.PDController.critically_damped(6.0 * n, max_accel=0.05)
    fm_hf = perturbations.ForceModel.j2_drag(area=1.0, mass=100.0)
    cmp = validation.compare_cw_high_fidelity(target0, rel0, times, fm_hf, n, controller=pd_ctrl)

    ts = pd.DataFrame({
        "t_s": times,
        "pos_error_m": cmp["pos_error"],
        "vel_error_mps": cmp["vel_error"],
    })
    ts_path = os.path.join(DAT, "cw_vs_high_fidelity_timeseries.csv")
    ts.to_csv(ts_path, index=False)
    print("\nControlled rendezvous: CW vs high-fidelity (two-body + J2 + drag):")
    print(cmp["summary"].to_string(index=False))

    err_fig = plotting.plot_error_growth(
        times, cmp["pos_error"], cmp["vel_error"],
        os.path.join(FIG, "cw_vs_high_fidelity_error.png"),
        title="CW vs high-fidelity error over time (controlled rendezvous)",
    )

    # --- Part 3: perturbation-toggle free-drift study ---
    rel0_drift = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0])  # 1 km along-track
    models = [
        perturbations.ForceModel.two_body(),
        perturbations.ForceModel.j2_only(),
        perturbations.ForceModel.drag_only(area=1.0, mass=100.0),
        perturbations.ForceModel.j2_drag(area=1.0, mass=100.0),
    ]
    series = []
    summaries = []
    for fm in models:
        c = validation.compare_cw_high_fidelity(target0, rel0_drift, times, fm, n, controller=None)
        series.append((fm.label, c["pos_error"]))
        summaries.append(c["summary"])
    toggles = pd.concat(summaries, ignore_index=True)
    toggles_path = os.path.join(DAT, "cw_vs_high_fidelity_toggles.csv")
    toggles.to_csv(toggles_path, index=False)
    print("\nFree-drift CW vs high-fidelity divergence by force-model toggle "
          "(1 km along-track offset, 1 orbit):")
    print(toggles.to_string(index=False))

    toggles_fig = plotting.plot_error_curves(
        times, series, os.path.join(FIG, "cw_vs_high_fidelity_toggles.png"),
        ylabel="CW vs high-fidelity position error [m]",
        title="Model divergence by perturbation toggle (free drift)", logy=True,
    )

    print()
    for p in (verify_path, ts_path, toggles_path, err_fig, toggles_fig):
        print(f"Wrote: {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
