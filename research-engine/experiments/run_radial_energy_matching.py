"""Energy-matched vs zero-velocity radial offset.

A pure radial offset with zero LVLH relative velocity is not energy-matched: to
first order it places the chaser on an orbit of slightly different period, so it
drifts along-track. The bounded-motion (drift-free) condition for a radial offset
x0 in the Clohessy-Wiltshire model is ydot0 = -2 n x0, which makes the two orbits
share a semi-major axis and therefore a period and specific energy. This
experiment isolates how much of the poor radial behaviour reported in the main
separation sweep is due to that energy/period mismatch, by comparing the two
initial conditions under an otherwise identical closed-loop PD rendezvous.

Run:  python experiments/run_radial_energy_matching.py
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PROJ = os.path.dirname(ROOT)  # "Orbital Mechanics Research"
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
    times = np.linspace(0.0, period, 2001)
    omega_n = 6.0 * n
    a_max = 0.05
    pos_tol = 1.0
    fm = perturbations.ForceModel.two_body()

    # Compact separation set spanning close range, the radial 1 m onset (~5.4 km),
    # and the regime where the zero-velocity reference stops converging (~8-14 km).
    separations = np.array([500.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0, 12000.0])

    def ic(kind, x0):
        if kind == "zero_velocity":
            return np.array([x0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # energy-matched / bounded: ydot0 = -2 n x0 removes the CW secular drift
        return np.array([x0, 0.0, 0.0, 0.0, -2.0 * n * x0, 0.0])

    rows = []
    for x0 in separations:
        for kind in ("zero_velocity", "energy_matched"):
            rel0 = ic(kind, x0)
            for mode in ("closed_loop", "open_loop"):
                ctrl = (controller.PDController.critically_damped(omega_n, max_accel=a_max)
                        if mode == "closed_loop" else None)
                cmp = validation.compare_cw_high_fidelity(target0, rel0, times, fm, n, controller=ctrl)
                s = cmp["summary"].iloc[0]
                cwm = validation.rendezvous_metrics(times, cmp["cw_states"], cmp["cw_controls"], pos_tol=pos_tol).iloc[0]
                hfm = validation.rendezvous_metrics(times, cmp["hf_states"], cmp["hf_controls"], pos_tol=pos_tol).iloc[0]
                rows.append({
                    "ic": kind,
                    "sep_m": float(x0),
                    "ydot0_mps": float(rel0[4]),
                    "mode": mode,
                    "max_traj_diff_m": float(s["max_pos_error_m"]),
                    "final_traj_diff_m": float(s["final_pos_error_m"]),
                    "rel_max_diff": float(s["max_pos_error_m"] / x0),
                    "cw_final_pos_err_m": float(cwm["final_pos_error_m"]),
                    "hf_final_pos_err_m": float(hfm["final_pos_error_m"]),
                    "cw_conv_time_s": float(cwm["convergence_time_s"]),
                    "hf_conv_time_s": float(hfm["convergence_time_s"]),
                    "cw_delta_v_mps": float(cwm["delta_v_mps"]),
                    "hf_delta_v_mps": float(hfm["delta_v_mps"]),
                    "cw_max_accel_mps2": float(cwm["max_control_accel_mps2"]),
                    "hf_max_accel_mps2": float(hfm["max_control_accel_mps2"]),
                    "hf_saturated": bool(hfm["max_control_accel_mps2"] >= 0.999 * a_max),
                })

    df = pd.DataFrame(rows)
    for path in (os.path.join(PROJ, "RADIAL_ENERGY_MATCHING.csv"),
                 os.path.join(DAT, "radial_energy_matching.csv")):
        df.to_csv(path, index=False)

    # Console summary: closed-loop comparison of the two ICs.
    cl = df[df["mode"] == "closed_loop"]
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
        for kind in ("zero_velocity", "energy_matched"):
            sub = cl[cl["ic"] == kind][["sep_m", "max_traj_diff_m", "rel_max_diff",
                                        "hf_conv_time_s", "hf_final_pos_err_m",
                                        "hf_delta_v_mps", "hf_max_accel_mps2", "hf_saturated"]]
            print(f"\n=== closed-loop, two-body, radial, {kind} ===")
            print(sub.to_string(index=False))

    # Open-loop free-drift peak divergence (the mechanism): zero-vel drifts, matched stays bounded.
    ol = df[df["mode"] == "open_loop"]
    print("\n=== open-loop free-drift peak CW-to-reference difference [m] ===")
    piv = ol.pivot(index="sep_m", columns="ic", values="max_traj_diff_m")
    print(piv.to_string())

    print(f"\nWrote RADIAL_ENERGY_MATCHING.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
