import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PROJ = os.path.dirname(ROOT)
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

    separations = np.array([500.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0, 12000.0])

    def ic(kind, x0):
        if kind == "zero_velocity":
            return np.array([x0, 0.0, 0.0, 0.0, 0.0, 0.0])
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

    cl = df[df["mode"] == "closed_loop"]
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
        for kind in ("zero_velocity", "energy_matched"):
            sub = cl[cl["ic"] == kind][["sep_m", "max_traj_diff_m", "rel_max_diff",
                                        "hf_conv_time_s", "hf_final_pos_err_m",
                                        "hf_delta_v_mps", "hf_max_accel_mps2", "hf_saturated"]]
            print(f"\n=== closed-loop, two-body, radial, {kind} ===")
            print(sub.to_string(index=False))

    ol = df[df["mode"] == "open_loop"]
    print("\n=== open-loop free-drift peak CW-to-reference difference [m] ===")
    piv = ol.pivot(index="sep_m", columns="ic", values="max_traj_diff_m")
    print(piv.to_string())

    print(f"\nWrote RADIAL_ENERGY_MATCHING.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
