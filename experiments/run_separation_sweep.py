"""Experiment 4 -- separation sweep (central experiment).

Answers at what initial separation does a PD-controlled Clohessy-Wiltshire
guidance law diverge meaningfully from the nonlinear reference model.

For each pure offset direction (radial, along-track, cross-track) and each
initial separation, the same scenario is run in the CW model and in the
nonlinear reference (differenced full two-body / +J2+drag propagations expressed
in LVLH), driven by the identical PD controller. The divergence metric is the
peak position difference between the two closed-loop relative trajectories over
one orbit, both in metres and as a fraction of the initial separation. An
uncontrolled (open-loop, free-drift) variant is run for context.

Run:  python experiments/run_separation_sweep.py
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
    # 2001 samples/orbit (dt ~= 2.78 s): the convergence study shows the reported
    # divergence metric is within 0.01% of the 16001-sample (finest) result here.
    times = np.linspace(0.0, period, 2001)
    omega_n = 6.0 * n
    a_max = 0.05
    pos_tol = 1.0

    directions = {
        "radial": np.array([1.0, 0.0, 0.0]),
        "along-track": np.array([0.0, 1.0, 0.0]),
        "cross-track": np.array([0.0, 0.0, 1.0]),
    }
    separations = np.geomspace(50.0, 50_000.0, 12)  # 50 m .. 50 km
    force_models = {
        "two-body": perturbations.ForceModel.two_body(),
        "j2_drag": perturbations.ForceModel.j2_drag(area=1.0, mass=100.0),
    }
    # closed-loop two-body = central experiment; open-loop two-body = free-drift
    # baseline; closed-loop J2+drag = perturbation effect on the divergence.
    configs = [("closed_loop", "two-body"), ("open_loop", "two-body"), ("closed_loop", "j2_drag")]

    rows = []
    for dname, u in directions.items():
        for sep in separations:
            rel0 = np.concatenate((u * sep, np.zeros(3)))
            for mode, fmname in configs:
                fm = force_models[fmname]
                if True:
                    ctrl = (controller.PDController.critically_damped(omega_n, max_accel=a_max)
                            if mode == "closed_loop" else None)
                    cmp = validation.compare_cw_high_fidelity(target0, rel0, times, fm, n, controller=ctrl)
                    s = cmp["summary"].iloc[0]
                    cwm = validation.rendezvous_metrics(times, cmp["cw_states"], cmp["cw_controls"], pos_tol=pos_tol).iloc[0]
                    hfm = validation.rendezvous_metrics(times, cmp["hf_states"], cmp["hf_controls"], pos_tol=pos_tol).iloc[0]
                    rows.append({
                        "direction": dname,
                        "sep_m": float(sep),
                        "force_model": fmname,
                        "mode": mode,
                        "max_traj_diff_m": float(s["max_pos_error_m"]),
                        "final_traj_diff_m": float(s["final_pos_error_m"]),
                        "rel_max_diff": float(s["max_pos_error_m"] / sep),
                        "cw_final_pos_err_m": float(cwm["final_pos_error_m"]),
                        "hf_final_pos_err_m": float(hfm["final_pos_error_m"]),
                        "cw_conv_time_s": float(cwm["convergence_time_s"]),
                        "hf_conv_time_s": float(hfm["convergence_time_s"]),
                        "cw_delta_v_mps": float(cwm["delta_v_mps"]),
                        "hf_delta_v_mps": float(hfm["delta_v_mps"]),
                        "cw_max_accel_mps2": float(cwm["max_control_accel_mps2"]),
                    })

    df = pd.DataFrame(rows)
    out = os.path.join(DAT, "separation_sweep.csv")
    df.to_csv(out, index=False)

    # Console summary: closed-loop, two-body, divergence vs separation per direction.
    view = df[(df["mode"] == "closed_loop") & (df["force_model"] == "two-body")]
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
        for dname in directions:
            sub = view[view["direction"] == dname][["sep_m", "max_traj_diff_m", "rel_max_diff", "cw_max_accel_mps2", "hf_conv_time_s"]]
            print(f"\n=== closed-loop, two-body, {dname} ===")
            print(sub.to_string(index=False))
    print(f"\nWrote {os.path.relpath(out, ROOT)}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
