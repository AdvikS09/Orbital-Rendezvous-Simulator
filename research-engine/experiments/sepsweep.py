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
    radius = orbit.DEFRAD
    mu = orbit.MUEARTH
    inc = np.radians(51.6)
    n = cw.cwmeanmotion(radius)
    period = orbit.orbperiod(radius)
    target0 = orbit.circorbitstate(radius, mu, inc)
    times = np.linspace(0.0, period, 2001)
    omegan = 6.0 * n
    amax = 0.05
    postol = 1.0

    direct = {
        "radial": np.array([1.0, 0.0, 0.0]),
        "along-track": np.array([0.0, 1.0, 0.0]),
        "cross-track": np.array([0.0, 0.0, 1.0]),
    }
    seps = np.geomspace(50.0, 50_000.0, 12)
    forcemod = {
        "two-body": perturbations.ForceMod.twobody(),
        "j2_drag": perturbations.ForceMod.j2_drag(area=1.0, mass=100.0),
    }
    configs = [("closed_loop", "two-body"), ("open_loop", "two-body"), ("closed_loop", "j2_drag")]

    rows = []
    for dname, u in direct.items():
        for sep in seps:
            rel0 = np.concatenate((u * sep, np.zeros(3)))
            for mode, fmname in configs:
                fm = forcemod[fmname]
                if True:
                    ctrl = (controller.PDCtrl.critdamped(omegan, max_accel=amax)
                            if mode == "closed_loop" else None)
                    cmp = validation.cmpfid(target0, rel0, times, fm, n, controller=ctrl)
                    s = cmp["summary"].iloc[0]
                    cwm = validation.rdvmetrics(times, cmp["cw_states"], cmp["cw_controls"], pos_tol=postol).iloc[0]
                    hfm = validation.rdvmetrics(times, cmp["hf_states"], cmp["hf_controls"], pos_tol=postol).iloc[0]
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
    out = os.path.join(DAT, "sepsweep.csv")
    df.to_csv(out, index=False)

    view = df[(df["mode"] == "closed_loop") & (df["force_model"] == "two-body")]
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
        for dname in direct:
            sub = view[view["direction"] == dname][["sep_m", "max_traj_diff_m", "rel_max_diff", "cw_max_accel_mps2", "hf_conv_time_s"]]
            print(f"\n=== closed-loop, two-body, {dname} ===")
            print(sub.to_string(index=False))
    print(f"\nWrote {os.path.relpath(out, ROOT)}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
