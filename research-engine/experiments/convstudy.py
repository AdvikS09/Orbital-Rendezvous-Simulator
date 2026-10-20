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
    omegan = 6.0 * n
    amax = 0.05

    sampcounts = [251, 501, 1001, 2001, 4001, 8001, 16001]
    u = np.array([1000.0, 0.0, 500.0, 0.0, 0.0, 0.0])
    rel0cl = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0])
    fm = perturbations.ForceMod.twobody()

    rows = []
    for N in sampcounts:
        times = np.linspace(0.0, period, N)
        dt = period / (N - 1)

        rk4, _ = cw.propcw(u, times, n, None)
        exact = cw.propcwanal(u, times, n)
        cwinterr = float(np.max(np.linalg.norm(rk4[:, :3] - exact[:, :3], axis=1)))

        ctrl = controller.PDCtrl.critdamped(omegan, max_accel=amax)
        cmp = validation.cmpfid(target0, rel0cl, times, fm, n, controller=ctrl)
        hfm = validation.rdvmetrics(times, cmp["hf_states"], cmp["hf_controls"]).iloc[0]

        rows.append({
            "samples_per_orbit": N,
            "dt_s": dt,
            "cw_rk4_vs_stm_max_err_m": cwinterr,
            "max_traj_diff_m": float(cmp["summary"].iloc[0]["max_pos_error_m"]),
            "hf_delta_v_mps": float(hfm["delta_v_mps"]),
            "hf_final_pos_err_m": float(hfm["final_pos_error_m"]),
        })

    df = pd.DataFrame(rows)
    finest = df["max_traj_diff_m"].iloc[-1]
    df["max_traj_diff_rel_change"] = (df["max_traj_diff_m"] - finest).abs() / finest
    out = os.path.join(DAT, "convergence.csv")
    df.to_csv(out, index=False)
    with pd.option_context("display.float_format", lambda v: f"{v:.6g}"):
        print(df.to_string(index=False))
    print(f"\nWrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
