import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PROJ = os.path.dirname(ROOT)

from src import controller, cw, orbit, perturbations, validation


def main():
    radius = orbit.DEFRAD
    mu = orbit.MUEARTH
    n = cw.cwmeanmotion(radius)
    period = orbit.orbperiod(radius)
    target0 = orbit.circorbitstate(radius, mu, np.radians(51.6))
    times = np.linspace(0.0, period, 2001)
    omegan = 6.0 * n
    amax = 0.05
    postol = 1.0
    fm = perturbations.ForceMod.twobody()
    ctrl = controller.PDCtrl.critdamped(omegan, max_accel=amax)

    vgates = [0.10, 0.05]

    cas = {
        "radial": [500.0, 1000.0, 2000.0, 4000.0, 5500.0, 8000.0, 14000.0],
        "along-track": [1000.0, 4000.0, 7700.0, 14000.0, 27000.0],
        "cross-track": [1000.0, 5000.0, 10000.0, 30000.0, 50000.0],
    }
    unit = {"radial": np.array([1., 0, 0]),
            "along-track": np.array([0, 1., 0]),
            "cross-track": np.array([0, 0, 1.])}

    rows = []
    for dname, seps in cas.items():
        for sep in seps:
            rel0 = np.concatenate((unit[dname] * sep, np.zeros(3)))
            relstates, controls, _, _ = validation.highfidrel(
                target0, rel0, times, fm, controller=ctrl)
            perr = validation.poserrseries(relstates)
            verr = validation.velerrseries(relstates)
            inside = perr < postol
            convt = validation.convtime(times, perr, postol)
            arrpos = np.isfinite(convt) and perr[-1] < postol
            if inside.any():
                first = int(np.argmax(inside))
                ventry = float(verr[first])
                tentry = float(times[first])
                vinside = float(verr[inside].max())
            else:
                ventry = np.nan; tentry = np.nan; vinside = np.nan
            row = {
                "direction": dname, "sep_m": float(sep),
                "closest_approach_m": float(perr.min()),
                "final_pos_err_m": float(perr[-1]),
                "arrived_position": bool(arrpos),
                "t_first_1m_entry_s": tentry,
                "v_at_first_1m_entry_mps": ventry,
                "v_max_inside_1m_mps": vinside,
                "v_final_mps": float(verr[-1]),
            }
            for g in vgates:
                row[f"arrived_pos_and_v_lt_{g:g}"] = bool(
                    arrpos and np.isfinite(verr[-1]) and verr[-1] < g)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PROJ, "terminal.csv"), index=False)

    with pd.option_context("display.float_format", lambda v: f"{v:.4g}",
                           "display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False))
    for g in vgates:
        col = f"arrived_pos_and_v_lt_{g:g}"
        agree = int((df["arrived_position"] == df[col]).sum())
        print(f"\nposition-only vs (position AND v<{g} m/s): agree on {agree}/{len(df)} cases; "
              f"disagreements: {int((df['arrived_position'] != df[col]).sum())}")
    print("\nMax final relative speed among position-arrived cases: %.3e m/s"
          % df.loc[df.arrived_position, "v_final_mps"].max())


if __name__ == "__main__":
    main()
