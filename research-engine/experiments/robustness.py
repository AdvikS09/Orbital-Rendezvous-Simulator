import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PROJ = os.path.dirname(ROOT)

from src import controller, cw, orbit, perturbations, validation


def logcross(seps, diffs, level=1.0):
    seps = np.asarray(seps, float)
    diffs = np.asarray(diffs, float)
    order = np.argsort(seps)
    seps, diffs = seps[order], diffs[order]
    if np.all(diffs >= level):
        return ("below_range", float(seps[0]))
    if np.all(diffs < level):
        return ("above_range", float(seps[-1]))
    for i in range(1, len(seps)):
        if diffs[i - 1] < level <= diffs[i]:
            lx0, lx1 = np.log(seps[i - 1]), np.log(seps[i])
            ly0, ly1 = np.log(diffs[i - 1]), np.log(diffs[i])
            lx = lx0 + (np.log(level) - ly0) * (lx1 - lx0) / (ly1 - ly0)
            return ("crossing", float(np.exp(lx)))
    return ("above_range", float(seps[-1]))


def main():
    radius = orbit.DEFRAD
    mu = orbit.MUEARTH
    n = cw.cwmeanmotion(radius)
    period = orbit.orbperiod(radius)
    target0 = orbit.circorbitstate(radius, mu, np.radians(51.6))
    amax = 0.05
    postol = 1.0
    fm = perturbations.ForceMod.twobody()

    direct = {
        "radial": np.array([1.0, 0.0, 0.0]),
        "along-track": np.array([0.0, 1.0, 0.0]),
        "cross-track": np.array([0.0, 0.0, 1.0]),
    }
    bandw = [4.0, 6.0, 8.0]
    horizons = [0.5, 1.0, 2.0]
    seps = np.geomspace(50.0, 50_000.0, 16)

    gridrows = []
    for wratio in bandw:
        omegan = wratio * n
        ctrl = controller.PDCtrl.critdamped(omegan, max_accel=amax)
        for h in horizons:
            N = int(round(2000 * h)) + 1
            times = np.linspace(0.0, h * period, N)
            for dname, u in direct.items():
                for sep in seps:
                    rel0 = np.concatenate((u * sep, np.zeros(3)))
                    cmp = validation.cmpfid(
                        target0, rel0, times, fm, n, controller=ctrl)
                    s = cmp["summary"].iloc[0]
                    hfm = validation.rdvmetrics(
                        times, cmp["hf_states"], cmp["hf_controls"], pos_tol=postol).iloc[0]
                    gridrows.append({
                        "omega_n_over_n": wratio,
                        "horizon_orbits": h,
                        "direction": dname,
                        "sep_m": float(sep),
                        "max_traj_diff_m": float(s["max_pos_error_m"]),
                        "final_traj_diff_m": float(s["final_pos_error_m"]),
                        "rel_max_diff": float(s["max_pos_error_m"] / sep),
                        "hf_conv_time_s": float(hfm["convergence_time_s"]),
                        "hf_final_pos_err_m": float(hfm["final_pos_error_m"]),
                        "hf_max_accel_mps2": float(hfm["max_control_accel_mps2"]),
                        "hf_saturated": bool(hfm["max_control_accel_mps2"] >= 0.999 * amax),
                    })

    grid = pd.DataFrame(gridrows)
    grid.to_csv(os.path.join(PROJ, "robustness.csv"), index=False)

    crossrows = []
    for wratio in bandw:
        for h in horizons:
            for dname in direct:
                sub = grid[(grid.omega_n_over_n == wratio) &
                           (grid.horizon_orbits == h) &
                           (grid.direction == dname)]
                kind, val = logcross(sub.sep_m.values, sub.max_traj_diff_m.values)
                crossrows.append({
                    "omega_n_over_n": wratio, "horizon_orbits": h, "direction": dname,
                    "crossing_kind": kind,
                    "one_metre_crossing_km": (val / 1000.0) if kind == "crossing" else np.nan,
                    "note": {"crossing": "", "above_range": ">50 km (no crossing)",
                             "below_range": "<50 m"}[kind],
                })
    cross = pd.DataFrame(crossrows)
    cross.to_csv(os.path.join(PROJ, "robustnesscrossings.csv"), index=False)

    pd.set_option("display.width", 160)
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
        print("\n=== one-metre CW-to-reference crossing [km], by bandwidth x horizon ===")
        for dname in direct:
            piv = cross[cross.direction == dname].pivot(
                index="omega_n_over_n", columns="horizon_orbits",
                values="one_metre_crossing_km")
            print(f"\n{dname}:")
            print(piv.to_string())
        print("\n=== directional ranking per cell (should be radial < along < cross) ===")
        ok = True
        for wratio in bandw:
            for h in horizons:
                c = {d: cross[(cross.omega_n_over_n == wratio) & (cross.horizon_orbits == h) &
                              (cross.direction == d)].iloc[0] for d in direct}
                def key(d):
                    r = c[d]
                    return (np.inf if r.crossing_kind == "above_range"
                            else -np.inf if r.crossing_kind == "below_range"
                            else r.one_metre_crossing_km)
                r, a, cx = key("radial"), key("along-track"), key("cross-track")
                ranked = r <= a <= cx
                ok = ok and ranked
                print(f"  wn/n={wratio}, h={h}: radial={r:.3g} along={a:.3g} cross={cx:.3g}  "
                      f"{'RANK OK' if ranked else 'RANK CHANGED'}")
        print("\nDirectional ranking preserved in every cell:" , ok)
    print(f"\nWrote robustness.csv ({len(grid)} rows) and crossings ({len(cross)} rows)")


if __name__ == "__main__":
    main()
