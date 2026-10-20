import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import reportutil as RU

from cwvalid.constants import DEFRAD, mean_motion, period
from cwvalid.control import CmdSpec, PDLaw
from cwvalid.simulate import SolverCfg, simulate

R = DEFRAD
N = mean_motion(R)
T = period(R)
TIGHT = SolverCfg(rtol=1e-13, atol_rel_pos=1e-12, atol_rel_vel=1e-15, atol_dv=1e-13, max_step=10.0)
DIRS = {"+R": (1, 0, 0), "+V": (0, 1, 0), "+H": (0, 0, 1), "mixed": (0.6, -0.5, 0.62)}
SEPS = np.geomspace(10.0, 5000.0, 10)


def main():
    out = os.path.join(RU.RESULTS, "verification")
    rows = []
    t = np.linspace(0, T, 4001)
    for mode in ("free drift", "unsaturated PD"):
        cmd = CmdSpec("none") if mode == "free drift" else CmdSpec("continuous", law=PDLaw.bandw(6 * N, None))
        for dname, u in DIRS.items():
            u = np.array(u, float) / np.linalg.norm(u)
            for s in SEPS:
                x0 = np.concatenate((u * s, np.zeros(3)))
                sims = {m: simulate(m, x0, cmd, T, n=N, R=R, solver=TIGHT) for m in ("cw", "so", "nerm")}
                ref = sims["nerm"].states(t)[:, :3]
                rows.append({"mode": mode, "direction": dname, "sep_m": s,
                             "cw_err_m": float(np.linalg.norm(sims["cw"].states(t)[:, :3] - ref, axis=1).max()),
                             "so_err_m": float(np.linalg.norm(sims["so"].states(t)[:, :3] - ref, axis=1).max())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "scaling.csv"), index=False)
    sl = []
    for (mode, dname), g in df.groupby(["mode", "direction"]):
        for m, floor in (("cw", 1e-8), ("so", 1e-8)):
            q = g[g[f"{m}_err_m"] > floor]
            if len(q) >= 3:
                p = np.polyfit(np.log(q.sep_m), np.log(q[f"{m}_err_m"]), 1)
                sl.append({"mode": mode, "direction": dname, "model": m, "slope": p[0], "n_points": len(q),
                           "sep_range_m": f"{q.sep_m.min():.0f}-{q.sep_m.max():.0f}"})
    S = pd.DataFrame(sl)
    S.to_csv(os.path.join(out, "scalingslopes.csv"), index=False)
    RU.writetable(S, "scalingslopes", "Log-log slope of the peak one-orbit position error of CW and the "
                  "second-order model against the exact relative model (NERM), versus initial separation "
                  "(points above 1e-8 m). Expected: 2 for CW, 3 for second order.", "tab:scaling",
                  fmt={"slope": "{:.3f}"})
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for a, mode in zip(axs, ("free drift", "unsaturated PD")):
        g = df[df["mode"] == mode]
        for dname, col, mk in (("+R", RU.BLUE, "o"), ("+V", RU.ORANGE, "s"), ("+H", RU.AQUA, "^"), ("mixed", RU.VIOLET, "D")):
            q = g[g.direction == dname]
            a.loglog(q.sep_m, q.cw_err_m, color=col, marker=mk, ls="-", ms=3.5, label=f"CW, {dname}")
            a.loglog(q.sep_m, q.so_err_m.clip(lower=1e-13), color=col, marker=mk, ls="--", mfc="white", ms=3.5, label=f"2nd order, {dname}")
        a.set_title(f"({'a' if mode == 'free drift' else 'b'}) {mode}")
        a.set_xlabel("initial separation [m]")
    axs[0].set_ylabel("peak position error vs exact model [m]")
    axs[1].legend(fontsize=6, ncol=2, loc="lower right")
    fig.tight_layout()
    RU.save(fig, "scalingorder", df)
    print(S.to_string())


if __name__ == "__main__":
    main()
