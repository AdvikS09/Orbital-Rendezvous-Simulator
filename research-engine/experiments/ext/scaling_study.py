"""Small-separation limit and approximation order of CW and second-order models.

For free drift (no control) and for unsaturated PD feedback (omega_n = 6n,
no cap), the peak position difference over one orbit between each
approximate model (CW, second-order) and the exact relative model (NERM) is
computed for separations 10 m - 5 km in four directions. Expected scaling:
CW error ~ rho0^2 (neglected quadratic gravity terms), second-order error ~
rho0^3. Slopes are fitted in log-log on points well above the numerical floor.

Output: results/verification/scaling_study.csv, scaling_slopes.csv, figure
fig_scaling_order. Usage: python experiments/ext/scaling_study.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import report_utils as RU  # noqa: E402

from cwvalid.constants import DEFAULT_RADIUS, mean_motion, period  # noqa: E402
from cwvalid.control import CommandSpec, PDLaw  # noqa: E402
from cwvalid.simulate import SolverConfig, simulate  # noqa: E402

R = DEFAULT_RADIUS
N = mean_motion(R)
T = period(R)
TIGHT = SolverConfig(rtol=1e-13, atol_rel_pos=1e-12, atol_rel_vel=1e-15, atol_dv=1e-13, max_step=10.0)
DIRS = {"+R": (1, 0, 0), "+V": (0, 1, 0), "+H": (0, 0, 1), "mixed": (0.6, -0.5, 0.62)}
SEPS = np.geomspace(10.0, 5000.0, 10)


def main():
    out = os.path.join(RU.RESULTS, "verification")
    rows = []
    t = np.linspace(0, T, 4001)
    for mode in ("free drift", "unsaturated PD"):
        cmd = CommandSpec("none") if mode == "free drift" else CommandSpec("continuous", law=PDLaw.from_bandwidth(6 * N, None))
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
    df.to_csv(os.path.join(out, "scaling_study.csv"), index=False)
    sl = []
    for (mode, dname), g in df.groupby(["mode", "direction"]):
        for m, floor in (("cw", 1e-8), ("so", 1e-8)):
            q = g[g[f"{m}_err_m"] > floor]
            if len(q) >= 3:
                p = np.polyfit(np.log(q.sep_m), np.log(q[f"{m}_err_m"]), 1)
                sl.append({"mode": mode, "direction": dname, "model": m, "slope": p[0], "n_points": len(q),
                           "sep_range_m": f"{q.sep_m.min():.0f}-{q.sep_m.max():.0f}"})
    S = pd.DataFrame(sl)
    S.to_csv(os.path.join(out, "scaling_slopes.csv"), index=False)
    RU.write_table(S, "tab_scaling_slopes", "Log-log slope of the peak one-orbit position error of CW and the "
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
    RU.save(fig, "fig_scaling_order", df)
    print(S.to_string())


if __name__ == "__main__":
    main()
