import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import reportutil as RU

from cwvalid.runner import loadres

MG = os.path.join(RU.RESULTS, "maingrid")


def limit(g, col):
    g = g.sort_values("sep_m").drop_duplicates("sep_m")
    v = g[col].astype(bool).to_numpy()
    s = g.sep_m.to_numpy()
    idx = np.where(v[:-1] & ~v[1:])[0]
    if idx.size == 0:
        return np.nan, np.nan, 0
    i = idx[0]
    return s[i], s[i + 1], int(idx.size)


def main():
    RU.setup()
    d = pd.concat([loadres(os.path.join(MG, s)) for s in ("grid", "refine", "refinebands")], ignore_index=True)
    d = d[d.meta_velocity == "zero"]
    rows = []
    for (ax, w, a), g in d.groupby(["meta_axis", "wn_ratio", "a_max"]):
        c0, c1, nc = limit(g, "cw_terminal_arrived")
        r0, r1, nr = limit(g, "eci_terminal_arrived")
        if not np.isfinite(r0) and not np.isfinite(c0):
            continue
        cm, rm = 0.5 * (c0 + c1), 0.5 * (r0 + r1)
        rows.append({"axis": ax, "direction": RU.AXISLABEL[ax], "wn/n": w, "a_max": a,
                     "CW capture limit [m]": f"{c0:.1f}-{c1:.1f}", "reference capture limit [m]": f"{r0:.1f}-{r1:.1f}",
                     "offset [%]": 100 * (cm - rm) / rm, "offset uncertainty [%]": 100 * 0.5 * ((c1 - c0) / cm + (r1 - r0) / rm),
                     "arrival changes CW / ref": f"{nc} / {nr}"})
    t = pd.DataFrame(rows)
    RU.writetable(t.drop(columns=["axis"]), "capturelimits",
                  "Capture limits (largest initial separation with a primary terminal arrival, zero initial velocity, one "
                  "orbit) of the CW loop and the inertial nonlinear reference loop, and their relative offset. Positive "
                  "offset: CW predicts arrival over a band where the reference does not.", "tab:capture",
                  fmt={"offset [%]": "{:+.3f}", "offset uncertainty [%]": "{:.3f}"})
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    xt, xl = [], []
    data = []
    k = 0
    for base in ("R", "V", "H"):
        for sgn in ("+", "-"):
            axn = sgn + base
            q = t[t.axis == axn].sort_values(["a_max", "wn/n"])
            c, ls, mk = RU.axisstyle(axn)
            xs = np.arange(k, k + len(q))
            ax.errorbar(xs, q["offset [%]"], yerr=q["offset uncertainty [%]"], fmt=mk, color=c, ms=4, lw=0.8, capsize=1.5,
                        mfc=c if sgn == "+" else "white", label=RU.AXISLABEL[axn])
            for x_, (_, r) in zip(xs, q.iterrows()):
                data.append({"x_position": int(x_), "axis": axn, "wn_ratio": r["wn/n"], "a_max": r["a_max"],
                             "offset_pct": r["offset [%]"], "unc_pct": r["offset uncertainty [%]"],
                             "cw_capture_limit_m": r["CW capture limit [m]"],
                             "reference_capture_limit_m": r["reference capture limit [m]"]})
            if len(q):
                xt.append(xs.mean())
                xl.append(RU.AXISLABEL[axn].replace("-track", ""))
            k += len(q) + 1
    ax.axhline(0, color=RU.MUTED, lw=0.8)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=7)
    ax.set_ylabel("CW minus reference capture limit [%]")
    ax.set_ylim(-0.5, 0.5)
    ax.text(0.99, 0.97, "above 0: CW arrives where the reference does not (falsely favourable CW)", transform=ax.transAxes,
            fontsize=6.5, color=RU.INK2, va="top", ha="right")
    ax.text(0.01, 0.03, "below 0: the reference arrives where CW does not", transform=ax.transAxes, fontsize=6.5,
            color=RU.INK2, va="bottom")
    ax.set_xlabel("initial offset direction (within each group: a_max 0.01, 0.05, 0.10; each with wn/n 4, 6, 8)", fontsize=7)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    RU.save(fig, "captureoffset", pd.DataFrame(data))
    print(t.to_string())


if __name__ == "__main__":
    main()
