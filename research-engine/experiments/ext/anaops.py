import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reportutil as RU

OP = os.path.join(RU.RESULTS, "operational")
STAGELABEL = {"model": "CW vs inertial (two-body), continuous", "formulation": "NERM vs inertial (check)",
              "pert_J2": "two-body vs J2 (inertial)", "pert_drag": "J2 vs J2 + drag (equal B)",
              "diff_drag_hi": "J2+drag: chaser B x2 (exploratory)", "diff_drag_lo": "J2+drag: chaser B x0.5 (exploratory)",
              "sampling_1s": "continuous vs 1 s ZOH (inertial)", "sampling_10s": "continuous vs 10 s ZOH (inertial)",
              "cw_sampling_10s": "continuous vs 10 s ZOH (CW)", "combined": "CW continuous vs J2+drag 10 s ZOH",
              "frame_fix": "J2: exact vs baseline frame rate", "model_s10": "CW vs inertial, both 10 s ZOH"}


def main():
    RU.setup()
    st = pd.read_csv(os.path.join(OP, "stages.csv"))
    st["category"] = np.where(st.meta_label.fillna("").str.contains("representative"), "representative", "boundary")
    rows = []
    for (stage, cat), g in st.groupby(["stage", "category"], sort=False):
        flips = g.outcome.isin(["cw_only", "nl_only"])
        rows.append({"stage": STAGELABEL.get(stage, stage), "key": stage, "cases": cat, "N": len(g),
                     "median peak diff [m]": g.max_pos_disc_m.median(), "90th pct [m]": g.max_pos_disc_m.quantile(0.9),
                     "max [m]": g.max_pos_disc_m.max(),
                     "outcome changes": int(flips.sum()),
                     "a arrives, b not": int((g.outcome == "cw_only").sum()),
                     "b arrives, a not": int((g.outcome == "nl_only").sum()),
                     "median |dv change| [m/s]": float((g.a_dv - g.b_dv).abs().median())})
    t = pd.DataFrame(rows)
    order = list(STAGELABEL)
    t["o"] = t.key.map({k: i for i, k in enumerate(order)})
    t = t.sort_values(["cases", "o"]).drop(columns=["o"])
    RU.writetable(t.drop(columns=["key"]), "operationalstages",
                  "Staged comparison, split into representative cases (1 and 3 km, four directions, 6n, 0.05 m/s^2) and "
                  "boundary cases (capture-limit and threshold bracket ends; deliberately the most sensitive scenarios, so "
                  "their medians are not typical) (circular 400 km chief, i = 51.6 deg, "
                  "one orbit): peak position difference between the two branches of each stage, and the number of "
                  "cases whose primary terminal-arrival label changes between them ('a arrives, b not' counts the "
                  "first-named branch arriving alone). Spacecraft Cd 2.2, A 1 m^2, m 100 kg; the differential-drag "
                  "rows use an exploratory 2x / 0.5x chaser area.", "tab:operational",
                  fmt={"median peak diff [m]": "{:.3g}", "90th pct [m]": "{:.3g}", "max [m]": "{:.3g}",
                       "median |dv change| [m/s]": "{:.3g}"})
    print(t.to_string())
    import matplotlib.pyplot as plt
    keys = ["model", "pert_J2", "pert_drag", "diff_drag_hi", "sampling_1s", "sampling_10s", "frame_fix"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    cas = st[st.stage == "model"].sort_values("max_pos_disc_m").sid.tolist()
    xpos = {s: i for i, s in enumerate(cas)}
    styles = {"model": (RU.BLUE, "o"), "pert_J2": (RU.ORANGE, "s"), "pert_drag": (RU.AQUA, "^"), "diff_drag_hi": (RU.VIOLET, "D"),
              "sampling_1s": (RU.GRAYDARK, "v"), "sampling_10s": (RU.INK, "x"), "frame_fix": (RU.MUTED, "+")}
    data = []
    for k in keys:
        g = st[st.stage == k]
        c, m = styles[k]
        x = [xpos[s] for s in g.sid if s in xpos]
        y = [v for s, v in zip(g.sid, g.max_pos_disc_m) if s in xpos]
        ax.scatter(x, np.clip(y, 1e-10, None), s=14, color=c, marker=m, label=STAGELABEL[k], lw=0.8)
        for s, v in zip(g.sid, g.max_pos_disc_m):
            data.append({"stage": k, "sid": s, "peak_diff_m": v})
    ax.set_yscale("log")
    ax.axhline(1.0, color=RU.MUTED, lw=0.8)
    ax.axhline(0.1, color=RU.MUTED, lw=0.8, ls=(0, (1, 2)))
    ax.set_xlabel("case (sorted by two-body CW-to-reference peak difference)")
    ax.set_ylabel("peak position difference [m]")
    ax.set_xticks([])
    ax.legend(fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    RU.save(fig, "operationalstages", pd.DataFrame(data))


if __name__ == "__main__":
    main()
