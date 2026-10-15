"""Analyse the mechanism experiments (step 6) against the main grid.

Usage: python experiments/ext/analyze_mechanisms.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import report_utils as RU  # noqa: E402

from cwvalid.runner import load_results  # noqa: E402

REF = "eci"
MG = os.path.join(RU.RESULTS, "main_grid", "grid")
ME = os.path.join(RU.RESULTS, "mechanisms")
D = f"cw_vs_{REF}_max_pos_disc_m"


def log_cross(seps, vals, level):
    """All upward crossings of ``level`` by log-log interpolation (grid-limited)."""
    s, v = np.asarray(seps, float), np.asarray(vals, float)
    o = np.argsort(s)
    s, v = s[o], v[o]
    out = []
    for i in range(1, len(s)):
        if v[i - 1] < level <= v[i] and v[i - 1] > 0:
            lx = np.log(s[i - 1]) + (np.log(level) - np.log(v[i - 1])) * (np.log(s[i]) - np.log(s[i - 1])) / (np.log(v[i]) - np.log(v[i - 1]))
            out.append(float(np.exp(lx)))
    return out


def saturation_vs_unrestricted(g, u):
    import matplotlib.pyplot as plt
    rows, data = [], []
    for ax in ["+R", "-R", "+V", "-V", "+H", "-H"]:
        for w in (4.0, 6.0, 8.0):
            uu = u[(u.meta_axis == ax) & (u.wn_ratio == w)].sort_values("sep_m")
            r = {"direction": RU.AXIS_LABEL[ax], "wn/n": w}
            c = log_cross(uu.sep_m, uu[D], 1.0)
            r["unrestricted 1 m crossing [km]"] = c[0] / 1e3 if c else np.nan
            r["unrestricted outcomes"] = ";".join(f"{k}:{v}" for k, v in uu.outcome_terminal_cw.value_counts().items())
            r["unrestricted max disc @50 km [m]"] = float(uu[D].iloc[-1])
            for a in (0.01, 0.05, 0.10):
                gg = g[(g.meta_axis == ax) & (g.wn_ratio == w) & (g.a_max == a)].sort_values("sep_m")
                c = log_cross(gg.sep_m, gg[D], 1.0)
                r[f"a_max={a:g} 1 m crossing [km]"] = c[0] / 1e3 if c else np.nan
            rows.append(r)
    t = pd.DataFrame(rows)
    RU.write_table(t, "tab_mech_saturation", "Mechanism A: first 1 m crossing of the peak CW-to-reference difference "
                   "(log-log interpolation on the 40-point grid) with the unrestricted PD law (same gains, no cap; a "
                   "diagnostic, not an available actuator) and with each acceleration cap; outcome classes of the "
                   "unrestricted runs.", "tab:mech_sat",
                   fmt={c: "{:.2f}" for c in t.columns if "km" in c} | {"unrestricted max disc @50 km [m]": "{:.3g}"})
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)
    for a_, ax in zip(axs, ["+R", "+V", "+H"]):
        uu = u[(u.meta_axis == ax) & (u.wn_ratio == 6.0)].sort_values("sep_m")
        a_.loglog(uu.sep_m, uu[D], color=RU.INK2, ls=":", marker="x", ms=3, label="unrestricted")
        for am, col, mk in ((0.01, RU.ORANGE, "s"), (0.05, RU.BLUE, "o"), (0.10, RU.AQUA, "^")):
            gg = g[(g.meta_axis == ax) & (g.wn_ratio == 6.0) & (g.a_max == am)].sort_values("sep_m")
            a_.loglog(gg.sep_m, gg[D], color=col, marker=mk, ms=3, label=f"a$_{{max}}$={am:g} m/s$^2$")
            for r in gg.itertuples():
                data.append({"axis": ax, "a_max": am, "sep_m": r.sep_m, "peak_disc_m": getattr(r, D)})
        for r in uu.itertuples():
            data.append({"axis": ax, "a_max": "unrestricted", "sep_m": r.sep_m, "peak_disc_m": getattr(r, D)})
        a_.axhline(1.0, color=RU.MUTED, lw=0.8)
        a_.set_title(f"{RU.AXIS_LABEL[ax]}, $\\omega_n$=6n", fontsize=8)
        a_.set_xlabel("separation [m]")
    axs[0].set_ylabel("peak CW-to-reference difference [m]")
    axs[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    RU.save(fig, "fig_mech_saturation", pd.DataFrame(data))
    return t


def sign_comparison(g):
    rows = []
    for base in ("R", "V", "H"):
        for w in (4.0, 6.0, 8.0):
            for a in (0.01, 0.05, 0.10):
                p = g[(g.meta_axis == f"+{base}") & (g.wn_ratio == w) & (g.a_max == a)].sort_values("sep_m")
                m = g[(g.meta_axis == f"-{base}") & (g.wn_ratio == w) & (g.a_max == a)].sort_values("sep_m")
                ratio = (p[D].to_numpy() / m[D].to_numpy())
                cp_, cm_ = log_cross(p.sep_m, p[D], 1.0), log_cross(m.sep_m, m[D], 1.0)
                rows.append({"axis": base, "wn/n": w, "a_max": a,
                             "+ 1 m crossing [km]": cp_[0] / 1e3 if cp_ else np.nan,
                             "- 1 m crossing [km]": cm_[0] / 1e3 if cm_ else np.nan,
                             "median disc ratio (+/-)": float(np.median(ratio)),
                             "+ terminal arrivals (NL)": int(p[f"{REF}_terminal_arrived"].sum()),
                             "- terminal arrivals (NL)": int(m[f"{REF}_terminal_arrived"].sum()),
                             "+ outcome disagreements": int(p.outcome_terminal_cw.isin(["cw_only", "nl_only"]).sum()),
                             "- outcome disagreements": int(m.outcome_terminal_cw.isin(["cw_only", "nl_only"]).sum())})
    t = pd.DataFrame(rows)
    RU.write_table(t, "tab_mech_sign", "Mechanism B: positive versus negative offsets on the main grid (40 "
                   "separations each): first 1 m crossing, median ratio of the peak differences, and counts of "
                   "reference terminal arrivals and CW/reference outcome disagreements.", "tab:mech_sign",
                   fmt={"+ 1 m crossing [km]": "{:.2f}", "- 1 m crossing [km]": "{:.2f}", "median disc ratio (+/-)": "{:.3f}"})
    return t


def energy(g, e):
    import matplotlib.pyplot as plt
    rows, data = [], []
    for ax in ("+R", "-R"):
        for w in (4.0, 6.0, 8.0):
            for a in (0.01, 0.05, 0.10):
                r = {"direction": RU.AXIS_LABEL[ax], "wn/n": w, "a_max": a}
                for vel, src in (("zero", g), ("em1", e), ("em_exact", e)):
                    q = src[(src.meta_axis == ax) & (src.wn_ratio == w) & (src.a_max == a)]
                    if vel != "zero":
                        q = q[q.meta_velocity == vel]
                    q = q.sort_values("sep_m")
                    c = log_cross(q.sep_m, q[D], 1.0)
                    r[f"{vel} 1 m [km]"] = c[0] / 1e3 if c else np.nan
                    r[f"{vel} NL arrivals"] = int(q[f"{REF}_terminal_arrived"].sum())
                    r[f"{vel} disagreements"] = int(q.outcome_terminal_cw.isin(["cw_only", "nl_only"]).sum())
                    if w == 6.0 and a == 0.05:
                        for x in q.itertuples():
                            data.append({"axis": ax, "velocity": vel, "sep_m": x.sep_m, "peak_disc_m": getattr(x, D),
                                         "nl_arrived": getattr(x, f"{REF}_terminal_arrived"), "cw_arrived": x.cw_terminal_arrived})
                rows.append(r)
    t = pd.DataFrame(rows)
    RU.write_table(t, "tab_mech_energy", "Mechanism C: radial offsets released with zero LVLH velocity, the "
                   "first-order energy-matched rate ydot0 = -2 n x0, and the exact equal-energy rate; first 1 m "
                   "crossing, reference terminal arrivals (of 40) and outcome disagreements.", "tab:mech_energy",
                   fmt={c: "{:.2f}" for c in t.columns if "[km]" in c})
    D_ = pd.DataFrame(data)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for a_, ax in zip(axs, ("+R", "-R")):
        for vel, col, mk, ls in (("zero", RU.BLUE, "o", "-"), ("em1", RU.ORANGE, "s", "--"), ("em_exact", RU.AQUA, "^", ":")):
            q = D_[(D_.axis == ax) & (D_.velocity == vel)]
            lab = {"zero": "zero velocity", "em1": "first-order matched", "em_exact": "exact energy matched"}[vel]
            a_.loglog(q.sep_m, q.peak_disc_m, color=col, marker=mk, ls=ls, ms=3, label=lab)
            f = q[~q.nl_arrived.astype(bool)]
            a_.scatter(f.sep_m, f.peak_disc_m, s=30, facecolors="none", edgecolors=col, lw=0.8)
        a_.axhline(1.0, color=RU.MUTED, lw=0.8)
        a_.set_title(f"{RU.AXIS_LABEL[ax]}, $\\omega_n$=6n, a$_{{max}}$=0.05", fontsize=8)
        a_.set_xlabel("separation [m]")
    axs[0].set_ylabel("peak CW-to-reference difference [m]")
    axs[0].legend(fontsize=6.5, loc="upper left")
    axs[1].text(0.02, 0.02, "open circles: reference does not arrive", transform=axs[1].transAxes, fontsize=6.5, color=RU.INK2)
    fig.tight_layout()
    RU.save(fig, "fig_mech_energy", D_)
    return t


def horizon(g, h):
    key = ["meta_axis", "wn_ratio", "a_max", "meta_grid_index"]
    m = g.merge(h, on=key, suffixes=("_1", "_2"))
    rows = []
    for (w, a), s in m.groupby(["wn_ratio", "a_max"]):
        rows.append({"wn/n": w, "a_max": a, "N": len(s),
                     "NL arrivals 1 orbit": int(s[f"{REF}_terminal_arrived_1"].sum()),
                     "NL arrivals 2 orbits": int(s[f"{REF}_terminal_arrived_2"].sum()),
                     "CW arrivals 1 orbit": int(s["cw_terminal_arrived_1"].sum()),
                     "CW arrivals 2 orbits": int(s["cw_terminal_arrived_2"].sum()),
                     "disagreements 1 orbit": int(s["outcome_terminal_cw_1"].isin(["cw_only", "nl_only"]).sum()),
                     "disagreements 2 orbits": int(s["outcome_terminal_cw_2"].isin(["cw_only", "nl_only"]).sum()),
                     "peak disc changed >1% (arrived both)": int(((s[f"{D}_2"] / s[f"{D}_1"] - 1).abs() > 0.01)[s[f"{REF}_terminal_arrived_1"] & s[f"{REF}_terminal_arrived_2"]].sum()),
                     "disc>1 m 1 orbit": int((s[f"{D}_1"] > 1).sum()), "disc>1 m 2 orbits": int((s[f"{D}_2"] > 1).sum())})
    t = pd.DataFrame(rows)
    RU.write_table(t, "tab_mech_horizon", "Mechanism D: one- versus two-orbit horizon for the full main grid (same "
                   "initial states and controller settings); terminal arrivals, outcome disagreements and peak-"
                   "difference changes.", "tab:mech_horizon")
    return t


def main():
    RU.setup()
    g = load_results(MG)
    out = {}
    if os.path.exists(os.path.join(ME, "unrestricted", "results.jsonl")):
        out["sat"] = saturation_vs_unrestricted(g, load_results(os.path.join(ME, "unrestricted")))
        print(out["sat"].to_string())
    out["sign"] = sign_comparison(g)
    print(out["sign"].to_string())
    if os.path.exists(os.path.join(ME, "energy", "results.jsonl")):
        out["energy"] = energy(g, load_results(os.path.join(ME, "energy")))
        print(out["energy"].to_string())
    if os.path.exists(os.path.join(ME, "horizon2", "results.jsonl")):
        out["h"] = horizon(g, load_results(os.path.join(ME, "horizon2")))
        print(out["h"].to_string())


if __name__ == "__main__":
    main()
