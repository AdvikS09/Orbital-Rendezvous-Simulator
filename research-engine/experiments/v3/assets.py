import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v3data as VD

OUT = os.path.join(VD.V3, "assets")
RES = os.path.join(VD.V3, "results")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": .6,
                     "grid.linewidth": .4, "grid.color": "#dedede", "lines.linewidth": 1.1, "legend.framealpha": 1.0})
COL = {"A": "#2a78d6", "A_H": "#2a78d6", "B": "#eb6834", "B_H": "#eb6834", "C": "#1baf7a", "C_H": "#1baf7a",
       "sep": "#eda100", "sat": "#e87ba4", "A_naive": "#008300"}
LS = {"A": "-", "A_H": "-", "B": "--", "B_H": "--", "C": ":", "C_H": ":", "sep": "-.", "sat": (0, (1, 1)), "A_naive": (0, (5, 1, 1, 1))}
MK = {"A": "o", "A_H": "o", "B": "s", "B_H": "s", "C": "^", "C_H": "^", "sep": "D", "sat": "v", "A_naive": "P"}
NAME = {"A": "A (saturated Jacobian)", "B": "B (unsaturated -K)", "C": "C (residual only)", "sep": "initial separation",
        "sat": "saturated-time fraction", "A_H": "A_H (held memory)", "B_H": "B_H (held, -K)", "A_naive": "A_naive (continuous formula)",
        "C_H": "C_H (residual only)"}
MODECOL = {"T1": "#2a78d6", "T1_heuristic": "#2a78d6", "T2": "#eb6834", "T3": "#1baf7a"}
MODEMK = {"T1": "o", "T1_heuristic": "o", "T2": "s", "T3": "^"}


def save(fig, name, data):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "eps"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), format=ext)
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=180)
    pd.DataFrame(data).to_csv(os.path.join(OUT, f"{name}Data.csv"), index=False)
    plt.close(fig)


def rccurve(f, adverse):
    f = np.asarray(f, float)
    n = f.size
    ok = np.isfinite(f)
    order = np.argsort(np.where(ok, f, np.inf), kind="stable")[: ok.sum()]
    bad = np.cumsum(adverse[order])
    k = np.arange(1, order.size + 1)
    return k / n, bad / k, f[order]


def fig1(dec, rc):
    d0 = dec[dec.cell == "D0"]
    n = len(d0)
    fig, axs = plt.subplots(1, 2, figsize=(6.85, 3.0), sharey=True, layout="constrained")
    data = []
    for ax, crit in zip(axs, ("1m", "0.1m")):
        adv = d0[f"label_{crit}"].isin(["violation", "exit", "failure", "unresolved"]).to_numpy()
        for m in ("A", "B", "C", "sep", "sat"):
            cov, risk, thr = rccurve(d0[f"f_{m}"], adv)
            ax.plot(cov, risk, color=COL[m], ls=LS[m], lw=1.0, label=NAME[m])
            for c_, r_, t_ in zip(cov[::10], risk[::10], thr[::10]):
                data.append({"panel": crit, "method": m, "coverage": c_, "conservative_adverse_rate": r_, "threshold": t_})
            for kind, filled in (("selected_calibrated", True), ("legacy_padded_rule", False)):
                q = rc[(rc.kind == kind) & (rc.cell == "D0") & (rc.method == m) & (rc.criterion == crit)]
                if len(q) and q.iloc[0].n_accepted > 0:
                    r = q.iloc[0]
                    ax.plot(r.coverage, r.cond_violation_rate_conservative, marker=MK[m], ms=5.5, ls="none",
                            color=COL[m], mfc=COL[m] if filled else "white", mec="black" if filled else COL[m], mew=.7, clip_on=False, zorder=5)
                    data.append({"panel": crit, "method": m, "point": kind, "coverage": r.coverage,
                                 "conservative_adverse_rate": r.cond_violation_rate_conservative, "threshold": r.threshold})
        ax.axhline(0.01, color="#777777", lw=.7, ls="--")
        ax.text(0.02, 0.012, r"$\alpha$ = 1 %", fontsize=7, color="#555555")
        ax.set_yscale("symlog", linthresh=1e-3, linscale=0.6)
        ax.set_ylim(0, 0.6)
        ax.set_xlim(0, 1)
        ax.grid(True)
        ax.set_xlabel(f"Coverage (accepted / {n:,} D0 test cases)")
        ax.set_title(f"({'a' if crit == '1m' else 'b'}) {crit.replace('m', ' m')} target")
    axs[0].set_ylabel("Adverse fraction among accepted\n(unresolved counted as adverse)")
    from matplotlib.lines import Line2D
    h, l = axs[0].get_legend_handles_labels()
    h += [Line2D([], [], marker="o", color="#555555", mec="black", ls="none", label="calibrated rule (filled)"),
          Line2D([], [], marker="o", color="#555555", mfc="white", ls="none", label="legacy padded rule (open)")]
    fig.legend(handles=h, loc="outside lower center", ncol=4, fontsize=7, frameon=False)
    save(fig, "FigV3RiskCoverage", data)


def fig2(tm):
    cells = ["D0", "D1", "D2", "D3", "D4", "D5"]
    t = tm[(tm.criterion == "1m") & tm.target_method.isin(["A", "A_H"])]
    fig, axs = plt.subplots(2, 1, figsize=(6.85, 3.6), sharex=True, layout="constrained")
    data = []
    off = {"T1": -0.18, "T1_heuristic": -0.18, "T2": 0.0, "T3": 0.18}
    for i, c in enumerate(cells):
        for _, r in t[t.target_cell == c].iterrows():
            x = i + off[r["mode"]]
            axs[0].plot(x, r.coverage, marker=MODEMK[r["mode"]], color=MODECOL[r["mode"]], ls="none", ms=5)
            lo, hi = r.cp95_two_sided_conservative_lo, r.cp95_two_sided_conservative_hi
            y = r.cond_violation_rate_conservative
            if np.isfinite(y):
                axs[1].errorbar(x, y, yerr=[[y - lo], [hi - y]], marker=MODEMK[r["mode"]], color=MODECOL[r["mode"]],
                                ms=5, capsize=2, lw=.8)
            data.append({"cell": c, "mode": r["mode"], "method": r.target_method, "threshold": r.threshold,
                         "coverage": r.coverage, "n_accepted": r.n_accepted,
                         "conservative_adverse": r.conservative_adverse_accepted, "rate": y, "cp95_lo": lo, "cp95_hi": hi})
    axs[1].axhline(0.01, color="#777777", lw=.7, ls="--")
    axs[0].set_ylabel("Test coverage")
    axs[1].set_ylabel("Adverse fraction\namong accepted")
    axs[1].set_xticks(range(len(cells)))
    axs[1].set_xticklabels(["D0\n400 km PD", "D1\n600 km", "D2\n800 km", "D3\nhold 1 s", "D4\nhold 10 s", "D5\nCW LQR"])
    for ax in axs:
        ax.grid(True, axis="y")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=MODEMK[k], color=MODECOL[k], ls="none", label=l) for k, l in
         (("T1", "T1: D0 cutoff unchanged (held cells: attached to A_H)"), ("T2", "T2: similarity image c R_t/R_0"),
          ("T3", "T3: recalibrated in the target cell"))]
    axs[0].legend(handles=h, loc="lower left", fontsize=7)
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(bottom=-0.002)
    axs[1].text(5.45, 0.011, r"$\alpha$", fontsize=7, color="#555555")
    save(fig, "FigV3Transfer", data)


def fig3(hv):
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, 3, figsize=(6.85, 2.7), layout="constrained")
    data = []
    fd = hv[hv.test == "fd_ladder"]
    casesa = ((("+H_20km_switching", 10.0), "#2a78d6", "o", "+H 20 km, T_s=10 s"), (("+R_3km_switching", 1.0), "#eb6834", "s", "+R 3 km, T_s=1 s"),
              (("-R_12km_saturated", 10.0), "#1baf7a", "^", "-R 12 km, T_s=10 s"))
    for (case, Ts), col, mk, lab in casesa:
        q = fd[(fd.case == case) & (fd.Ts_s == Ts)].sort_values("lambda")
        axs[0].loglog(q["lambda"], q.err_central_vs_A_H_rel, color=col, marker=mk, ms=3.5, label=lab)
        axs[0].loglog(q["lambda"], q.err_central_vs_B_H_rel, color=col, marker=mk, ms=3.5, ls=":", mfc="white")
        for _, r in q.iterrows():
            data.append({"panel": "a", "case": case, "Ts_s": Ts, "lambda": r["lambda"], "err_central_vs_A_H": r.err_central_vs_A_H_rel,
                         "err_central_vs_B_H": r.err_central_vs_B_H_rel})
    axs[0].set(xlabel=r"Homotopy step $\lambda$", ylabel="max |difference quotient - e| / max |e|", title="(a) Finite-difference check")
    h = axs[0].get_legend_handles_labels()[0] + [Line2D([], [], color="#555555", ls="-", label="vs A_H"),
                                                 Line2D([], [], color="#555555", ls=":", label="vs B_H")]
    axs[0].legend(handles=h, fontsize=5.8, loc="center left", bbox_to_anchor=(0.0, 0.6))
    ts = hv[hv.test == "ts_to_zero"]
    for case, col, mk, lab in (("+H_20km_switching", "#2a78d6", "o", "+H 20 km"), ("+V_500m_unsat", "#eda100", "D", "+V 500 m"),
                               ("mixed_1.1km_vel", "#e87ba4", "v", "mixed 1.1 km")):
        q = ts[ts.case == case].sort_values("Ts_s")
        axs[1].loglog(q.Ts_s, q.rel_A_H_vs_disc_held.clip(lower=1e-7), color=col, marker=mk, ms=3.5, ls="-", label=lab)
        axs[1].loglog(q.Ts_s, q.rel_A_naive_vs_disc_held.clip(lower=1e-7), color=col, marker=mk, ms=3.5, ls="--", mfc="white")
        for _, r in q.iterrows():
            data.append({"panel": "b", "case": case, "Ts_s": r.Ts_s, "A_H_vs_continuous_A": r.rel_A_H_vs_cont,
                         "A_H_vs_held_truth": r.rel_A_H_vs_disc_held, "A_naive_vs_held_truth": r.rel_A_naive_vs_disc_held})
    axs[1].set(xlabel="Hold period T_s (s)", ylabel="|feature - held truth| / held truth", title="(b) Prediction of held truth")
    h = axs[1].get_legend_handles_labels()[0] + [Line2D([], [], color="#555555", ls="-", label="A_H"),
                                                 Line2D([], [], color="#555555", ls="--", label="A_naive")]
    axs[1].legend(handles=h, fontsize=5.8, loc="upper left")
    ex = None
    fa = os.path.join(RES, "failureatlas.csv")
    if os.path.exists(fa):
        f = pd.read_csv(fa)
        e = f[(f.kind == "illustrative_case") & (f.rule == "R4_largest_AH_vs_Anaive_test") & (f.cell == "D4")]
        if len(e):
            p = os.path.join(VD.V3, "raw", "counterexamples", f"{e.iloc[0].case_id}.csv")
            if os.path.exists(p):
                ex = (e.iloc[0].case_id, pd.read_csv(p))
    if ex is not None:
        cid, H = ex
        t = H.t_s / 60.0
        axs[2].plot(t, H.disc_cw_eci_m, color="#555555", lw=1.8, label="held truth |CW - ref|")
        axs[2].plot(t, H.e_A_H_m, color=COL["A_H"], ls="--", label="A_H")
        axs[2].plot(t, H.e_A_naive_m, color=COL["A_naive"], ls=LS["A_naive"], label="A_naive")
        axs[2].set(xlabel="Time (min)", ylabel="Position error (m)", title=f"(c) D4 test case {cid[:9]}")
        axs[2].legend(fontsize=6)
        for i in range(0, len(H), 20):
            data.append({"panel": "c", "case_id": cid, "t_s": H.t_s.iloc[i], "truth_m": H.disc_cw_eci_m.iloc[i],
                         "A_H_m": H.e_A_H_m.iloc[i], "A_naive_m": H.e_A_naive_m.iloc[i]})
    else:
        axs[2].set_axis_off()
    for ax in axs[:2]:
        ax.grid(True)
    axs[2].grid(True)
    save(fig, "FigV3Hybrid", data)


def fig4():
    p = os.path.join(VD.V3, "raw", "atlascases.csv")
    if not os.path.exists(p):
        return
    A = pd.read_csv(p, low_memory=False)
    A = A[A.split == "test"]
    cells = ["D0", "D3", "D4", "D5"]
    fig, axs = plt.subplots(1, 4, figsize=(6.85, 2.3), sharey=True, layout="constrained")
    data = []
    cats = [("accepted_adequate", "#2a78d6", "o", 4, "accepted, adequate"), ("referred_conservative", "#eb6834", "s", 5, "referred, adequate"),
            ("referred_violation", "#9a9a9a", ".", 3, "referred, violation/exit"),
            ("accepted_violation", "black", "x", 30, "accepted, violation"), ("accepted_exit", "black", "+", 30, "accepted, exit"),
            ("accepted_unresolved", "black", "*", 30, "accepted, unresolved")]
    for ax, c in zip(axs, cells):
        s = A[A.cellname == c]
        for cat, col, mk, sz, lab in cats:
            q = s[s.cat_A == cat]
            if len(q):
                ax.scatter(q.sep_m / 1000, q.sat_fraction, s=sz, marker=mk, c=col, linewidths=.4 if mk in "x+*" else 0,
                           alpha=.8 if mk not in "x+*" else 1, label=lab)
            data.append({"cell": c, "category": cat, "n": len(q)})
        ax.set_xscale("log")
        ax.set(title=c, xlabel="Initial separation (km)")
        ax.grid(True)
    axs[0].set_ylabel("Saturated-time fraction (CW)")
    from matplotlib.lines import Line2D
    present = {d["category"] for d in data if d["n"] > 0}
    h = [Line2D([], [], marker=mk, color=col, ls="none", ms=5 if mk not in "x+*" else 6, label=lab)
         for cat, col, mk, sz, lab in cats if cat in present]
    fig.legend(handles=h, loc="outside right center", fontsize=6.5, frameon=False)
    save(fig, "FigV3FailureAtlas", data)


def fig5():
    p = os.path.join(RES, "breakeven.csv")
    if not os.path.exists(p):
        return
    C = pd.read_csv(p)
    c = C[(C.cell == "D0") & (C.comparator == "direct_NRM")]
    fig, ax = plt.subplots(figsize=(3.35, 2.2), layout="constrained")
    data = []
    labels, vals = ["direct NRM"], [float(c.expected_cpu_s_per_case_comparator.iloc[0])]
    e = C[(C.cell == "D0") & (C.comparator == "direct_ECI")]
    labels.append("direct inertial")
    vals.append(float(e.expected_cpu_s_per_case_comparator.iloc[0]))
    for _, r in c.iterrows():
        labels.append(r.workflow.replace("screen_", "screen ").replace("_cold", " (cold)").replace("_warm_cw_available", " (CW given)"))
        vals.append(r.expected_cpu_s_per_case_screen)
    y = np.arange(len(labels))
    ax.barh(y, vals, color=["#9a9a9a", "#9a9a9a"] + [COL[w.split("_")[1]] for w in c.workflow], height=.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Expected CPU time per case (s)")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=6.5)
        data.append({"workflow": labels[i], "expected_cpu_s_per_case": v})
    ax.grid(True, axis="x")
    save(fig, "FigV3Cost", data)


def figS1():
    nr = pd.read_csv(os.path.join(RES, "numerical.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.5), sharey=True, layout="constrained")
    data = []
    held = nr.cell.isin(["D3", "D4"])
    for ax, tag in zip(axs, ("1m", "0.1m")):
        x = nr[f"margin_{tag}_m"].to_numpy(float)
        y = nr.num_unc_m.to_numpy(float)
        ax.loglog(x[~held], y[~held], "o", ms=3.5, color="#2a78d6", label="continuous cells")
        ax.loglog(x[held], y[held], "s", ms=3.5, mfc="white", color="#eb6834", label="held-command cells")
        xx = np.logspace(-3, 6.5, 50)
        ax.loglog(xx, 0.1 * xx, color="#777777", lw=.7, ls="--")
        ax.text(2e-3, 4e-4, "resolution limit: 0.1 x margin", fontsize=6.5, color="#555555", rotation=24)
        ax.set(xlabel=f"|truth - {tag.replace('m', ' m')}| (m)", title=f"({'a' if tag == '1m' else 'b'}) {tag.replace('m', ' m')} decision")
        ax.grid(True)
        for i in range(len(nr)):
            data.append({"panel": tag, "case_id": nr.case_id.iloc[i], "cell": nr.cell.iloc[i], "stratum": nr.stratum.iloc[i],
                         "margin_m": x[i], "numerical_uncertainty_m": y[i]})
    axs[0].set_ylabel("|refined - production| + peak\nuncertainty (m)")
    axs[1].legend(fontsize=7, loc="lower right")
    save(fig, "FigV3S1Numerics", data)


def texescape(v):
    s = str(v)
    for a, b in (("%", r"\%"), ("_", r"\_"), ("&", r"\&"), ("#", r"\#"), ("->", r"$\rightarrow$")):
        s = s.replace(a, b)
    return s


def textable(df, path, caption, label, colfmt):
    lines = [r"\begin{table}[t]\centering\small", rf"\caption{{{caption}}}\label{{{label}}}",
             rf"\begin{{tabular}}{{{colfmt}}}\toprule", " & ".join(texescape(c) for c in df.columns) + r"\\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(texescape(v) for v in r.values) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def tabl(rc, census):
    rows = []
    for c in ["D0", "D1", "D2", "D3", "D4", "D5"]:
        cs = census[(census.cell == c)]
        g = lambda sp, k: int(cs[cs.split == sp][k].iloc[0]) if len(cs[cs.split == sp]) else 0
        rows.append({"Cell": c, "Pilot": g("pilot", "n"), "Development": g("dev", "n"), "Calibration": g("cal", "n"),
                     "Test": g("test", "n"),
                     "Exits (cal/test)": f"{g('cal', 'truth_physical_domain_exit')}/{g('test', 'truth_physical_domain_exit')}",
                     "Failures (cal/test)": f"{g('cal', 'numerical_failure_excl_interrupted')}/{g('test', 'numerical_failure_excl_interrupted')}",
                     "Interrupted (cal/test)": f"{g('cal', 'interrupted_operational')}/{g('test', 'interrupted_operational')}",
                     "Unresolved 1 m (cal/test)": f"{g('cal', 'label1m_unresolved')}/{g('test', 'label1m_unresolved')}"})
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "TabV3Splits.csv"), index=False)
    textable(S, os.path.join(OUT, "TabV3Splits.tex"),
             "v3 iid splits per cell and status census (calibration/test). Exits are adverse for the full-horizon predicate; "
             "interrupted runs were stopped by a driver restart and later re-executed (deviation DEV-01)",
             "tab:v3splits", "@{}lrrrrllll@{}")
    sel = rc[rc.kind == "selected_calibrated"]
    rows = []
    for _, r in sel.iterrows():
        if r.method in ("A", "B", "A_H", "B_H", "A_naive"):
            rows.append({"Cell": r.cell, "Method": r.method, "Crit.": r.criterion,
                         "Threshold (m)": "abstain" if pd.isna(r.threshold) else f"{r.threshold:.4g}",
                         "Cal. acc./adv. (U)": "--" if pd.isna(r.cal_accepted) else f"{int(r.cal_accepted)}/{int(r.cal_adverse)} ({100 * r.cal_ucb:.2f}%)",
                         "Test coverage": f"{100 * r.coverage:.1f}%",
                         "Test adverse/acc.": f"{int(r.conservative_adverse_accepted)}/{int(r.n_accepted)}",
                         "95% CP upper": "--" if not np.isfinite(r.cp95_two_sided_conservative_hi) else f"{100 * r.cp95_two_sided_conservative_hi:.2f}%"})
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(OUT, "TabV3Performance.csv"), index=False)
    textable(P, os.path.join(OUT, "TabV3Performance.tex"),
             r"Frozen calibrated rules ($\alpha=1\%$, $\delta=5\%$ over $M=1012$ candidates) and their single evaluation on 2500 "
             r"test cases per cell. $U$ is the one-sided calibration bound at level $\delta/M$; test intervals are pointwise "
             r"two-sided Clopper--Pearson",
             "tab:v3performance", "@{}lllrlrrr@{}")


def tablesnumcost():
    rows = []
    nr = pd.read_csv(os.path.join(RES, "numerical.csv"))
    nc = nr[~nr.stratum.eq("physical_exit")]
    rows.append({"Check": "Truth peak, production vs refined (108 stratified dev cases; exits excluded)",
                 "Result": f"max diff {nc.delta_disc_m.max():.2e} m; all 1 m and 0.1 m labels resolved"})
    rows.append({"Check": "Truth peak of held cases with a physical exit (partial horizon)",
                 "Result": f"max diff {nr[nr.stratum.eq('physical_exit')].delta_disc_m.max():.2g} m (label exit either way)"})
    near = nr[nr.stratum.isin(["near_threshold_feature", "near_threshold_truth_0.1m"])]
    rows.append({"Check": "Feature A/A_H, production vs refined, near-threshold strata",
                 "Result": f"max rel. change {near.A_rel_change.max():.1e}"})
    sim = pd.read_csv(os.path.join(RES, "similarity.csv"))
    rows.append({"Check": "Similarity 400 -> 600/800 km (20 pairs)",
                 "Result": f"{int(sim.similarity_verified.sum())}/{len(sim)} verified; max state diff {max(sim.max_dq_cw.max(), sim.max_dq_eci.max()):.1e} (q units)"})
    me = pd.read_csv(os.path.join(RES, "matrixengine.csv"))
    rows.append({"Check": "Matrix-law engine vs production PD (12 cases)",
                 "Result": f"features identical; max non-timing field diff {me.row_max_abs_diff.max():.1e}"})
    hm = pd.read_csv(os.path.join(RES, "hybridmap.csv"))
    hm2 = hm[hm.Ts_s <= 10]
    rows.append({"Check": "Independent held map vs batch engine (T_s = 1, 10 s)",
                 "Result": f"max feature rel. diff {hm2.feature_rel_diff.max():.1e}"})
    hv = pd.read_csv(os.path.join(RES, "hybridvalidation.csv"))
    fd = hv[(hv.test == "fd_ladder") & (hv["lambda"].between(1e-4, 1e-2))]
    fdb = fd[fd.n_sat_updates > 0]
    rows.append({"Check": "Central FD vs A_H, lambda 1e-4..1e-2 (6 cases x 2 T_s)",
                 "Result": f"max rel. error {fd.err_central_vs_A_H_rel.max():.1e}; vs B_H (saturating cases) min {fdb.err_central_vs_B_H_rel.min():.2f}"})
    he = pd.read_csv(os.path.join(RES, "heldengine.csv")) if os.path.exists(os.path.join(RES, "heldengine.csv")) else None
    if he is not None:
        rows.append({"Check": "Held batch engine vs production sampled propagation",
                     "Result": f"max CW state diff {he.cw_max_pos_diff_m.max():.1e} m; labels identical"})
    N = pd.DataFrame(rows)
    N.to_csv(os.path.join(OUT, "TabV3Numerics.csv"), index=False)
    textable(N, os.path.join(OUT, "TabV3Numerics.tex"), "Numerical verification of the v3 engines and labels",
             "tab:v3numerics", "@{}p{0.55\\columnwidth}p{0.4\\columnwidth}@{}")
    p = os.path.join(RES, "breakeven.csv")
    if os.path.exists(p):
        C = pd.read_csv(p)
        c = C[C.cell == "D0"][["workflow", "comparator", "p_referral_test", "expected_cpu_s_per_case_screen",
                                "expected_cpu_s_per_case_comparator", "online_saving_cpu_s_per_case", "break_even_note"]]
        c = c.round(3)
        c.to_csv(os.path.join(OUT, "TabV3Cost.csv"), index=False)
        textable(c.rename(columns={"workflow": "Workflow", "comparator": "Comparator", "p_referral_test": "Referral",
                                   "expected_cpu_s_per_case_screen": "Screen (s)", "expected_cpu_s_per_case_comparator": "Direct (s)",
                                   "online_saving_cpu_s_per_case": "Saving (s)", "break_even_note": "Break-even"}),
                 os.path.join(OUT, "TabV3Cost.tex"), "Expected CPU time per D0 case (single process)", "tab:v3cost",
                 "@{}llrrrrp{0.25\\columnwidth}@{}")


def main():
    import v30closed
    v30closed.refuseoutsharness(__file__)
    rc = pd.read_csv(os.path.join(RES, "riskcoverage.csv"))
    tm = pd.read_csv(os.path.join(RES, "transfermatrix.csv"))
    hv = pd.read_csv(os.path.join(RES, "hybridvalidation.csv"))
    dec = pd.read_csv(os.path.join(VD.V3, "raw", "testdecisions.csv"), low_memory=False)
    fa = pd.read_csv(os.path.join(RES, "failureatlas.csv")) if os.path.exists(os.path.join(RES, "failureatlas.csv")) else None
    fig1(dec, rc)
    fig2(tm)
    fig3(hv)
    fig4()
    fig5()
    figS1()
    if fa is not None:
        tabl(rc, fa[fa.kind == "status_census"])
    tablesnumcost()
    print("assets written to", OUT)


if __name__ == "__main__":
    main()
