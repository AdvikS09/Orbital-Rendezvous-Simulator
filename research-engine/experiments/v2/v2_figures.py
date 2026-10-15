"""Publication figures for the v2 extension (Fig1-Fig8), drawn only from saved v2 outputs.

Fig1  estimated vs measured peak discrepancy for estimators A, B, C (held-out sets), by regime
Fig2  estimate/measured ratio by regime (development and fresh confirmatory sets)
Fig3  acceptance, incorrect acceptance and unnecessary referral: trajectory-only and combined rules,
      fresh confirmatory set and the three retrospective sets
Fig4  acceptance-target rules (25/50/75 % on development data): achieved acceptance and errors on the fresh set
Fig5  independent verification: peak-discrepancy agreement, entry-time agreement, solver convergence
Fig6  new arrival-boundary families: CW-minus-reference capture-limit offsets with uncertainty
Fig7  status-aware branch accounting (two-orbit campaign) and second-order failure growth

Each figure: FigN.pdf, FigN.eps (vector, fonts embedded), FigN.png (300 dpi preview), FigN_data.csv.
Usage: python experiments/v2/v2_figures.py [--only Fig1,Fig3]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_common as C  # noqa: E402
import fig_style as S  # noqa: E402

S.setup()
import matplotlib.pyplot as plt  # noqa: E402

EV = os.path.join(C.DERIVED, "estimator")
FLOOR = 1e-6
HELD = ("test", "stress", "band_test", "fresh")


def _dec(set_name):
    p = os.path.join(EV, f"decisions_{set_name}.csv")
    return pd.read_csv(p).assign(set=set_name) if os.path.exists(p) else None


def fig1():
    held = pd.concat([d for d in (_dec(s) for s in HELD) if d is not None], ignore_index=True)
    dev = _dec("dev")
    fig, axs = plt.subplots(1, 3, figsize=(S.DOUBLE, 2.45), sharey=True)
    rows = []
    for ax, est in zip(axs, ("d1", "d1u", "cres")):
        lab = S.EST_STYLE[est][3]
        for d, bg in ((dev, True), (held, False)):
            d = d[d.label_valid.astype(bool)]
            meas = np.maximum(d.disc_m.to_numpy(float), FLOOR)
            e = d[est].to_numpy(float)
            fail = ~np.isfinite(e)
            ev = np.where(fail, FLOOR / 3, np.maximum(e, FLOOR))
            if bg:
                ax.scatter(meas, ev, s=1.2, color="#d4d3cc", marker=".", lw=0, label="development (fitting)", zorder=1)
                continue
            for rg, (col, mk, rl) in S.REGIME_STYLE.items():
                m = (d.regime == rg).to_numpy() & ~fail
                ax.scatter(meas[m], ev[m], s=7, marker=mk, facecolor="none" if rg == "saturated" else col, edgecolor=col,
                           lw=0.6, label=f"held-out, {rl}", zorder=3)
            if fail.any():
                ax.scatter(meas[fail], ev[fail], s=12, marker="x", color=S.INK, label="estimate failed", zorder=4)
            for sid, st, mv, evv, rg, fl in zip(d.sid, d.set, d.disc_m, e, d.regime, fail):
                rows.append({"panel": est, "sid": sid, "set": st, "measured_peak_m": mv, "estimate_m": evv,
                             "regime": rg, "estimate_failed": bool(fl)})
        lim = [3e-7, 3e6]
        ax.plot(lim, lim, color=S.INK2, lw=0.7, ls="-")
        ax.plot(lim, [2 * x for x in lim], color=S.MUTED, lw=0.5, ls="--")
        ax.plot(lim, [x / 2 for x in lim], color=S.MUTED, lw=0.5, ls="--")
        ax.axvline(FLOOR, color=S.MUTED, lw=0.5, ls=":")
        ax.axhline(1.0, color=S.MUTED, lw=0.4, ls="-.")
        ax.axvline(1.0, color=S.MUTED, lw=0.4, ls="-.")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"({'abc'[('d1', 'd1u', 'cres').index(est)]}) {lab}")
        if est == "d1u":
            ax.set_xlabel("measured peak CW-to-reference difference [m] (values below 1e-6 m plotted at the floor, dotted)")
    axs[0].set_ylabel("CW-only estimate [m]")
    h, l = axs[0].get_legend_handles_labels()
    keep = dict(zip(l, h))
    fig.legend(keep.values(), keep.keys(), loc="lower center", ncol=5, fontsize=6, bbox_to_anchor=(0.5, -0.07), markerscale=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    S.save(fig, "Fig1", pd.DataFrame(rows))


def fig2():
    R = pd.read_csv(os.path.join(EV, "ratio_stats.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.3), sharey=True)
    regs = ["unsaturated", "saturated", "near_boundary"]
    rows = []
    for ax, st, title in zip(axs, ("dev", "fresh"), ("(a) development set (in-sample)", "(b) fresh confirmatory set")):
        q = R[R.set == st]
        for j, est in enumerate(("d1", "d1u", "cres")):
            col, mk, ls, lab = S.EST_STYLE[est]
            for i, rg in enumerate(regs):
                r = q[(q.estimator == est) & (q.regime == rg)]
                if r.empty or not np.isfinite(r.median_ratio.iloc[0]):
                    continue
                r = r.iloc[0]
                x = i + (j - 1) * 0.22
                ax.errorbar([x], [r.median_ratio], yerr=[[r.median_ratio - r.p01_ratio], [r.p99_ratio - r.median_ratio]],
                            fmt=mk, color=col, mfc=col if est != "d1u" else "white", ms=4, lw=0.8, capsize=2,
                            label=lab if i == 0 else None)
                ax.text(x, r.p99_ratio * 1.6, f"{int(r.n_ratio)}", fontsize=5.5, ha="center", color=S.INK2)
                rows.append({"panel": st, "estimator": est, "regime": rg, "n_ratio": r.n_ratio, "median": r.median_ratio,
                             "p01": r.p01_ratio, "p99": r.p99_ratio, "n_below_floor": r.n_below_floor,
                             "n_estimate_failed": r.n_estimate_failed})
        ax.axhline(1, color=S.INK2, lw=0.6)
        ax.set_yscale("log")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["unsaturated", "saturated", "near arrival\nboundary"])
        ax.set_title(title)
    axs[0].set_ylabel("estimate / measured\n(median, 1st-99th pct.)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.99, 0.01, "numbers: cases in ratio statistics", fontsize=5.5, ha="right", color=S.INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    S.save(fig, "Fig2", pd.DataFrame(rows))


def fig3():
    P = pd.read_csv(os.path.join(EV, "performance_long.csv"))
    P = P[(P.rule_status == "ok") & ~P.set.astype(str).str.contains(":")]
    traj = ["traj1_d1", "traj1_d1u", "traj1_cres", "traj1_sep", "traj1_sat"]
    comb = ["comb_d1", "comb_d1u", "comb_cres", "v1_sep_only", "v1_sat_only"]
    feat = {"traj1_d1": "d1", "traj1_d1u": "d1u", "traj1_cres": "cres", "traj1_sep": "sep", "traj1_sat": "sat",
            "comb_d1": "d1", "comb_d1u": "d1u", "comb_cres": "cres", "v1_sep_only": "sep", "v1_sat_only": "sat"}
    names = {"d1": "A", "d1u": "B", "cres": "C", "sep": "sep.", "sat": "sat."}
    sets = [s for s in ("fresh", "test", "stress", "band_test") if s in set(P.set)]
    titles = {"fresh": "fresh (confirmatory, 300)", "test": "broad test (retro., 400)", "stress": "stress (retro., 150)",
              "band_test": "band challenge (retro., 69)"}
    fig, axs = plt.subplots(2, len(sets), figsize=(S.DOUBLE, 3.9), sharey=True, squeeze=False)
    rows = []
    for i, (rules, rlab) in enumerate(((traj, "trajectory-only\n(peak <= 1 m)"), (comb, "combined / baselines\n(adequate)"))):
        for j, st in enumerate(sets):
            ax = axs[i, j]
            for k, rule in enumerate(rules):
                r = P[(P.set == st) & (P.rule == rule)]
                if r.empty:
                    continue
                r = r.iloc[0]
                col, mk, ls, _ = S.EST_STYLE[feat[rule]]
                a = 100 * r.accept_rate
                ax.errorbar([k], [a], yerr=[[a - 100 * r.accept_rate_lo], [100 * r.accept_rate_hi - a]], fmt=mk, color=col,
                            mfc=col, ms=4, lw=0.8, capsize=1.5)
                u = 100 * r.unnecessary_rate if np.isfinite(r.unnecessary_rate) else np.nan
                ax.plot([k + 0.28], [u], marker=mk, mfc="white", mec=col, ms=3.5, lw=0, color=col)
                ax.text(k, 104, f"{int(r.incorrect_accepted)}", ha="center", fontsize=6,
                        color=S.INK if r.incorrect_accepted == 0 else "#b3261e", fontweight="bold")
                rows.append({"row": rlab.replace("\n", " "), "set": st, "rule": rule, "accepted": r.accepted,
                             "n_evaluated": r.n_evaluated, "accept_pct": a, "accept_lo_pct": 100 * r.accept_rate_lo,
                             "accept_hi_pct": 100 * r.accept_rate_hi, "incorrect_accepted": r.incorrect_accepted,
                             "unnecessary_referral_pct": u, "referred": r.referred})
            ax.set_xticks(range(len(rules)))
            ax.set_xticklabels([names[feat[r]] for r in rules])
            ax.set_ylim(-3, 112)
            if i == 0:
                ax.set_title(titles[st], fontsize=6.8)
            if j == 0:
                ax.set_ylabel(rlab + "\n[%]")
    fig.text(0.01, 0.003, "filled: accepted (95% Clopper-Pearson CI); open: unnecessary referrals among referred; "
             "number above: incorrect acceptances", fontsize=6, color=S.INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    S.save(fig, "Fig3", pd.DataFrame(rows))


def fig4():
    P = pd.read_csv(os.path.join(EV, "performance_long.csv"))
    P = P[(P.rule_status == "ok") & (P.set == "fresh") & P.rule.str.startswith("acc")]
    fr = json.load(open(C.FROZEN_V2))
    fig, axs = plt.subplots(1, 2, figsize=(S.DOUBLE * 0.75, 2.3))
    rows = []
    for est in ("d1", "d1u", "cres", "sep", "sat"):
        col, mk, ls, lab = S.EST_STYLE[est]
        xs, acc, inc = [], [], []
        for p in (25, 50, 75):
            r = P[P.rule == f"acc{p}_{est}"]
            if r.empty:
                continue
            r = r.iloc[0]
            chk = fr["dev_fit_checks"][f"acc{p}_{est}"]
            xs.append(p)
            acc.append(100 * r.accept_rate)
            inc.append(100 * r.incorrect_rate if np.isfinite(r.incorrect_rate) else np.nan)
            rows.append({"feature": est, "dev_target_pct": p, "dev_achieved_pct": 100 * chk["dev_accepted"] / chk["dev_n"],
                         "fresh_accept_pct": 100 * r.accept_rate, "fresh_incorrect_pct_of_accepted": inc[-1],
                         "fresh_incorrect": r.incorrect_accepted, "fresh_accepted": r.accepted})
        axs[0].plot(xs, acc, marker=mk, color=col, ls=ls, label=lab)
        axs[1].plot(xs, inc, marker=mk, color=col, ls=ls, label=lab)
    axs[0].plot([20, 80], [20, 80], color=S.MUTED, lw=0.5)
    axs[0].set_xlabel("development acceptance target [%]")
    axs[0].set_ylabel("fresh-set acceptance [%]")
    axs[1].set_xlabel("development acceptance target [%]")
    axs[1].set_ylabel("incorrect among accepted [%]")
    axs[0].set_title("(a) achieved acceptance")
    axs[1].set_title("(b) acceptance errors (peak > 1 m)")
    axs[0].legend(fontsize=5.8)
    fig.tight_layout()
    S.save(fig, "Fig4", pd.DataFrame(rows))


def fig5():
    V = os.path.join(C.DERIVED, "verifier")
    R = pd.read_csv(os.path.join(V, "verifier_comparison.csv"))
    Cv = pd.read_csv(os.path.join(V, "verifier_convergence.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(S.DOUBLE, 2.4))
    rows = []
    for a, (col, mk) in (("cw", (S.BLUE, "o")), ("so", (S.ORANGE, "s"))):
        x = R[f"{a}_ref_peak_prod_m"].to_numpy(float)
        y = np.maximum(R[f"{a}_ref_peak_absdiff_m"].to_numpy(float), 1e-12)
        axs[0].scatter(x, y, marker=mk, s=12, facecolor=col if a == "cw" else "white", edgecolor=col, lw=0.7,
                       label={"cw": "CW vs reference", "so": "second order vs reference"}[a])
        for sid, xx, yy in zip(R.sid, x, y):
            rows.append({"panel": "a", "sid": sid, "pair": a, "peak_prod_m": xx, "abs_diff_m": yy})
    xx = np.geomspace(1e-4, 1e7, 50)
    axs[0].plot(xx, np.where(xx > 10, 1e-3 * xx, 0.01), color=S.INK2, ls="--", lw=0.6, label="initial target")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("production peak discrepancy [m]")
    axs[0].set_ylabel("|production - independent| [m]")
    axs[0].set_title("(a) peak discrepancy")
    axs[0].legend(fontsize=5.6)
    for k, (b, mk) in enumerate((("ref", "D"), ("cw", "o"), ("so", "s"))):
        imp = np.maximum(R[f"{b}_entry_impl_diff_s"].to_numpy(float), 1e-9)
        mon = np.maximum(R[f"{b}_entry_monitor_diff_s"].to_numpy(float), 1e-9)
        idx = np.arange(len(R)) + (k - 1) * 0.25
        axs[1].scatter(idx, imp, marker=mk, s=10, color=S.BLUE, lw=0.6, label="implementation" if k == 0 else None)
        axs[1].scatter(idx, mon, marker=mk, s=10, facecolor="white", edgecolor=S.ORANGE, lw=0.6,
                       label="monitoring (0.01 s vs 0.5 s)" if k == 0 else None)
        for sid, a1, a2 in zip(R.sid, imp, mon):
            rows.append({"panel": "b", "sid": sid, "branch": b, "entry_impl_diff_s": a1, "entry_monitor_diff_s": a2})
    axs[1].axhline(0.1, color=S.INK2, ls="--", lw=0.6)
    axs[1].set_yscale("log")
    axs[1].set_xlabel("benchmark case (marker: reference / CW / SO)")
    axs[1].set_ylabel("final-entry time difference [s]")
    axs[1].set_title("(b) final-entry time")
    axs[1].text(0.99, 0.02, "differences < 1e-9 s plotted at 1e-9 s", transform=axs[1].transAxes, fontsize=5.5,
                ha="right", va="bottom", color=S.INK2)
    axs[1].legend(fontsize=5.6, loc="upper left")
    pk = R.set_index("sid").cw_ref_peak_prod_m
    q = Cv[Cv.family == "radau"].copy()
    q["rtol"] = q.setting.str.replace("radau_", "").astype(float)
    q["peak"] = q.sid.map(pk)
    q["cw_rel"] = q.cw_peak_diff_to_finest_m / q.peak
    g = q.groupby("rtol").apply(lambda d: pd.Series({
        "cw_abs_small": d[d.peak <= 10].cw_peak_diff_to_finest_m.max(), "cw_rel_large": d[d.peak > 10].cw_rel.max(),
        "ref_entry": d.ref_entry_diff_to_finest_s.max()})).reset_index()
    g = g[g.rtol > 1e-12]
    axs[2].plot(g.rtol, np.maximum(g.cw_abs_small, 1e-12), marker="o", color=S.BLUE, label="CW-ref peak, peak <= 10 m [m]")
    axs[2].plot(g.rtol, np.maximum(g.cw_rel_large, 1e-12), marker="s", color=S.ORANGE, ls=":", label="CW-ref peak, peak > 10 m [relative]")
    axs[2].plot(g.rtol, np.maximum(g.ref_entry, 1e-12), marker="D", color=S.VIOLET, ls="--", label="reference entry time [s]")
    for r in g.itertuples():
        rows.append({"panel": "c", "rtol": r.rtol, "max_cw_peak_abs_change_small_m": r.cw_abs_small,
                     "max_cw_peak_rel_change_large": r.cw_rel_large, "max_ref_entry_change_s": r.ref_entry})
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("Radau relative tolerance")
    axs[2].set_ylabel("max change vs rtol 1e-12")
    axs[2].set_title("(c) independent solver convergence")
    axs[2].legend(fontsize=5.6)
    fig.tight_layout()
    S.save(fig, "Fig5", pd.DataFrame(rows))


def fig6():
    B = os.path.join(C.RAW, "boundary_families")
    off = pd.read_csv(os.path.join(B, "offsets.csv"))
    fams = json.load(open(os.path.join(B, "families.json")))["families"]
    exp_path = os.path.join(B, "brackets_exploratory.csv")
    expl = pd.read_csv(exp_path) if os.path.exists(exp_path) else pd.DataFrame()
    order = [f["id"] for f in fams]
    lab = {f["id"]: f"{f['id']} {f['direction_label']}" + ("" if f["velocity_rule"] == "zero" else f" ({f['velocity_rule']})")
           + f" {f['wn_ratio']:g}n/{f['a_max']:g}" for f in fams}
    fig, ax = plt.subplots(figsize=(S.DOUBLE, 2.9))
    rows = []
    for i, fid in enumerate(order):
        q = off[off.family_id == fid]
        if q.empty or (q.status == "no boundary in search range").all():
            ax.plot([i], [0], marker="x", color=S.MUTED, ms=4, lw=0)
            rows.append({"family_id": fid, "status": "no boundary in search range"})
            continue
        for r in q.itertuples():
            if not np.isfinite(getattr(r, "offset_rel", np.nan)):
                rows.append({"family_id": fid, "kind": r.kind, "status": r.status,
                             "cw_lo_m": r.cw_lo_m, "ref_lo_m": r.ref_lo_m})
                continue
            res = r.status == "resolved"
            ax.errorbar([i], [100 * r.offset_rel], yerr=[100 * r.offset_unc_rel], fmt="o", color=S.BLUE,
                        mfc=S.BLUE if res else "white", ms=4, lw=0.8, capsize=1.5)
            rows.append({"family_id": fid, "kind": r.kind, "offset_pct": 100 * r.offset_rel,
                         "offset_unc_pct": 100 * r.offset_unc_rel, "status": r.status})
    # unpaired boundaries: annotated, not plotted as offsets
    i11, i12 = order.index("F11"), order.index("F12")
    note11 = "F11: no CW limit in 0.2-80 km;\nreference limit 77.11 km"
    if not expl.empty and (expl.family_id == "F11").any():
        cwl = float(expl[(expl.family_id == "F11") & (expl.model == "cw")].lo_m.iloc[0])
        note11 += f";\nexploratory: CW limit {cwl / 1e3:.2f} km ({100 * (cwl / 77107.0 - 1):+.1f} %)"
        rows.append({"family_id": "F11", "kind": "arrive->fail", "status": "exploratory (beyond 80 km)",
                     "offset_pct": 100 * (cwl / 77107.0 - 1)})
    ax.plot([i11], [0.0], marker="^", color=S.ORANGE, ms=5, lw=0)
    ax.annotate(note11, xy=(i11, 0.0), xytext=(i11 - 3.5, -0.42), fontsize=5.5, color=S.INK2,
                arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.5))
    ax.plot([i12], [0.82], marker="v", color=S.ORANGE, ms=5, lw=0)
    ax.annotate("F12: also a CW-only arrival island at 39.76-40.28 km (reference fails)", xy=(i12, 0.82),
                xytext=(i12 + 0.5, 0.82), fontsize=5.5, color=S.INK2, va="center")
    ax.axhline(0, color=S.INK2, lw=0.6)
    ax.set_ylim(-0.5, 0.95)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([lab[f] for f in order], rotation=60, ha="right", fontsize=5.6)
    ax.set_ylabel("CW minus reference\ncapture limit [%]")
    fig.text(0.01, 0.005, "> 0: CW predicts arrival over a band where the reference does not (falsely favourable); "
             "< 0: the reference arrives where CW does not (falsely unfavourable).\nCircles: resolved offsets "
             "(bracket uncertainty <= 0.002 %, smaller than the markers). Triangles: boundaries without a partner in 0.2-80 km.",
             fontsize=5.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    S.save(fig, "Fig6", pd.DataFrame(rows))


def fig7():
    F = os.path.join(C.DERIVED, "failure_accounting")
    cen = pd.read_csv(os.path.join(F, "branch_status_census.csv"))
    inv = pd.read_csv(os.path.join(F, "so_failure_investigation.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(S.DOUBLE, 2.4))
    q = cen[cen.campaign == "mechanisms/horizon2"].set_index("model").loc[["cw", "so", "nerm", "eci"]]
    cats = [("completed_arrival", "completed, arrival", "white", ""), ("completed_no_arrival", "completed, no arrival", S.GRAY, ""),
            ("domain_termination", "physical-domain exit", "white", "////"), ("numerical_failure", "numerical failure", S.INK, "")]
    rows = []
    left = np.zeros(len(q))
    for c, lab, fc, hatch in cats:
        v = q[c].to_numpy(float)
        axs[0].barh(range(len(q)), v, left=left, color=fc, edgecolor=S.INK, hatch=hatch, lw=0.6, label=lab, height=0.6)
        for m, vv in zip(q.index, v):
            rows.append({"panel": "a", "model": m, "category": c, "count": int(vv)})
        left += v
    axs[0].set_yticks(range(len(q)))
    axs[0].set_yticklabels(["CW", "second order", "NERM", "inertial ref."])
    axs[0].set_xlabel("two-orbit scenarios")
    axs[0].set_title("(a) branch status, two-orbit campaign (2,160)")
    axs[0].legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    for k, r in enumerate(inv.itertuples()):
        z = np.load(os.path.join(C.RAW, "so_failures", f"{r.sid}.npz"))
        t, rng = z["t"], z["range_m"]
        dt = t[-1] - t
        m = dt > 0
        mk = ["o", "s", "^", "D", "v", "P", "X"][k % 7]
        axs[1].plot(dt[m], rng[m] / 6778137.0, marker=mk, ms=2, lw=0.6, markevery=max(1, m.sum() // 12), label=r.sid)
        for a, b in zip(dt[m][::5], rng[m][::5]):
            rows.append({"panel": "b", "sid": r.sid, "time_to_failure_s": a, "range_over_R": b / 6778137.0})
    xx = np.geomspace(1e-6, 1e3, 20)
    axs[1].plot(xx, 1e2 * (xx / 1e0) ** -2.0, color=S.INK2, ls="--", lw=0.6, label="slope -2")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].invert_xaxis()
    axs[1].axhline(1, color=S.MUTED, lw=0.5, ls=":")
    axs[1].set_xlabel("time before solver failure [s]")
    axs[1].set_ylabel("second-order range / R")
    axs[1].set_title("(b) second-order finite-time blow-up")
    axs[1].legend(fontsize=5, ncol=2)
    fig.tight_layout()
    S.save(fig, "Fig7", pd.DataFrame(rows))


FIGS = {"Fig1": fig1, "Fig2": fig2, "Fig3": fig3, "Fig4": fig4, "Fig5": fig5, "Fig6": fig6, "Fig7": fig7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    which = [w for w in a.only.split(",") if w] or list(FIGS)
    for w in which:
        try:
            FIGS[w]()
            print(f"{w}: written")
        except FileNotFoundError as exc:
            print(f"{w}: SKIPPED (missing input: {exc.filename})")
    C.log_run("figures", "python experiments/v2/v2_figures.py" + (f" --only {a.only}" if a.only else ""), "completed", 0.0, ",".join(which))


if __name__ == "__main__":
    main()
