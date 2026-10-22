import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import reportutil as RU

from cwvalid.runner import loadres

AXES = ["+R", "-R", "+V", "-V", "+H", "-H"]
REF = "eci"
OUTCOMES = ["both_succeed", "cw_only", "nl_only", "both_fail"]


def load(gdir, rdir):
    df = loadres(gdir)
    df["src"] = "grid"
    if rdir and os.path.exists(os.path.join(rdir, "results.jsonl")):
        r = loadres(rdir)
        r["src"] = "refine"
        df = pd.concat([df, r], ignore_index=True)
    return df


def outctables(df):
    g = df[df.src == "grid"]
    rows = []
    for (w, a), s in g.groupby(["wn_ratio", "a_max"]):
        r = {"wn/n": w, "a_max": a, "N": len(s)}
        for o in OUTCOMES:
            r[o] = int((s.outcome_terminal_cw == o).sum())
        r["ambiguous"] = int(s.ambiguous.sum())
        for o in ("cw_only", "nl_only"):
            r[f"legacy_{o}"] = int((s.outcome_legacy_cw == o).sum())
        r["disc>0.1 m"] = int((s[f"cw_vs_{REF}_max_pos_disc_m"] > 0.1).sum())
        r["disc>1 m"] = int((s[f"cw_vs_{REF}_max_pos_disc_m"] > 1.0).sum())
        rows.append(r)
    t = pd.DataFrame(rows)
    tot = {"wn/n": "all", "a_max": "all", **{c: int(t[c].sum()) for c in t.columns if c not in ("wn/n", "a_max")}}
    t = pd.concat([t, pd.DataFrame([tot])], ignore_index=True)
    RU.writetable(t, "outcomecounts", "Primary terminal-arrival outcome classes of the 2,160 main-grid paired "
                  "scenarios (CW vs inertial nonlinear reference, each with its own PD loop), by controller setting; "
                  "legacy columns use the manuscript position-only test; 'disc' counts peak CW-to-reference "
                  "differences above 0.1 m and 1 m. Zero initial LVLH velocity, one orbit.", "tab:outcomes")
    rows = []
    for ax, s in g.groupby("meta_axis"):
        r = {"direction": RU.AXISLABEL[ax], "N": len(s)}
        for o in OUTCOMES:
            r[o] = int((s.outcome_terminal_cw == o).sum())
        r["legacy cw_only"] = int((s.outcome_legacy_cw == "cw_only").sum())
        r["legacy nl_only"] = int((s.outcome_legacy_cw == "nl_only").sum())
        rows.append(r)
    RU.writetable(pd.DataFrame(rows), "outcomedirection",
                  "Main-grid terminal-arrival outcome classes by signed initial offset direction (all nine controller settings pooled).",
                  "tab:outcomes_dir")
    return t


def boundtable(df, tol):
    from cwvalid.refine import findbrack, predic
    br = findbrack(df, predic(REF, (0.1, 1.0)), tol)
    if br.empty:
        return br
    rows = []
    for (ax, w, a), s in br.groupby(["meta_axis", "wn_ratio", "a_max"]):
        r = {"direction": RU.AXISLABEL[ax], "wn/n": w, "a_max": a}
        for p in ("disc_gt_0.1m", "disc_gt_1m", "cw_terminal", "nl_terminal", "nl_legacy", "terminal_disagree"):
            q = s[s.predicate == p].sort_values("sep_lo")
            if q.empty:
                r[p] = "none in range"
            else:
                parts = [f"{x.sep_lo/1e3:.3f}-{x.sep_hi/1e3:.3f}" + ("" if x.resolved else "*") for x in q.itertuples()]
                r[p] = "; ".join(parts)
        rows.append(r)
    t = pd.DataFrame(rows)
    RU.writetable(t, "boundaries", "Refined separation brackets [km] at which each predicate changes value "
                  "(all changes listed, ordered by separation; '*' marks a bracket wider than the refinement "
                  f"tolerance of {tol*100:.1f}% relative width). disc_gt: peak CW-to-reference difference above "
                  "the threshold; cw/nl_terminal: arrival under the primary terminal criterion; nl_legacy: "
                  "manuscript position-only test for the reference; terminal_disagree: CW and reference "
                  "terminal labels differ.", "tab:boundaries")
    br.to_csv(os.path.join(RU.TABDIR, "bracketsall.csv"), index=False)
    return t


def figbasecurve(df):
    import matplotlib.pyplot as plt
    s = df[(df.wn_ratio == 6) & (df.a_max == 0.05)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True)
    data = []
    for ax in AXES:
        q = s[s.meta_axis == ax].sort_values("sep_m")
        c, ls, mk = RU.axisstyle(ax)
        axs[0].loglog(q.sep_m, q[f"cw_vs_{REF}_max_pos_disc_m"], color=c, ls=ls, marker=mk, ms=3, lw=1.2, label=RU.AXISLABEL[ax])
        axs[1].loglog(q.sep_m, q[f"so_vs_{REF}_max_pos_disc_m"].clip(lower=1e-12), color=c, ls=ls, marker=mk, ms=3, lw=1.2)
        bad = q[q.outcome_terminal_cw.isin(["cw_only", "nl_only"])]
        axs[0].scatter(bad.sep_m, bad[f"cw_vs_{REF}_max_pos_disc_m"], s=36, facecolors="none", edgecolors=RU.INK, lw=0.9, zorder=5)
        for r in q.itertuples():
            data.append({"axis": ax, "sep_m": r.sep_m, "cw_peak_disc_m": getattr(r, f"cw_vs_{REF}_max_pos_disc_m"),
                         "so_peak_disc_m": getattr(r, f"so_vs_{REF}_max_pos_disc_m"), "outcome": r.outcome_terminal_cw})
    sd = s.sat_distance_m.iloc[0]
    for a in axs:
        a.axhline(1.0, color=RU.MUTED, lw=0.8)
        a.axhline(0.1, color=RU.MUTED, lw=0.8, ls=(0, (1, 2)))
        a.axvline(sd, color=RU.MUTED, lw=0.8)
        a.set_xlabel("initial separation [m]")
    axs[0].text(sd * 1.08, 3e4, "a$_{max}$/K$_p$", fontsize=7, color=RU.INK2)
    axs[0].text(60, 1.25, "1 m", fontsize=7, color=RU.INK2)
    axs[0].text(60, 0.125, "0.1 m", fontsize=7, color=RU.INK2)
    axs[0].set_ylabel("peak position difference to reference [m]")
    axs[0].set_title("(a) CW vs nonlinear reference")
    axs[1].set_title("(b) second-order vs nonlinear reference")
    if s.outcome_terminal_cw.isin(["cw_only", "nl_only"]).any():
        axs[0].scatter([], [], s=36, facecolors="none", edgecolors=RU.INK, lw=0.9, label="terminal outcomes differ")
    axs[0].legend(fontsize=6.5, loc="upper left", ncol=1)
    fig.tight_layout()
    RU.save(fig, "discsep", pd.DataFrame(data))


def gridmat(s, col):
    seps = np.sort(s.sep_m.unique())
    M = np.full((len(AXES), len(seps)), np.nan, dtype=object)
    for i, ax in enumerate(AXES):
        q = s[s.meta_axis == ax].set_index("sep_m")
        for j, sp in enumerate(seps):
            if sp in q.index:
                M[i, j] = q.loc[sp, col]
    return seps, M


def figmaps(df):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    g = df[df.src == "grid"]
    W, A = sorted(g.wn_ratio.unique()), sorted(g.a_max.unique())
    fig, axs = plt.subplots(len(A), len(W), figsize=(7.4, 5.6), sharex=True, sharey=True)
    data = []
    edges = [-12, -3, -2, -1, 0, 1, 2, 12]
    cmapb = ListedColormap(["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    normb = BoundaryNorm(edges, cmapb.N)
    for i, a in enumerate(A):
        for j, w in enumerate(W):
            s = g[(g.wn_ratio == w) & (g.a_max == a)]
            seps, M = gridmat(s, f"cw_vs_{REF}_max_pos_disc_m")
            Z = np.log10(M.astype(float))
            ax = axs[i, j]
            im = ax.pcolormesh(np.arange(len(seps) + 1), np.arange(len(AXES) + 1), Z, cmap=cmapb, norm=normb,
                               shading="flat")
            ax.contour(np.arange(len(seps)) + 0.5, np.arange(len(AXES)) + 0.5, Z, levels=[-1, 0], colors=[RU.ORANGE, RU.INK], linewidths=[0.9, 1.1], linestyles=["--", "-"])
            ax.set_title(f"$\\omega_n$={w:g}n, $a_{{max}}$={a:g}", fontsize=7.5)
            ax.set_yticks(np.arange(len(AXES)) + 0.5)
            ax.set_yticklabels([RU.AXISLABEL[x] for x in AXES], fontsize=6)
            ticks = [0, 13, 26, 39]
            ax.set_xticks(np.array(ticks) + 0.5)
            ax.set_xticklabels([f"{seps[k]/1e3:.2g}" for k in ticks], fontsize=6)
            ax.grid(False)
            for ii, axn in enumerate(AXES):
                for jj, sp in enumerate(seps):
                    data.append({"wn_ratio": w, "a_max": a, "axis": axn, "sep_m": sp, "log10_peak_disc": Z[ii, jj]})
    for ax in axs[-1, :]:
        ax.set_xlabel("separation [km]")
    cb = fig.colorbar(im, ax=axs, shrink=0.6, label="peak CW-to-reference difference [m]")
    cb.set_ticks([-7.5, -2.5, -1.5, -0.5, 0.5, 1.5, 7.0])
    cb.set_ticklabels(["< 1e-3", "1e-3 to 1e-2", "1e-2 to 0.1", "0.1 to 1", "1 to 10", "10 to 100", "> 100"])
    fig.text(0.01, 0.005, "contours: 0.1 m (dashed), 1 m (solid)", fontsize=6.5, color=RU.INK2)
    RU.save(fig, "agreementmap", pd.DataFrame(data))
    fig, axs = plt.subplots(len(A), len(W), figsize=(7.4, 5.6), sharex=True, sharey=True)
    code = {o: k for k, o in enumerate(OUTCOMES)}
    cmap = ListedColormap([RU.OUTCCOLOR[o] for o in OUTCOMES])
    norm = BoundaryNorm(np.arange(-0.5, len(OUTCOMES)), cmap.N)
    data = []
    for i, a in enumerate(A):
        for j, w in enumerate(W):
            s = g[(g.wn_ratio == w) & (g.a_max == a)]
            seps, M = gridmat(s, "outcome_terminal_cw")
            Z = np.vectorize(lambda o: code.get(o, np.nan))(M).astype(float)
            ax = axs[i, j]
            ax.pcolormesh(np.arange(len(seps) + 1), np.arange(len(AXES) + 1), Z, cmap=cmap, norm=norm,
                          shading="flat")
            for yy_ in range(1, len(AXES)):
                ax.axhline(yy_, color="white", lw=1.0)
            _, Amb = gridmat(s, "ambiguous")
            Amb = np.vectorize(lambda v: (v is True) or (isinstance(v, (bool, np.bool_)) and bool(v)))(Amb)
            yy, xx = np.where(Amb)
            ax.scatter(xx + 0.5, yy + 0.5, marker="x", s=8, color=RU.INK, lw=0.6)
            ax.set_title(f"$\\omega_n$={w:g}n, $a_{{max}}$={a:g}", fontsize=7.5)
            ax.set_yticks(np.arange(len(AXES)) + 0.5)
            ax.set_yticklabels([RU.AXISLABEL[x] for x in AXES], fontsize=6)
            ticks = [0, 13, 26, 39]
            ax.set_xticks(np.array(ticks) + 0.5)
            ax.set_xticklabels([f"{seps[k]/1e3:.2g}" for k in ticks], fontsize=6)
            ax.grid(False)
            for ii, axn in enumerate(AXES):
                for jj, sp in enumerate(seps):
                    data.append({"wn_ratio": w, "a_max": a, "axis": axn, "sep_m": sp, "outcome": M[ii, jj],
                                 "ambiguous": bool(Amb[ii, jj])})
    for ax in axs[-1, :]:
        ax.set_xlabel("separation [km]")
    from matplotlib.patches import Patch
    handles = [Patch(color=RU.OUTCCOLOR[o], label=RU.OUTCLABEL[o]) for o in OUTCOMES]
    handles.append(plt.Line2D([], [], marker="x", ls="", color=RU.INK, label="numerically ambiguous"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    RU.save(fig, "outcomemap", pd.DataFrame(data))


def modelcmp(df):
    g = df[df.src == "grid"].copy()
    rows = []
    for (w, a), s in g.groupby(["wn_ratio", "a_max"]):
        full = s[s[f"{REF}_termination"] == "horizon"]
        r = {"wn/n": w, "a_max": a, "N": len(s), "N dv (complete horizon)": len(full)}
        for m in ("cw", "so"):
            disc = s[f"{m}_vs_{REF}_max_pos_disc_m"]
            r[f"{m} median disc [m]"] = float(disc.median())
            r[f"{m} disc>1 m"] = int((disc > 1).sum())
            r[f"{m} outcome agrees"] = int((s[f"{m}_terminal_arrived"] == s[f"{REF}_terminal_arrived"]).sum())
            dv = (full[f"{m}_dv_aug_mps"] - full[f"{REF}_dv_aug_mps"]).abs() / full[f"{REF}_dv_aug_mps"]
            r[f"{m} median |dv err|"] = float(dv.median())
            r[f"{m} max |dv err|"] = float(dv.max())
        rows.append(r)
    t = pd.DataFrame(rows)
    RU.writetable(t, "modelcomparison", "CW and second-order (SO) closed-loop predictions against the inertial "
                  "nonlinear reference on the main grid (each model with its own PD loop): peak position difference, "
                  "agreement of the primary terminal-arrival label, and relative Delta-v error (Delta-v compared only "
                  "for scenarios whose reference branch completed the horizon; reference branches that left the "
                  "physical domain are excluded from the Delta-v columns).", "tab:modelcmp",
                  fmt={c: "{:.3g}" for c in t.columns if "disc [m]" in c or "dv err" in c})
    return t


def modeloutcomesrefined():
    s = {"one orbit, grid + refinement + band points": [os.path.join(RU.RESULTS, "maingrid", s) for s in ("grid", "refine", "refinebands")],
         "two orbits, main grid": [os.path.join(RU.RESULTS, "mechanisms", "horizon2")]}
    rows = []
    for name, paths in s.items():
        parts = [loadres(p) for p in paths if os.path.exists(os.path.join(p, "results.jsonl"))]
        if not parts:
            continue
        d = pd.concat(parts, ignore_index=True).drop_duplicates(["meta_axis", "wn_ratio", "a_max", "sep_m", "horizon_orbits"])
        for m, lab in (("cw", "CW"), ("so", "second order"), ("nerm", "NERM (exact, cross-check)")):
            col = f"outcome_terminal_{m}"
            if col not in d:
                continue
            rows.append({"data": name, "model": lab, "N": len(d),
                         "label agrees": int(d[col].isin(["both_succeed", "both_fail"]).sum()),
                         "model arrives, reference not": int((d[col] == "cw_only").sum()),
                         "reference arrives, model not": int((d[col] == "nl_only").sum())})
    t = pd.DataFrame(rows)
    RU.writetable(t, "outcomesrefined", "Terminal-arrival label agreement of each model with the inertial "
                  "nonlinear reference (each model with its own PD loop). The one-orbit set is concentrated at threshold "
                  "and capture-limit brackets by construction, so its disagreement rate is not representative of the "
                  "grid as a whole.", "tab:model_outcomes")
    return t


def figsatlink(df):
    import matplotlib.pyplot as plt
    g = df[df.src == "grid"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    data = []
    for o in OUTCOMES:
        s = g[g.outcome_terminal_cw == o]
        axs[0].scatter(s.cw_sat_total_s / s.horizon_s, s[f"cw_vs_{REF}_max_pos_disc_m"], s=6,
                       color=RU.OUTCCOLOR[o], label=RU.OUTCLABEL[o], alpha=0.85, lw=0)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("CW-predicted saturated fraction of horizon")
    axs[0].set_ylabel("peak CW-to-reference difference [m]")
    axs[0].axhline(1, color=RU.MUTED, lw=0.8)
    axs[0].set_title("(a) saturation vs trajectory difference")
    axs[0].legend(fontsize=6, loc="lower right", markerscale=2)
    x = g["cw_q2_impulse_sat_mps"]
    y = g[f"cw_vs_{REF}_max_pos_disc_m"]
    ok = x > 0
    axs[1].loglog(x[ok], y[ok], ".", ms=2.5, color=RU.BLUE)
    axs[1].set_xlabel("CW-trajectory 2nd-order residual impulse while saturated [m/s]")
    axs[1].set_ylabel("peak CW-to-reference difference [m]")
    axs[1].set_title("(b) CW-only residual indicator")
    for r in g.itertuples():
        data.append({"sid": r.sid, "sat_frac": r.cw_sat_total_s / r.horizon_s, "q2_sat": r.cw_q2_impulse_sat_mps,
                     "peak_disc_m": getattr(r, f"cw_vs_{REF}_max_pos_disc_m"), "outcome": r.outcome_terminal_cw})
    fig.tight_layout()
    RU.save(fig, "saturationlink", pd.DataFrame(data))
    rho = pd.Series(np.log10(y[ok])).corr(pd.Series(np.log10(x[ok])), method="spearman")
    return rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(RU.RESULTS, "maingrid", "grid"))
    ap.add_argument("--refine", default=os.path.join(RU.RESULTS, "maingrid", "refine"))
    ap.add_argument("--tol", type=float, default=0.005)
    a = ap.parse_args()
    RU.setup()
    df = load(a.dir, a.refine)
    t = outctables(df)
    print(t.to_string())
    b = boundtable(df, a.tol)
    if len(b):
        print(b.to_string())
    figbasecurve(df)
    figmaps(df)
    print(modelcmp(df).to_string())
    print(modeloutcomesrefined().to_string())
    print("spearman(log disc, log q2_sat) =", figsatlink(df))


if __name__ == "__main__":
    main()
