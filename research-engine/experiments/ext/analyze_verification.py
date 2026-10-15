"""Analyse the numerical verification study and produce its tables and figures.

Uncertainty of a reported quantity at production settings (DOP853, rtol 1e-12,
scaled atol, max_step 20 s, dt_eval 0.5 s) is taken as the largest of:
  formulation : |metric(ECI reference) - metric(NERM)|
  tolerance   : |metric(production) - metric(rtol 1e-13, max_step 5 s, dt_eval 0.25 s)|
  step cap    : |metric(production) - metric(max_step inf)|, |... - metric(max_step 5 s)|
  extraction  : |metric(production) - metric(dt_eval 0.25 s)|
This is an empirical spread, not a rigorous error bound.

Usage: python experiments/ext/analyze_verification.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import report_utils as RU  # noqa: E402

VER = os.path.join(RU.RESULTS, "verification")
PROD = "rtol1e-13"


def main():
    global VER
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""
    if a.tag:
        VER = os.path.join(VER, a.tag)
    RU.setup()
    cases = pd.read_csv(os.path.join(VER, "verification_cases.csv"))
    v = pd.read_csv(os.path.join(VER, "verification_variants.csv"))
    orig = pd.read_csv(os.path.join(VER, "verification_original_engine.csv"))
    meta = cases.set_index("sid")
    rows = []
    for sid, g in v.groupby("sid"):
        g = g.set_index("variant")
        p, ref = g.loc[PROD], g.loc["ref"]
        m = eval(meta.loc[sid, "meta"]) if isinstance(meta.loc[sid, "meta"], str) else {}
        rel0 = eval(meta.loc[sid, "rel0"])
        label = m.get("label") or f"{m.get('axis', 'mixed')} {meta.loc[sid, 'sep_m']:.0f} m"
        if m.get("velocity") not in (None, "zero"):
            label += f" ({m['velocity']})"
        if meta.loc[sid, "wn_ratio"] != 6 or meta.loc[sid, "a_max"] != 0.05:
            label += f" [wn={meta.loc[sid, 'wn_ratio']:g}n, a_max={meta.loc[sid, 'a_max']:g}]"
        pk = "cw_vs_nerm_max_pos_disc_m"
        # |peak(CW-ECI) - peak(CW-NERM)| <= max_t |rho_ECI - rho_NERM| (triangle inequality)
        u_form = p["eci_vs_nerm_max_pos_disc_m"]
        u_tol = abs(p[pk] - ref[pk])
        u_step = max(abs(p[pk] - g.loc["maxstep_inf", pk]), abs(p[pk] - g.loc["maxstep_5", pk]))
        u_ext = abs(p[pk] - g.loc["dteval0.25", pk])
        u = max(u_form, u_tol, u_step, u_ext)
        dop = [k for k in g.index if k.startswith("rtol") or k.startswith("noevents") or k.startswith("maxstep") or k.startswith("dteval") or k == "ref"]
        tight = [k for k in dop if k not in ("rtol1e-09", "noevents_rtol1e-12")]
        cls_all = g.loc[dop, "outcome_terminal_cw"].nunique() == 1
        cls_tight = g.loc[tight, "outcome_terminal_cw"].nunique() == 1
        cls_rk4 = set(g.loc[[k for k in g.index if k.startswith("rk4")], "outcome_terminal_cw"])
        arr = {}
        for mdl in ("cw", "nerm", "eci"):
            a_p, a_r = p[f"{mdl}_terminal_arrival_time_s"], ref[f"{mdl}_terminal_arrival_time_s"]
            arr[mdl] = abs(a_p - a_r) if np.isfinite(a_p) and np.isfinite(a_r) else np.nan
        o = orig[(orig.sid == sid) & (orig.samples == 2001)]
        h = np.load(os.path.join(VER, "histories", f"{sid}.npz"))
        nl_end = float(h["rtol1e-13__nerm_t"][-1])
        rows.append({
            "sid": sid, "case": label, "sep_m": meta.loc[sid, "sep_m"],
            "x0": rel0[0], "y0": rel0[1], "z0": rel0[2], "v0_norm": float(np.linalg.norm(rel0[3:])),
            "wn_ratio": meta.loc[sid, "wn_ratio"], "a_max": meta.loc[sid, "a_max"],
            "cw_sat_s": p["cw_sat_total_s"], "nl_sat_s": p["eci_sat_total_s"],
            "peak_disc_m": p[pk], "peak_disc_legacy_grid_m": p["cw_vs_nerm_max_pos_disc_legacy_grid_m"],
            "nl_branch_end_s": nl_end, "domain_truncated": bool(nl_end < 5553.6),
            "u_formulation_m": u_form, "u_tolerance_m": u_tol, "u_stepcap_m": u_step, "u_extraction_m": u_ext,
            "u_total_m": u, "u_rel": u / p[pk] if p[pk] > 0 else np.nan,
            "eci_vs_nerm_traj_max_m": p["eci_vs_nerm_hist_max_dpos_m"],
            "eci_vs_nerm_vel_max_mps": p["eci_vs_nerm_hist_max_dvel_mps"],
            "eci_vs_nerm_cmd_max_mps2": p["eci_vs_nerm_hist_max_dcmd_mps2"],
            "nerm_vs_ref_traj_max_m": p.get("nerm_vs_ref_max_dpos_m", np.nan),
            "eci_vs_ref_traj_max_m": p.get("eci_vs_ref_max_dpos_m", np.nan),
            "cw_vs_ref_traj_max_m": p.get("cw_vs_ref_max_dpos_m", np.nan),
            "arr_time_diff_cw_s": arr["cw"], "arr_time_diff_nerm_s": arr["nerm"], "arr_time_diff_eci_s": arr["eci"],
            "arr_time_nerm_vs_eci_s": abs(p["nerm_terminal_arrival_time_s"] - p["eci_terminal_arrival_time_s"])
            if np.isfinite(p["nerm_terminal_arrival_time_s"]) and np.isfinite(p["eci_terminal_arrival_time_s"]) else np.nan,
            "cw_terminal": p["cw_terminal_arrived"], "nl_terminal": p["eci_terminal_arrived"],
            "nerm_terminal": p["nerm_terminal_arrived"], "cw_legacy": p["cw_legacy_arrived"], "nl_legacy": p["eci_legacy_arrived"],
            "outcome_terminal": p["outcome_terminal_cw"], "outcome_legacy": p["outcome_legacy_cw"],
            "class_stable_all_dop853": cls_all, "class_stable_rtol_le_1e-10": cls_tight,
            "rk4_outcomes": ";".join(sorted(cls_rk4)),
            "rk4_T2000_peak_err_m": abs(g.loc["rk4_T/2000", pk] - ref[pk]) if nl_end >= 5553.6 else np.nan,
            "rk4_T2000_traj_err_nerm_m": g.loc["rk4_T/2000"].get("nerm_vs_ref_max_dpos_m", np.nan),
            "rk4_T8000_traj_err_nerm_m": g.loc["rk4_T/8000"].get("nerm_vs_ref_max_dpos_m", np.nan),
            "orig_engine_2001_peak_m": float(o["orig_max_traj_diff_m"].iloc[0]) if len(o) else np.nan,
            "orig_engine_2001_peak_err_m": (abs(float(o["orig_max_traj_diff_m"].iloc[0]) - ref[pk]) if len(o) and nl_end >= 5553.6 else np.nan),
            "dv_nerm_vs_eci_mps": abs(p["nerm_dv_aug_mps"] - p["eci_dv_aug_mps"]),
            "dv_aug_vs_simpson_mps": abs(p["nerm_dv_aug_mps"] - p["nerm_dv_simpson_mps"]),
            "ambiguity_flags": p["ambiguity_flags"] if isinstance(p["ambiguity_flags"], str) else "",
            "nfev_cw": p["cw_nfev"], "nfev_nerm": p["nerm_nfev"], "nfev_eci": p["eci_nfev"],
        })
    S = pd.DataFrame(rows).sort_values(["sep_m"])
    S.to_csv(os.path.join(VER, "verification_summary.csv"), index=False)

    # paper-style table
    T = S[["case", "peak_disc_m", "u_total_m", "eci_vs_nerm_traj_max_m", "eci_vs_nerm_cmd_max_mps2",
           "arr_time_nerm_vs_eci_s", "outcome_terminal", "class_stable_rtol_le_1e-10", "rk4_T2000_peak_err_m"]].copy()
    T.columns = ["case", "peak CW-ref diff [m]", "uncertainty [m]", "ECI vs NERM traj [m]",
                 "ECI vs NERM cmd [m/s^2]", "ECI vs NERM arrival [s]", "terminal outcome",
                 "class stable", "RK4 T/2000 peak err [m]"]
    RU.write_table(T, "tab_verification" + sfx, "Verification cases: peak CW-to-reference difference at production "
                   "settings, empirical numerical uncertainty, agreement of the two nonlinear formulations, and "
                   "error of fixed-step RK4 at the manuscript sweep step. Circular 400 km chief, one orbit.",
                   "tab:verification",
                   fmt={"peak CW-ref diff [m]": "{:.4g}", "uncertainty [m]": "{:.1e}", "ECI vs NERM traj [m]": "{:.1e}",
                        "ECI vs NERM cmd [m/s^2]": "{:.1e}", "ECI vs NERM arrival [s]": "{:.1e}",
                        "RK4 T/2000 peak err [m]": "{:.1e}"})

    # convergence figure: trajectory error vs rtol (NERM, ECI) and RK4 vs dt
    import matplotlib.pyplot as plt
    pick = S.sort_values("sep_m")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    rt = [1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
    data = []
    sat = pick[pick.cw_sat_s > 0]
    uns = pick[pick.cw_sat_s == 0]
    for grp, col, mk, lab in ((uns, RU.BLUE, "o", "unsaturated cases"), (sat, RU.ORANGE, "s", "saturated cases")):
        errs = []
        for r in rt:
            e = v[(v.variant == f"rtol{r:.0e}") & (v.sid.isin(grp.sid))]["nerm_vs_ref_max_dpos_m"].max()
            e2 = v[(v.variant == f"rtol{r:.0e}") & (v.sid.isin(grp.sid))]["eci_vs_ref_max_dpos_m"].max()
            errs.append((e, e2))
            data.append({"panel": "a", "group": lab, "rtol": r, "nerm_worst_traj_err_m": e, "eci_worst_traj_err_m": e2})
        axs[0].loglog(rt, [max(e[0], 1e-12) for e in errs], marker=mk, color=col, ls="-", label=f"NERM, {lab}")
        axs[0].loglog(rt, [max(e[1], 1e-12) for e in errs], marker=mk, color=col, ls="--", mfc="white", label=f"inertial, {lab}")
    axs[0].axhline(1e-3, color=RU.MUTED, lw=0.8)
    axs[0].text(1.2e-8, 1.3e-3, "1 mm target", color=RU.INK2, fontsize=7)
    axs[0].set_xlabel("DOP853 relative tolerance")
    axs[0].set_ylabel("worst max position error vs tight reference [m]")
    axs[0].set_title("(a) adaptive DOP853")
    axs[0].legend(loc="upper left", fontsize=6.5)
    ks = [1000, 2000, 4000, 8000]
    for grp, col, mk, lab in ((uns, RU.BLUE, "o", "unsaturated"), (sat, RU.ORANGE, "s", "saturated")):
        e = [v[(v.variant == f"rk4_T/{k}") & (v.sid.isin(grp.sid))]["nerm_vs_ref_max_dpos_m"].max() for k in ks]
        dts = [5553.62 / k for k in ks]
        axs[1].loglog(dts, e, marker=mk, color=col, label=f"RK4, {lab} cases (worst)")
        for d_, e_ in zip(dts, e):
            data.append({"panel": "b", "group": lab, "rk4_dt_s": d_, "nerm_worst_traj_err_m": e_})
    d0 = np.array([5.55, 0.694])
    e_un = v[(v.variant == "rk4_T/1000") & (v.sid.isin(uns.sid))]["nerm_vs_ref_max_dpos_m"].max()
    axs[1].loglog(d0, e_un * (d0 / d0[0]) ** 4, color=RU.MUTED, lw=0.8, ls=":", label="slope 4 reference")
    axs[1].axvline(5553.62 / 2000, color=RU.MUTED, lw=0.8)
    axs[1].text(5553.62 / 2000 * 1.05, axs[1].get_ylim()[0] * 3, "manuscript\nsweep step", fontsize=6.5, color=RU.INK2)
    axs[1].set_xlabel("fixed RK4 step [s]")
    axs[1].set_ylabel("worst max position error vs reference [m]")
    axs[1].set_title("(b) fixed-step RK4 (original method)")
    axs[1].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    RU.save(fig, "fig_verification_convergence" + sfx, pd.DataFrame(data))
    print(S[["case", "peak_disc_m", "u_total_m", "eci_vs_nerm_traj_max_m", "outcome_terminal",
             "class_stable_rtol_le_1e-10", "rk4_T2000_peak_err_m", "orig_engine_2001_peak_err_m"]].to_string())
    print("max uncertainty", S.u_total_m.max(), "max ECI-vs-NERM traj", S.eci_vs_nerm_traj_max_m.max())


if __name__ == "__main__":
    main()
