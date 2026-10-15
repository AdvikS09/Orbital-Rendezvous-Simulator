"""Build the v2 publication tables (CSV + LaTeX + column definitions) from saved v2 outputs.

Tables (results/v2/derived/tables/):
  T1_estimator_frozen_rules        frozen v2 rules, thresholds, calibration status, development checks
  T2_confirmatory_performance      fresh 300-case set: every primary rule
  T3_retrospective_performance     broad test, stress and band challenge: same rules (retrospective)
  T4_estimate_ratio_by_regime      estimate / measured peak by regime, estimator and set
  T5_acceptance_target_rules       acceptance-target rules (descriptive): development target vs fresh result
  T6_arrival_condition_contribution what the arrival + dwell-margin conditions change relative to D1 alone
  T7_verifier_benchmark_cases      the 24 preselected verification cases and the production/independent comparison
  T8_verifier_convergence          independent solver ladder: worst change relative to the finest setting
  T9_boundary_families             24 predeclared families: brackets, offsets, status, frozen-rule results
  T10_model_outcomes_status_aware  corrected label-agreement table (and the superseded v1 version)
  T11_branch_status_census         completed/arrival, completed/no arrival, domain exit, numerical failure per campaign
  T12_so_failure_investigation     representative second-order failures
Usage: python experiments/v2/v2_make_tables.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_common as C  # noqa: E402
from v2_tables import write_table  # noqa: E402

EV = os.path.join(C.DERIVED, "estimator")
RULE_LABEL = {"traj1_d1": "A full D1, peak<=1 m", "traj1_d1u": "B unsat. Jacobian, peak<=1 m",
              "traj1_cres": "C residual only, peak<=1 m", "traj1_sep": "separation, peak<=1 m",
              "traj1_sat": "saturated fraction, peak<=1 m", "comb_d1": "combined with A", "comb_d1u": "combined with B",
              "comb_cres": "combined with C", "v1_sep_only": "v1 separation baseline", "v1_sat_only": "v1 saturation baseline",
              "traj01_d1": "A, peak<=0.1 m", "traj01_d1u": "B, peak<=0.1 m", "traj01_cres": "C, peak<=0.1 m",
              "traj01_sep": "separation, peak<=0.1 m", "traj01_sat": "saturated fraction, peak<=0.1 m"}
PRIMARY = ["traj1_d1", "traj1_d1u", "traj1_cres", "traj1_sep", "traj1_sat", "comb_d1", "comb_d1u", "comb_cres",
           "v1_sep_only", "v1_sat_only", "traj01_d1", "traj01_d1u", "traj01_cres", "traj01_sep"]
PERF_TEX = ["set", "description", "threshold", "accepted", "accept_pct", "accept_ci95", "incorrect", "incorrect_ci95_upper",
            "unnecessary_referrals", "unnecessary_pct", "ff_accepted", "fu_accepted"]
PERF_COLS = {
    "set": ("-", "evaluation set"), "status": ("-", "confirmatory / retrospective / in-sample"),
    "rule": ("-", "frozen rule identifier"), "description": ("-", "rule in words"),
    "target": ("-", "what an acceptance asserts: traj_ok_1 = peak CW-to-reference difference <= 1 m; traj_ok_01 <= 0.1 m; adequate = traj_ok_1 and same terminal-arrival label"),
    "threshold": ("m or -", "frozen threshold (m for D1/B/C/separation; dimensionless for saturated fraction)"),
    "evaluated": ("count", "scenarios with a valid reference label (denominator)"),
    "accepted": ("count", "scenarios accepted"), "accept_pct": ("%", "accepted / evaluated"),
    "accept_ci95": ("%", "Clopper-Pearson 95% interval of accept_pct"),
    "incorrect": ("count", "accepted scenarios violating the rule's target"),
    "incorrect_ci95_upper": ("%", "upper 95% Clopper-Pearson bound of incorrect / accepted (undefined if none accepted)"),
    "unnecessary_referrals": ("count", "referred scenarios that satisfied the target"),
    "unnecessary_pct": ("%", "unnecessary referrals / referred"),
    "traj_viol_accepted": ("count", "accepted scenarios with peak difference > 1 m"),
    "ff_accepted": ("count", "accepted falsely favourable arrival predictions (CW arrives, reference does not)"),
    "fu_accepted": ("count", "accepted falsely unfavourable arrival predictions (reference arrives, CW does not)"),
    "ff_in_set": ("count", "falsely favourable predictions in the set"), "fu_in_set": ("count", "falsely unfavourable predictions in the set"),
    "feature_failures": ("count", "non-finite CW-only feature (always referred)"),
    "ref_domain_exits": ("count", "reference branches ending at the 100 km altitude floor (physical non-arrival)"),
    "ref_numerical_failures": ("count", "reference branches with a solver failure (label invalid)"),
}


def perf_table(P, sets, status):
    rows = []
    for st in sets:
        for rule in PRIMARY:
            r = P[(P.set == st) & (P.rule == rule)]
            if r.empty:
                continue
            r = r.iloc[0]
            if r.rule_status != "ok":
                rows.append({"set": st, "status": status, "rule": rule, "description": RULE_LABEL.get(rule, rule),
                             "target": r.target, "threshold": np.nan})
                continue
            rows.append({"set": st, "status": status, "rule": rule, "description": RULE_LABEL.get(rule, rule), "target": r.target,
                         "threshold": r.threshold, "evaluated": int(r.n_evaluated), "accepted": int(r.accepted),
                         "accept_pct": 100 * r.accept_rate, "accept_ci95": f"{100 * r.accept_rate_lo:.1f}-{100 * r.accept_rate_hi:.1f}",
                         "incorrect": int(r.incorrect_accepted),
                         "incorrect_ci95_upper": 100 * r.incorrect_rate_hi if np.isfinite(r.incorrect_rate_hi) else np.nan,
                         "unnecessary_referrals": int(r.unnecessary_referrals),
                         "unnecessary_pct": 100 * r.unnecessary_rate if np.isfinite(r.unnecessary_rate) else np.nan,
                         "traj_viol_accepted": int(r.traj_violations_accepted), "ff_accepted": int(r.ff_accepted),
                         "fu_accepted": int(r.fu_accepted), "ff_in_set": int(r.ff_total), "fu_in_set": int(r.fu_total),
                         "feature_failures": int(r.feature_nonfinite), "ref_domain_exits": int(r.ref_domain_exits),
                         "ref_numerical_failures": int(r.ref_numerical_failures)})
    return pd.DataFrame(rows)


def t_estimator():
    fr = json.load(open(C.FROZEN_V2))
    rows = []
    for k, r in fr["rules"].items():
        chk = fr["dev_fit_checks"][k]
        rows.append({"rule": k, "feature": r["feature"], "target": r["target"], "threshold": r["threshold"],
                     "calibration": r["status"][:40], "combined": bool(r.get("combined", False)),
                     "descriptive": bool(r.get("descriptive", False)), "frozen_v1": bool(r.get("frozen_v1", False)),
                     "dev_accepted": chk["dev_accepted"] if isinstance(chk, dict) else np.nan,
                     "dev_violations_accepted": chk["dev_violations_accepted"] if isinstance(chk, dict) else np.nan})
    write_table(pd.DataFrame(rows), "T1_estimator_frozen_rules",
                f"Frozen v2 estimator-comparison rules (fitted on the 3,839 development scenarios; freeze file SHA-256 "
                f"{C.sha256(C.FROZEN_V2)[:16]}..., code commit {fr['code']['commit'][:7]}).",
                {"rule": ("-", "rule identifier (traj1/traj01: trajectory-only; comb: CW arrival + dwell margin >= 380.56 s + feature threshold; acc25/50/75: descriptive acceptance-target rules; v1_: frozen v1 baselines)"),
                 "feature": ("-", "d1 = A full D1; d1u = B unsaturated-Jacobian D1; cres = C residual-only diagnostic; sep = initial separation; sat = CW saturated-time fraction"),
                 "target": ("-", "target the acceptance asserts"), "threshold": ("m or -", "accept iff feature <= threshold"),
                 "calibration": ("-", "ok, or why no threshold could be calibrated"), "combined": ("-", "rule includes the CW arrival and dwell-margin conditions"),
                 "descriptive": ("-", "secondary descriptive rule (acceptance target)"), "frozen_v1": ("-", "threshold copied unchanged from the v1 freeze"),
                 "dev_accepted": ("count", "development scenarios accepted (of 3,839)"), "dev_violations_accepted": ("count", "accepted development scenarios violating the target")})
    P = pd.read_csv(os.path.join(EV, "performance_long.csv"))
    write_table(perf_table(P, ["fresh"], "confirmatory"), "T2_confirmatory_performance",
                "Confirmatory evaluation on the fresh 300-scenario set (seed 12001, broad-test distribution, generated and labelled after the freeze).",
                PERF_COLS, fmt={"accept_pct": "{:.1f}", "incorrect_ci95_upper": "{:.1f}", "unnecessary_pct": "{:.1f}", "threshold": "{:.4g}"},
                tex_columns=PERF_TEX)
    write_table(perf_table(P, ["test", "stress", "band_test"], "retrospective"), "T3_retrospective_performance",
                "Retrospective evaluation of the frozen v2 rules on the previously examined broad test (400), stress (150) and band challenge (69) sets. These sets are not untouched confirmatory tests.",
                PERF_COLS, fmt={"accept_pct": "{:.1f}", "incorrect_ci95_upper": "{:.1f}", "unnecessary_pct": "{:.1f}", "threshold": "{:.4g}"},
                tex_columns=PERF_TEX)
    R = pd.read_csv(os.path.join(EV, "ratio_stats.csv"))
    R = R[R.regime != "all"][["set", "estimator", "regime", "n", "n_estimate_failed", "n_below_floor", "n_ratio", "median_ratio",
                              "p01_ratio", "p99_ratio", "within_factor_1.1", "within_factor_2", "n_over_10pct", "n_under_10pct", "spearman"]]
    write_table(R, "T4_estimate_ratio_by_regime",
                "Ratio of CW-only estimate to measured peak CW-to-reference difference by regime (unsaturated: CW never saturated; near boundary: |dwell margin| <= 600 s for CW or reference, or label mismatch; saturated: other). Measured peaks below 1e-6 m are excluded from ratios.",
                {"set": ("-", "evaluation set"), "estimator": ("-", "d1 = A, d1u = B, cres = C"), "regime": ("-", "regime definition in caption"),
                 "n": ("count", "valid scenarios in regime"), "n_estimate_failed": ("count", "non-finite estimates"),
                 "n_below_floor": ("count", "measured peak < 1e-6 m (numerical floor)"), "n_ratio": ("count", "scenarios in ratio statistics"),
                 "median_ratio": ("-", "median estimate/measured"), "p01_ratio": ("-", "1st percentile"), "p99_ratio": ("-", "99th percentile"),
                 "within_factor_1.1": ("fraction", "share with ratio in [1/1.1, 1.1]"), "within_factor_2": ("fraction", "share with ratio in [0.5, 2]"),
                 "n_over_10pct": ("count", "overestimates by more than 10%"), "n_under_10pct": ("count", "underestimates by more than 10%"),
                 "spearman": ("-", "Spearman rank correlation of estimate and measured peak")},
                fmt={c: "{:.3g}" for c in ("median_ratio", "p01_ratio", "p99_ratio", "within_factor_1.1", "within_factor_2", "spearman")})
    fd = pd.read_csv(os.path.join(C.FIGS, "Fig4_data.csv"))
    write_table(fd, "T5_acceptance_target_rules",
                "Descriptive acceptance-target rules: threshold set to the development quantile giving 25/50/75% acceptance (frozen), then applied to the fresh set. Thresholds were not tuned on test outcomes.",
                {"feature": ("-", "feature (see T1)"), "dev_target_pct": ("%", "development acceptance target"),
                 "dev_achieved_pct": ("%", "development acceptance actually achieved (ties can exceed the target)"),
                 "fresh_accept_pct": ("%", "fresh-set acceptance"), "fresh_incorrect_pct_of_accepted": ("%", "accepted fresh scenarios with peak > 1 m, % of accepted"),
                 "fresh_incorrect": ("count", "accepted fresh scenarios with peak > 1 m"), "fresh_accepted": ("count", "fresh scenarios accepted")},
                fmt={"dev_achieved_pct": "{:.1f}", "fresh_accept_pct": "{:.1f}", "fresh_incorrect_pct_of_accepted": "{:.1f}"})
    A = pd.read_csv(os.path.join(EV, "arrival_condition_contribution.csv")).drop(columns=["removed_sids"])
    write_table(A, "T6_arrival_condition_contribution",
                "Effect of the CW-arrival and dwell-margin conditions: scenarios accepted by the trajectory-only D1 screen (A, D1 <= 0.7998 m) but referred by the combined rule.",
                {"set": ("-", "evaluation set"), "n": ("count", "valid scenarios"), "traj1_d1_accepted": ("count", "accepted by D1 alone"),
                 "comb_d1_accepted": ("count", "accepted by the combined rule"), "removed_by_arrival_conditions": ("count", "accepted by D1 alone but referred by the combined rule"),
                 "removed_that_were_adequate": ("count", "of those, adequate (unnecessary referrals)"),
                 "removed_that_were_outcome_mismatches": ("count", "of those, CW/reference arrival-label mismatches (errors prevented)"),
                 "removed_cw_nonarrivals": ("count", "removed because CW does not arrive"), "removed_cw_margin_below_m_min": ("count", "removed because CW dwell margin < 380.56 s"),
                 "outcome_mismatches_in_set": ("count", "arrival-label mismatches in the set"),
                 "outcome_mismatches_accepted_by_traj1_d1": ("count", "mismatches that D1 alone would have accepted")})


def t_verifier():
    V = os.path.join(C.DERIVED, "verifier")
    if not os.path.exists(os.path.join(V, "verifier_comparison.csv")):
        print("verifier tables skipped (no comparison yet)")
        return
    R = pd.read_csv(os.path.join(V, "verifier_comparison.csv"))
    cols = ["sid", "rule", "cw_ref_peak_prod_m", "cw_ref_peak_absdiff_m", "so_ref_peak_absdiff_m", "ref_entry_prod_s",
            "ref_entry_impl_diff_s", "ref_entry_monitor_diff_s", "cw_entry_impl_diff_s", "so_entry_impl_diff_s",
            "ref_margin_prod_s", "ref_traj_maxdiff_m", "ref_status_agree", "so_status_agree", "outcome_cw_prod", "outcome_cw_ind",
            "outcome_so_prod", "outcome_so_ind"]
    T = R[[c for c in cols if c in R]]
    defs = {"sid": ("-", "scenario ID (selection reason in results/v2/raw/verifier/selection.csv)"), "rule": ("-", "selection rule R1-R9"),
            "cw_ref_peak_prod_m": ("m", "production peak CW-to-reference difference"),
            "cw_ref_peak_absdiff_m": ("m", "|production - independent| of the CW-to-reference peak"),
            "so_ref_peak_absdiff_m": ("m", "|production - independent| of the second-order-to-reference peak"),
            "ref_entry_prod_s": ("s", "production final-entry time of the reference (NaN: no arrival)"),
            "ref_entry_impl_diff_s": ("s", "|production - independent| reference final-entry time, same 0.5 s monitoring convention (implementation check)"),
            "ref_entry_monitor_diff_s": ("s", "|independent 0.5 s convention - independent 0.01 s convention| (monitoring-resolution effect)"),
            "cw_entry_impl_diff_s": ("s", "as ref_entry_impl_diff_s for the CW branch"), "so_entry_impl_diff_s": ("s", "as ref_entry_impl_diff_s for the second-order branch"),
            "ref_margin_prod_s": ("s", "production reference dwell margin"), "ref_traj_maxdiff_m": ("m", "max |production - independent| reference relative position on the 0.5 s grid"),
            "ref_status_agree": ("-", "same reference branch status"), "so_status_agree": ("-", "same second-order branch status"),
            "outcome_cw_prod": ("-", "CW arrived / reference arrived (production)"), "outcome_cw_ind": ("-", "same, independent verifier"),
            "outcome_so_prod": ("-", "second order arrived / reference arrived (production)"), "outcome_so_ind": ("-", "same, independent verifier")}
    write_table(T, "T7_verifier_benchmark_cases", "Production pipeline versus the independent verifier (Radau, rtol 1e-12) on the 24 preselected benchmark cases.",
                {k: defs[k] for k in T.columns})
    Cv = pd.read_csv(os.path.join(V, "verifier_convergence.csv"))
    pk = R.set_index("sid")
    Cv["cw_peak_prod_m"] = Cv.sid.map(pk.cw_ref_peak_prod_m)
    Cv["so_peak_prod_m"] = Cv.sid.map(pk.so_ref_peak_prod_m)
    Cv["so_failed"] = Cv.sid.map(pk.so_status_prod) == "failed"
    rows = []
    for (fam, st), g in Cv.groupby(["family", "setting"]):
        small, large = g[g.cw_peak_prod_m <= 10.0], g[g.cw_peak_prod_m > 10.0]
        gs = g[~g.so_failed]
        ss, sl = gs[gs.so_peak_prod_m <= 10.0], gs[gs.so_peak_prod_m > 10.0]
        lab = [c for c in g.columns if c.endswith("label_agrees_with_finest")]
        rows.append({"family": fam, "setting": st, "compared_to": g.finest.iloc[0], "cases": g.sid.nunique(),
                     "cw_peak_abs_change_m": small.cw_peak_diff_to_finest_m.max(),
                     "cw_peak_rel_change": (large.cw_peak_diff_to_finest_m / large.cw_peak_prod_m).max(),
                     "so_peak_abs_change_m": ss.so_peak_diff_to_finest_m.max(),
                     "so_peak_rel_change": (sl.so_peak_diff_to_finest_m / sl.so_peak_prod_m).max(),
                     "ref_entry_change_s": g.ref_entry_diff_to_finest_s.max(), "cw_entry_change_s": g.cw_entry_diff_to_finest_s.max(),
                     "so_entry_change_s": gs.so_entry_diff_to_finest_s.max(), "median_wall_s": g.wall_s.median()})
    T8 = pd.DataFrame(rows)
    write_table(T8, "T8_verifier_convergence",
                "Independent verifier convergence: largest change over the 24 cases relative to the finest setting of the same solver family "
                "(cross: finest Radau vs finest DOP853 and RK4). Peaks above 10 m are compared relatively; the second-order branch of the one "
                "case where it fails numerically is excluded. Every one of the 288 runs gave the production status and arrival label for all branches.",
                {"family": ("-", "radau, dop853, rk4, or cross (finest settings of different solvers)"), "setting": ("-", "tolerance or step"),
                 "compared_to": ("-", "setting it is compared with"), "cases": ("count", "cases"),
                 "cw_peak_abs_change_m": ("m", "max change of the CW-to-reference peak, cases with peak <= 10 m"),
                 "cw_peak_rel_change": ("-", "max relative change of the CW-to-reference peak, cases with peak > 10 m"),
                 "so_peak_abs_change_m": ("m", "same for the second-order-to-reference peak, peak <= 10 m"),
                 "so_peak_rel_change": ("-", "same, relative, peak > 10 m"),
                 "ref_entry_change_s": ("s", "max change of the reference final-entry time"), "cw_entry_change_s": ("s", "same, CW"),
                 "so_entry_change_s": ("s", "same, second order"), "median_wall_s": ("s", "median wall time per case (3 branches)")},
                fmt={c: "{:.2e}" for c in ("cw_peak_abs_change_m", "cw_peak_rel_change", "so_peak_abs_change_m", "so_peak_rel_change",
                                           "ref_entry_change_s", "cw_entry_change_s", "so_entry_change_s")})


def t_families():
    F = os.path.join(C.DERIVED, "boundary_families", "family_results_predeclared.csv")
    if not os.path.exists(F):
        print("family table skipped")
        return
    T = pd.read_csv(F)
    defs = {"family_id": ("-", "predeclared family"), "direction": ("-", "offset direction"), "velocity_rule": ("-", "zero, em1 (ydot0 = -2 n x0) or scaled (v0 = nu n rho0 w)"),
            "wn_ratio": ("-", "omega_n / n"), "a_max": ("m/s^2", "acceleration cap"), "n_boundaries": ("count", "CW/reference boundary pairs found in 0.2-80 km"),
            "n_resolved": ("count", "pairs with |offset| >= 2 x combined bracket half-width"), "status": ("-", "boundary status"),
            "offset_pct": ("%", "(CW boundary / reference boundary - 1), per pair"), "offset_unc_pct": ("%", "combined bracket half-width, per pair"),
            "n_points": ("count", "diagnostic points"), "n_false_fav": ("count", "points where CW arrives and the reference does not"),
            "n_false_unfav": ("count", "points where the reference arrives and CW does not"), "ff_flagged": ("count", "falsely favourable points flagged by the frozen v1 dwell-margin flag"),
            "correct_arrivals": ("count", "points where both arrive"), "correct_arrivals_flagged": ("count", "correct arrivals flagged (false alarms)"),
            "combined_accepted": ("count", "points accepted by the frozen combined rule"), "combined_accepted_inadequate": ("count", "accepted points that were inadequate"),
            "outcome_accepted": ("count", "points accepted by the frozen outcome screen"), "outcome_accepted_mismatch": ("count", "accepted points with an arrival mismatch"),
            "outcome_referred_correct": ("count", "correct-label points referred by the outcome screen"), "traj1m_accepted": ("count", "points accepted by D1 <= 0.7998 m"),
            "sep_only_accepted": ("count", "accepted by the v1 separation baseline"), "sat_only_accepted": ("count", "accepted by the v1 saturation baseline"),
            "sep_only_accepted_inadequate": ("count", "of those inadequate"), "sat_only_accepted_inadequate": ("count", "of those inadequate"),
            "ambiguous_points": ("count", "points with a numerical ambiguity flag (|dwell margin| < 0.5 s or NERM/reference label difference)"),
            "nerm_ref_label_diff": ("count", "points where NERM and the inertial reference disagree on arrival"), "min_peak_m": ("m", "smallest peak difference among the points")}
    write_table(T, "T9_boundary_families", "Twenty-four predeclared arrival-boundary families (one orbit): boundaries, offsets and the frozen v1 screening rules on the diagnostic points. Designed challenge; points within a family are dependent.",
                {k: defs[k] for k in T.columns})


def t_accounting():
    F = os.path.join(C.DERIVED, "failure_accounting")
    cor = pd.read_csv(os.path.join(F, "model_outcomes_status_aware.csv"))
    write_table(cor, "T10_model_outcomes_status_aware",
                "Status-aware terminal-arrival label agreement with the inertial reference (supersedes tab_model_outcomes_refined, which counted 164 second-order numerical failures as agreeing non-arrivals; that version is kept in results/v2/derived/failure_accounting/model_outcomes_superseded_v1.csv).",
                {"data": ("-", "data set"), "model": ("-", "model branch"), "N attempted": ("count", "scenarios propagated"),
                 "model numerical failures": ("count", "model branch solver failures (excluded from comparison)"),
                 "reference numerical failures": ("count", "reference solver failures (excluded)"),
                 "reference domain terminations": ("count", "reference reached the 100 km altitude floor (physical non-arrival; compared)"),
                 "comparable": ("count", "scenarios compared"), "label agrees": ("count", "same terminal-arrival label"),
                 "model arrives, reference not": ("count", "falsely favourable"), "reference arrives, model not": ("count", "falsely unfavourable"),
                 "both arrive": ("count", "both arrive"), "neither arrives": ("count", "neither arrives")})
    cen = pd.read_csv(os.path.join(F, "branch_status_census.csv")).drop(columns=["other"])
    write_table(cen, "T11_branch_status_census", "Branch-status census for every campaign and model branch (categories are mutually exclusive and sum to the scenario count).",
                {"campaign": ("-", "result folder"), "model": ("-", "branch (cw, so = second order, nerm, eci = inertial reference)"),
                 "n_scenarios": ("count", "scenarios"), "completed_arrival": ("count", "full horizon, primary terminal arrival"),
                 "completed_no_arrival": ("count", "full horizon, no arrival"), "domain_termination": ("count", "stopped at the 100 km altitude floor"),
                 "numerical_failure": ("count", "solver failure"), "terminations": ("-", "termination reasons and counts")}, max_rows_tex=40)
    inv = pd.read_csv(os.path.join(F, "so_failure_investigation.csv"))
    inv = inv[["sid", "axis", "sep_m", "wn_ratio", "a_max", "rerun_so_t_final_s", "t_range_exceeds_0.1R_s", "t_range_exceeds_R_s",
               "growth_exponent_p", "last_step_s", "reference_status", "reference_termination", "reference_arrived", "reference_max_range_over_R"]]
    write_table(inv, "T12_so_failure_investigation", "Representative second-order (quadratic) two-orbit failures re-run with the production solver.",
                {"sid": ("-", "scenario"), "axis": ("-", "offset direction"), "sep_m": ("m", "initial separation"), "wn_ratio": ("-", "omega_n / n"),
                 "a_max": ("m/s^2", "cap"), "rerun_so_t_final_s": ("s", "solver failure time (identical to the stored run)"),
                 "t_range_exceeds_0.1R_s": ("s", "time the second-order range first exceeds 0.1 R"), "t_range_exceeds_R_s": ("s", "time it first exceeds R"),
                 "growth_exponent_p": ("-", "fitted p in |rho| ~ (t* - t)^p for R <= |rho| <= 1e4 R (pure quadratic blow-up: -2)"),
                 "last_step_s": ("s", "last accepted step before failure"), "reference_status": ("-", "reference branch status"),
                 "reference_termination": ("-", "reference termination"), "reference_arrived": ("-", "reference terminal arrival"),
                 "reference_max_range_over_R": ("-", "reference maximum range / R")})


def main():
    t_estimator()
    t_verifier()
    t_families()
    t_accounting()
    C.log_run("tables", "python experiments/v2/v2_make_tables.py", "completed", 0.0, "T1-T12")
    print("tables written to", C.TABLES)


if __name__ == "__main__":
    main()
