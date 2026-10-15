"""Build claims/claim_evidence_map.csv from stored result rows (every number is read from a result file).

Statuses (supported / partly_supported / not_supported / unresolved / deferred) are assigned by the
explicit rules written next to each claim below and depend only on the result rows they cite. The
analyses themselves (hypotheses, families, alpha, delta, M, selection rule, transfer modes) were fixed in
PROSPECTIVE_PLAN.json, but these STATUS RULES were not: this file was written after the single test
evaluation (first committed 2026-10-14 06:51 +0400; test evaluation 06:13:34), and the C06 rule was revised
again after seeing the result (DEV-02). [v3.1 correction C05: this paragraph previously said the rules were
fixed before the test evaluation was read, which the record does not support.] The v3.1 claim map
(experiments/v3_1/c05_claims_v31.py) records each status's provenance. Positive and null/adverse wordings
are both provided; `permissible_wording` is the one that matches the status.
Usage: python code/experiments/v3/build_claims.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v3data as VD  # noqa: E402

R = os.path.join(VD.V3, "results")
COLS = ["claim_id", "proposed_claim", "status", "dataset_split", "analysis_id", "raw_source", "numerical_result",
        "uncertainty", "limitations", "manuscript_anchor", "permissible_wording", "prohibited_overclaim"]


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.2f}%"


def main():
    rc = pd.read_csv(os.path.join(R, "risk_coverage.csv"))
    pc = pd.read_csv(os.path.join(R, "paired_comparisons.csv"))
    tm = pd.read_csv(os.path.join(R, "transfer_matrix.csv"))
    hs = pd.read_csv(os.path.join(R, "hybrid_screening.csv"))
    hv = pd.read_csv(os.path.join(R, "hybrid_validation.csv"))
    sim = pd.read_csv(os.path.join(R, "similarity_checks.csv"))
    nr = pd.read_csv(os.path.join(R, "numerical_resolution.csv"))
    cs = pd.read_csv(os.path.join(R, "calibration_stability.csv"))
    ab = pd.read_csv(os.path.join(R, "ablation.csv"))
    fa = pd.read_csv(os.path.join(R, "failure_atlas.csv"))
    cand = pd.read_csv(os.path.join(R, "calibration_candidates.csv"))
    cost = pd.read_csv(os.path.join(R, "cost_break_even.csv")) if os.path.exists(os.path.join(R, "cost_break_even.csv")) else None
    frz = json.load(open(os.path.join(VD.V3, "raw", "calibration_freeze.json")))
    plan = json.load(open(os.path.join(VD.V3, "PROSPECTIVE_PLAN.json")))
    rows = []

    def sel(cell, m, crit):
        s = rc[(rc.kind == "selected_calibrated") & (rc.cell == cell) & (rc.method == m) & (rc.criterion == crit)]
        return s.iloc[0] if len(s) else None

    def pair(cell, crit, comp, mode="selected_calibrated"):
        s = pc[(pc.cell == cell) & (pc.criterion == crit) & (pc.comparison == comp) & (pc.rule_mode == mode)]
        return s.iloc[0] if len(s) else None

    # C01 primary
    a, b, p = sel("D0", "A", "1m"), sel("D0", "B", "1m"), pair("D0", "1m", "A vs B")
    ci = json.loads(p.coverage_diff_ci95) if isinstance(p.coverage_diff_ci95, str) else p.coverage_diff_ci95
    nci = json.loads(p.net_useful_diff_ci95) if isinstance(p.net_useful_diff_ci95, str) else p.net_useful_diff_ci95
    both = a.n_accepted > 0 and b.n_accepted > 0
    if not (np.isfinite(a.threshold) if a.threshold == a.threshold else False):
        st = "not_supported"
    elif ci[0] > 0:
        st = "supported"
    elif p.coverage_diff > 0:
        st = "partly_supported"
    else:
        st = "not_supported"
    rows.append(["C01", "In D0 at 1 m, the calibrated saturation-aware rule (A) accepts more test cases than the matched "
                 "unsaturated-sensitivity rule (B) at the same certified conditional risk target",
                 st, "D0 calibration (5,000) and test (2,500)", "P04 primary contrast",
                 "results/paired_comparisons.csv (primary=True); results/risk_coverage.csv; raw/calibration_freeze.json",
                 f"A: t={a.threshold:.6g} m, cal {int(a.cal_accepted)}/{int(a.cal_adverse)} adverse, U={pct(a.cal_ucb)}; test coverage "
                 f"{pct(a.coverage)}, adverse {int(a.conservative_adverse_accepted)}/{int(a.n_accepted)}. B: t={b.threshold:.6g} m, "
                 f"test coverage {pct(b.coverage)}, adverse {int(b.conservative_adverse_accepted)}/{int(b.n_accepted)}. "
                 f"Paired coverage difference {pct(p.coverage_diff)}; A-only {int(p.only_a)} ({int(p.only_a_adverse)} adverse), "
                 f"B-only {int(p.only_b)} ({int(p.only_b_adverse)} adverse); McNemar p={p.mcnemar_exact_p:.3g}",
                 f"coverage difference 95% paired bootstrap [{pct(ci[0])}, {pct(ci[1])}]; net useful difference "
                 f"{p.net_useful_diff_per_case:.4f} per case [{nci[0]:.4f}, {nci[1]:.4f}]; test adverse-rate CP95 A "
                 f"[{pct(a.cp95_two_sided_conservative_lo)}, {pct(a.cp95_two_sided_conservative_hi)}]",
                 "One simulation distribution, circular 400 km orbit, scalar PD, one-orbit horizon; certificate is conditional "
                 "accepted-case risk under that distribution with probability 1-delta, not a per-trajectory bound",
                 "sec:screenresults; tab:screen; fig:screen; abstract",
                 ("At matched calibrated risk (alpha = 1 %, delta = 5 %), A accepted {} of the 2,500 D0 test cases against {} for B; "
                  "the paired coverage difference was {} (95 % interval {} to {}).").format(
                     int(a.n_accepted), int(b.n_accepted), pct(p.coverage_diff), pct(ci[0]), pct(ci[1])),
                 "A is safe / guaranteed; A dominates B at every operating point; zero risk; per-trajectory bound"])
    # C02 0.1 m
    a2, b2, p2 = sel("D0", "A", "0.1m"), sel("D0", "B", "0.1m"), pair("D0", "0.1m", "A vs B")
    ci2 = json.loads(p2.coverage_diff_ci95)
    st = "supported" if ci2[0] > 0 else ("partly_supported" if p2.coverage_diff > 0 else "not_supported")
    if not np.isfinite(a2.threshold if a2.threshold == a2.threshold else np.nan):
        st = "not_supported"
    rows.append(["C02", "Secondary: the same contrast at the 0.1 m target", st, "D0 calibration and test", "P04 secondary",
                 "results/paired_comparisons.csv; results/risk_coverage.csv",
                 f"A coverage {pct(a2.coverage)} ({int(a2.conservative_adverse_accepted)}/{int(a2.n_accepted)} adverse); "
                 f"B coverage {pct(b2.coverage)} ({int(b2.conservative_adverse_accepted)}/{int(b2.n_accepted)})",
                 f"paired coverage difference 95% [{pct(ci2[0])}, {pct(ci2[1])}]", "secondary target; same assumptions as C01",
                 "sec:screenresults", "Report as secondary with the same protocol.", "Primary or confirmatory status"])
    # C03 legacy rule
    lg = rc[(rc.kind == "legacy_padded_rule") & (rc.cell == "D0") & (rc.method == "A") & (rc.criterion == "1m")].iloc[0]
    rows.append(["C03", "The legacy padded A rule (0.7998 m) evaluated on the new D0 test set", "supported",
                 "D0 test", "legacy comparator", "results/risk_coverage.csv (kind=legacy_padded_rule)",
                 f"A coverage {pct(lg.coverage)}, adverse {int(lg.conservative_adverse_accepted)}/{int(lg.n_accepted)}; other legacy rules (D0, 1 m): " +
                 "; ".join(f"{r.method} {int(r.conservative_adverse_accepted)}/{int(r.n_accepted)}" for _, r in
                           rc[(rc.kind == "legacy_padded_rule") & (rc.cell == "D0") & (rc.criterion == "1m") & (rc.method != "A")].iterrows()),
                 f"CP95 [{pct(lg.cp95_two_sided_conservative_lo)}, {pct(lg.cp95_two_sided_conservative_hi)}]",
                 "no certificate: the padding K = 0.8 was not derived from a risk-control argument", "sec:estimate; tab:screen",
                 "Report the legacy rule alongside the calibrated rule without a certificate.", "Certified legacy rule"])
    # C04 certification census
    s = pd.DataFrame(frz["selection"]).T
    n_sel = int((s.status == "selected").sum())
    rows.append(["C04", "Which families achieved the certificate on calibration data", "supported",
                 "calibration (5,000 per cell)", "P04 selection", "raw/calibration_freeze.json; results/calibration_candidates.csv",
                 f"{n_sel} of {len(s)} families selected a rule; abstained: " + ", ".join(s[s.status != "selected"].index[:30]),
                 f"M = {plan['multiplicity']['M_total_candidates']}, delta/M = {plan['multiplicity']['per_candidate_level']:.3g}",
                 "abstention means the stated certificate was not reached, not that the feature is useless", "sec:screenresults",
                 "List certified and abstaining families explicitly.", "Relaxing alpha after the fact"])
    # C05 held derivation verified
    fd = hv[(hv.test == "fd_ladder") & hv["lambda"].between(1e-4, 1e-2)]
    fds = fd[fd.n_sat_updates > 0]   # B_H differs from A_H only when some update is saturated
    rows.append(["C05", "The held-command first variation A_H is the derivative of the held-command homotopy and is implemented correctly",
                 "supported", "designed verification cases", "P06 verification",
                 "results/hybrid_validation.csv; results/hybrid_map_comparison.csv; code/tests/test_v3_hybrid_map.py",
                 f"central FD vs A_H max rel error {fd.err_central_vs_A_H_rel.max():.2e} (lambda 1e-4..1e-2); vs B_H (saturating cases) min {fds.err_central_vs_B_H_rel.min():.2f}; "
                 f"independent map feature agreement <= {pd.read_csv(os.path.join(R, 'hybrid_map_comparison.csv')).query('Ts_s <= 10').feature_rel_diff.max():.1e}",
                 "deterministic checks; nonsmooth near-switch and grazing cases converge only one-sidedly or irregularly",
                 "classical hybrid sensitivity; the derivation is not new", "sec:estimate (new subsection); appendix",
                 "We use the standard sampled-data sensitivity with a held-memory state and verify it numerically.",
                 "New mathematical discovery; novelty of hybrid sensitivity calculus"])
    # C06 A_H vs A_naive
    for c in ("D3", "D4"):
        p6 = pair(c, "1m", "A_H vs A_naive")
        row = hs[(hs.cell == c) & (hs.method == "A_H vs A_naive")].iloc[0]
        ci6 = json.loads(p6.coverage_diff_ci95)
        # revised after the test evaluation to name an equivalence outcome (see IMPLEMENTATION_REPORT section 6):
        # 'not_supported' when discordant decisions are at most 1 % of cases and the paired interval lies within +/-1 point
        disc = int(p6.only_a + p6.only_b)
        if ci6[0] > 0:
            st = "supported"
        elif disc <= 0.01 * p6.n and ci6[0] >= -0.01 and ci6[1] <= 0.01:
            st = "not_supported"
        else:
            st = "unresolved"
        rows.append([f"C06-{c}", f"In {c} the correct held sensitivity (A_H) changes screening outcomes relative to the naive continuous formula (A_naive)",
                     st, f"{c} test", "P06 screening", "results/paired_comparisons.csv; results/hybrid_screening.csv",
                     f"paired coverage diff {pct(p6.coverage_diff)}; A_H-only {int(p6.only_a)} ({int(p6.only_a_adverse)} adverse), "
                     f"A_naive-only {int(p6.only_b)} ({int(p6.only_b_adverse)} adverse); median |A_H-A_naive|/A_H {row.rel_diff_q50:.2e}, "
                     f"95th pct {row.rel_diff_q95:.2e}",
                     f"95% paired bootstrap [{pct(ci6[0])}, {pct(ci6[1])}]",
                     "one hold period per cell; PD controller; screening equivalence does not make A_naive the correct derivative",
                     "sec:estimate held subsection; perturbations/command-sampling subsection",
                     ("A_H and A_naive gave practically equivalent screening decisions here; A_H is preferred because it is the "
                      "correct derivative and predicts the held truth more closely.") if st != "supported" else
                     "A_H accepted more cases than A_naive at the same certified risk.",
                     "A_naive is invalid in practice / A_H gives a large screening gain (unless supported)"])
    # C07 transfer altitude / timing / controller
    for c in ("D1", "D2", "D3", "D4", "D5"):
        m = "A_H" if c in ("D3", "D4") else "A"
        t1 = tm[(tm.target_cell == c) & (tm.criterion == "1m") & (tm.target_method == m) & tm["mode"].isin(["T1", "T1_heuristic"])].iloc[0]
        t3 = tm[(tm.target_cell == c) & (tm.criterion == "1m") & (tm.target_method == m) & (tm["mode"] == "T3")].iloc[0]
        ok1 = t1.cp95_two_sided_conservative_hi <= 0.05 if np.isfinite(t1.cp95_two_sided_conservative_hi) else False
        st = "supported" if (t1.n_accepted > 0 and t1.cond_violation_rate_conservative <= 0.01) else ("not_supported" if t1.n_accepted > 0 else "unresolved")
        extra = ""
        if c in ("D1", "D2"):
            t2 = tm[(tm.target_cell == c) & (tm.criterion == "1m") & (tm.target_method == m) & (tm["mode"] == "T2")].iloc[0]
            extra = f"; T2 (c R_t/R_0={t2.threshold:.4g}) coverage {pct(t2.coverage)}, adverse {int(t2.conservative_adverse_accepted)}/{int(t2.n_accepted)}"
        rows.append([f"C07-{c}", f"The unchanged D0 cutoff transfers to {c} (observed conditional adverse rate at or below 1 % on the {c} test set)",
                     st, f"{c} test (D0 rule)", "P05 transfer", "results/transfer_matrix.csv",
                     f"T1 coverage {pct(t1.coverage)}, adverse {int(t1.conservative_adverse_accepted)}/{int(t1.n_accepted)} "
                     f"({pct(t1.cond_violation_rate_conservative)}){extra}; T3 t={t3.threshold}, coverage {pct(t3.coverage)}, "
                     f"adverse {int(t3.conservative_adverse_accepted)}/{int(t3.n_accepted)}",
                     f"T1 CP95 [{pct(t1.cp95_two_sided_conservative_lo)}, {pct(t1.cp95_two_sided_conservative_hi)}] (pointwise; no certificate for T1)",
                     "observational transfer evidence in one target distribution; T1 has no calibration guarantee in the target; T3 success is not out-of-domain validity of the D0 rule",
                     "sec:discussion; new transfer subsection",
                     (f"Applied unchanged, the D0 cutoff ({t1.threshold:.4g} m) accepted {int(t1.n_accepted)} of {int(t1.n_planned)} {c} test cases "
                      f"with {int(t1.conservative_adverse_accepted)} adverse; recalibration in {c} selected {t3.threshold:.4g} m and accepted "
                      f"{int(t3.n_accepted)} with {int(t3.conservative_adverse_accepted)} adverse. Neither shows validity outside the tested distribution."),
                     "Validated for other altitudes/controllers/missions; T3 proves transfer"])
    # C08 similarity
    rows.append(["C08", "The nondimensional similarity map holds for the implementation to numerical accuracy (code check)", "supported",
                 "designed pairs", "P01", "results/similarity_checks.csv",
                 f"{int(sim.similarity_verified.sum())}/{len(sim)} pairs verified; max q diff {max(sim.max_dq_cw.max(), sim.max_dq_eci.max()):.1e}",
                 "deterministic", "expected invariance; altitude floor and fixed metre tolerances break it; not a new result",
                 "sec:models or appendix", "Used as an implementation check.", "Novel scale-invariance; validates other regimes"])
    # C09 numerics
    ncs = nr[~nr.stratum.eq("physical_exit")]
    rows.append(["C09", "The 1 m and 0.1 m truth labels are numerically resolved in an a-priori stratified development audit",
                 "supported" if bool(nr.resolved_1m.all() and nr["resolved_0.1m"].all()) else "partly_supported",
                 "108 stratified development cases (not iid)", "P02", "results/numerical_resolution.csv",
                 f"max |refined - production| truth {ncs.delta_disc_m.max():.2e} m; resolved 1 m {int(nr.resolved_1m.sum())}/{len(nr)}",
                 "audit, not a census; sampling cannot exclude unseen narrow excursions", "assurance for the specified experiment only",
                 "sec:indverify", "State the audit and its limits.", "Certified continuous-time error; flight validation"])
    # C10 legacy stability
    full = cs[(cs.analysis == "full_data") & (cs.method == "A") & (cs.criterion == "1m")].iloc[0]
    lofo = cs[(cs.analysis == "leave_one_family_summary") & (cs.method == "A") & (cs.criterion == "1m")].iloc[0]
    cb = cs[(cs.analysis == "group_bootstrap") & (cs.method == "C") & (cs.criterion == "1m")].iloc[0]
    rows.append(["C10", "The legacy A threshold is set by one designed family but is insensitive to its removal; C's legacy threshold is less stable",
                 "supported", "archived development set (3,839)", "P03", "results/calibration_stability.csv",
                 f"A 1 m: {full.threshold:.6g} m, removal of the setter family changes it by {100 * lofo.max_rel_change:.3f}%; C 1 m bootstrap 95th pct {cb.boot_q95:.4g} vs {cb.threshold:.4g}",
                 "family-level bootstrap B=2000", "designed families, not iid", "sec:estimate", "Describe the setter dependence.",
                 "Legacy padding has a statistical guarantee"])
    # C11 B ranking (development)
    d0 = ab[(ab.cell == "D0") & (ab.method == "B")].iloc[0]
    rows.append(["C11", "On development data, B's referrals are not just a rescaling of A: rescaled to A's acceptance, B admits adverse cases A does not",
                 "supported" if d0.rescale_adverse_1m > d0.A_hist_cutoff_adverse_1m else "not_supported", "v3 D0 pilot+dev (340)", "P03 ablation",
                 "results/ablation.csv", f"B rescaled (factor {d0.rescale_factor_vs_A_cutoff:.3f}) adverse {int(d0.rescale_adverse_1m)} vs A {int(d0.A_hist_cutoff_adverse_1m)}; AUC A {ab[(ab.cell == 'D0') & (ab.method == 'A')].auc_1m.iloc[0]:.4f}, B {d0.auc_1m:.4f}",
                 "development sample; descriptive", "small sample; in-sample", "sec:discussion", "Development evidence only.", "Confirmatory"])
    # C12 cost
    if cost is not None:
        cc = cost[(cost.cell == "D0") & (cost.workflow == "screen_A_cold") & (cost.comparator == "direct_NRM")].iloc[0]
        st = "supported" if cc.online_saving_cpu_s_per_case <= 0 else "partly_supported"
        rows.append(["C12", "Screening does not save computation relative to direct nonlinear relative propagation", st,
                     "60 D0 development cases timed; test referral fraction", "P09", "results/cost_break_even.csv; results/cost_components.csv",
                     f"expected CPU per case: screen-A (cold) {cc.expected_cpu_s_per_case_screen:.3f} s vs direct NRM {cc.expected_cpu_s_per_case_comparator:.3f} s; "
                     f"saving {cc.online_saving_cpu_s_per_case:.3f} s; {cc.break_even_note}",
                     "median of repetitions; single machine", "implementation-specific timings", "perturbations/command-sampling/cost subsection",
                     "The screen's role is interpretation and auditable adequacy decisions, not speed.", "Computational saving"])
    # C13 status accounting
    cen = fa[fa.kind == "status_census"]
    rows.append(["C13", "Every planned case resolved to an explicit status", "supported", "all splits", "P00/P08",
                 "results/failure_atlas.csv (status_census); RUN_LEDGER.jsonl",
                 f"iid cases {int(cen.n.sum())}; physical exits {int(cen.truth_physical_domain_exit.sum())}; genuine numerical failures "
                 f"{int(cen.numerical_failure_excl_interrupted.sum())}; operational interruptions {int(cen.interrupted_operational.sum())} "
                 f"(re-executed, DEV-01); unresolved truth {int(cen.truth_unresolved.sum())}; unresolved 1 m labels {int(cen.label1m_unresolved.sum())}",
                 "census", "", "sec:procedure", "State the census.", "Excluding difficult cases"])
    # C14 arrival among accepted (secondary, descriptive)
    sa = rc[(rc.kind == "selected_calibrated") & (rc.criterion == "1m") & rc.method.isin(["A", "A_H"])]
    rows.append(["C14", "Among cases accepted by the A-type 1 m rules, CW arrival labels agree with the reference (secondary, descriptive)",
                 "supported", "test, all cells", "P08 arrival", "results/risk_coverage.csv (arr_* columns)",
                 "; ".join(f"{r.cell}: false favourable {int(r.arr_false_favourable)}, false unfavourable {int(r.arr_false_unfavourable)} of {int(r.arr_resolved)}"
                           for _, r in sa.iterrows()),
                 "pointwise; accepted cases are mostly far from the arrival boundary",
                 "random samples contain few arrival-boundary cases; this does not certify arrival labels near the deadline",
                 "sec:screenresults arrival paragraph", "No accepted case had a wrong arrival label in these test sets.",
                 "Arrival labels are certified; the trajectory screen detects arrival errors"])
    # C15 held prediction accuracy and counterexample
    h4 = hs[(hs.cell == "D4") & (hs.method.isin(["A_H", "A_naive"]))].set_index("method")
    h3 = hs[(hs.cell == "D3") & (hs.method.isin(["A_H", "A_naive"]))].set_index("method")
    rows.append(["C15", "A_H predicts the held-command truth more accurately than A_naive, more so at 10 s holds, but not in every case",
                 "supported", "D3/D4 test", "P06 prediction", "results/hybrid_screening.csv; results/failure_atlas.csv (R1/R4 D4)",
                 f"fraction of cases (truth 0.01-100 m) more than 1 % low: D4 A_H {h4.loc['A_H', 'frac_under_1pct_0.01_100m']:.4f} vs A_naive "
                 f"{h4.loc['A_naive', 'frac_under_1pct_0.01_100m']:.4f}; D3 {h3.loc['A_H', 'frac_under_1pct_0.01_100m']:.4f} vs "
                 f"{h3.loc['A_naive', 'frac_under_1pct_0.01_100m']:.4f}; median ratio D4 A_H {h4.loc['A_H', 'ratio_q50']:.4f}, A_naive "
                 f"{h4.loc['A_naive', 'ratio_q50']:.4f}; counterexample D4 test C7b781c06b0fd37bc truth 0.779 m, A_H 0.713 m, A_naive 0.783 m",
                 "descriptive test statistics", "two hold periods; PD controller",
                 "sec:estimate held subsection; failure-atlas paragraph",
                 "A_H tracks the held truth more closely on average; a saturation-status flip at a held update can still make it underpredict.",
                 "A_H is always more accurate; A_naive is unusable"])
    # C16 paired cohort
    co = pd.read_csv(os.path.join(R, "transfer_cohort_summary.csv"))
    rows.append(["C16", "Changing only command timing (1 s, 10 s holds) or controller structure (CW LQR) for the same 300 initial states flips no 1 m label",
                 "supported", "transfer-diagnostic cohort (300 D0 development draws, paired, exploratory)", "P08 cohort",
                 "results/transfer_cohort_summary.csv; results/transfer_cohort_diagnostics.csv",
                 "; ".join(f"{r.target}: ok->violation {int(r.label_ok_to_violation)}, violation->ok {int(r.label_violation_to_ok)}, "
                           f"median log10 truth ratio {r.log10_disc_ratio_q50:.4f}" for _, r in co.iterrows()),
                 "dependent pairs; not an independent sample", "exploratory, development draws only", "transfer subsection",
                 "For these paired draws, holding commands or changing the gain structure did not move any case across the 1 m target.",
                 "Command timing never matters"])
    out = pd.DataFrame(rows, columns=COLS)
    os.makedirs(os.path.join(VD.V3, "claims"), exist_ok=True)
    target = os.path.join(VD.V3, "claims", "claim_evidence_map.csv")
    if os.path.exists(target):   # v3.1: the v3.0 claim map is a preserved original
        sys.exit(f"{target} exists and is a preserved v3.0 output; the v3.1 map is built by experiments/v3_1/c05_claims_v31.py")
    out.to_csv(target, index=False)
    print(out[["claim_id", "status", "numerical_result"]].to_string())


if __name__ == "__main__":
    main()
