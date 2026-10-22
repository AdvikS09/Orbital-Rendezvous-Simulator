import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import beta

import v31paths as P

R = os.path.join(P.V3, "results")
BUILDCLAIMSPROV = ("status assigned by a rule in experiments/v3/buildclaims.py, written after the test "
                   "evaluation (first committed 2026-10-14 06:51 +0400; test evaluation 06:13:34)")
NEWCOLS = ["evidence_type", "status_rule_provenance", "status_v30", "change_note"]


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.2f}%"


def pp(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.2f}"


def cpupper(b, n, conf=0.95):
    return 1.0 if b >= n else float(beta.ppf(conf, b + 1, n - b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(P.V311DOCS, "claims"))
    outdir = ap.parse_args().out
    old = pd.read_csv(os.path.join(P.V3, "claims", "claimmap.csv"))
    pc = pd.read_csv(os.path.join(R, "paired.csv"))
    rc = pd.read_csv(os.path.join(R, "riskcoverage.csv"))
    tm = pd.read_csv(os.path.join(R, "transfermatrix.csv"))
    nr = json.load(open(os.path.join(P.V31RES, "numerical31summary.json"), encoding="utf-8"))
    nrt = pd.read_csv(os.path.join(P.V31RES, "numerical31.csv"))
    dec = nr["decisions"]
    ex = json.load(open(os.path.join(P.V31RES, "exitsummary.json"), encoding="utf-8"))
    d1 = json.load(open(os.path.join(P.V31RES, "D1addendum.json"), encoding="utf-8"))
    ci = json.load(open(os.path.join(P.V31RES, "contractimpact.json"), encoding="utf-8"))
    rows = {r.claim_id: r.to_dict() for _, r in old.iterrows()}
    for r in rows.values():
        r.update({"status_v30": r["status"], "change_note": "", "evidence_type": "", "status_rule_provenance": BUILDCLAIMSPROV})

    def edit(cid, note, **kw):
        r = rows[cid]
        r.update(kw)
        r["change_note"] = (r["change_note"] + " " + note).strip()

    a = rc[(rc.kind == "selected_calibrated") & (rc.cell == "D0") & (rc.method == "A") & (rc.criterion == "1m")].iloc[0]
    b = rc[(rc.kind == "selected_calibrated") & (rc.cell == "D0") & (rc.method == "B") & (rc.criterion == "1m")].iloc[0]
    p = pc[(pc.cell == "D0") & (pc.criterion == "1m") & (pc.comparison == "A vs B") & (pc.rule_mode == "selected_calibrated")].iloc[0]
    c95 = json.loads(p.coverage_diff_ci95)
    olddiff = f"Paired coverage difference {pct(p.coverage_diff)};"
    assert rows["C01"]["numerical_result"].count(olddiff) == 1
    oldunc = f"coverage difference 95% paired bootstrap [{pct(c95[0])}, {pct(c95[1])}]"
    assert str(rows["C01"]["uncertainty"]).count(oldunc) == 1
    edit("C01", "Wording: same conditional-risk ceiling; best admissible frozen candidate; no observed adverse acceptance; "
                "coverage difference in percentage points (v3.1.1).",
         proposed_claim="In D0 at 1 m, the best admissible frozen candidate for A accepts more test cases than the best "
                        "admissible frozen candidate for B, both selected under the same conditional-risk ceiling",
         uncertainty=str(rows["C01"]["uncertainty"]).replace(
             oldunc, f"paired coverage difference {pp(p.coverage_diff)} percentage points, 95 % paired bootstrap interval "
                      f"{pp(c95[0])} to {pp(c95[1])} percentage points"),
         numerical_result=rows["C01"]["numerical_result"].replace(
             olddiff, f"Paired coverage difference {pp(p.coverage_diff)} percentage points (95 % paired bootstrap interval "
                       f"{pp(c95[0])} to {pp(c95[1])});"),
         evidence_type="confirmatory (prespecified primary contrast H1, two-sided, single test evaluation)",
         status_rule_provenance="analysis prespecified in plan.json (H1, primary_claim); " + BUILDCLAIMSPROV +
                                "; the rule used (paired 95 % interval excludes 0) is the conventional reading of H1",
         permissible_wording=(f"Under the same conditional-risk ceiling (alpha = 1 %, delta = 5 %, M = 1,012 candidates), the best "
                              f"admissible frozen A candidate accepted {int(a.n_accepted)} of the 2,500 D0 test cases with no observed "
                              f"adverse acceptance, against {int(b.n_accepted)} with {int(b.conservative_adverse_accepted)} adverse "
                              f"for B; the paired coverage difference was {pp(p.coverage_diff)} percentage points (95 % paired "
                              f"bootstrap interval {pp(c95[0])} to {pp(c95[1])} percentage points)."),
         prohibited_overclaim=rappend(rows["C01"]["prohibited_overclaim"], "'certified safe'; the ceiling holds for every trajectory"))
    p2 = pc[(pc.cell == "D0") & (pc.criterion == "0.1m") & (pc.comparison == "A vs B") & (pc.rule_mode == "selected_calibrated")].iloc[0]
    c95b = json.loads(p2.coverage_diff_ci95)
    edit("C02", "Evidence type and wording made explicit; coverage difference in percentage points (v3.1.1).",
         evidence_type="secondary (prespecified, same protocol)",
         uncertainty=f"paired coverage difference {pp(p2.coverage_diff)} percentage points, 95 % paired bootstrap interval "
                     f"{pp(c95b[0])} to {pp(c95b[1])} percentage points",
         status_rule_provenance="analysis prespecified (secondary_claims); " + BUILDCLAIMSPROV,
         permissible_wording="Report as secondary: under the same conditional-risk ceiling at 0.1 m, " + rows["C02"]["numerical_result"] +
                             "; no observed adverse acceptance for A.")
    edit("C03", "Status changed from 'supported' to 'descriptive': the legacy padding itself has no certificate.",
         status="descriptive", evidence_type="descriptive (legacy comparator, no certificate)",
         proposed_claim="Descriptive record of the legacy padded A rule (0.7998 m, padding K = 0.8) on the new D0 test set; the padding itself carries no certificate",
         permissible_wording=("The legacy padded rule (0.7998 m) has no certificate: its padding K = 0.8 was not derived from a "
                              "risk-control argument. The same value was entered as a frozen candidate (cid 0) and was selected "
                              "under the conditional-risk ceiling, so the selection statement attaches to the calibration procedure, "
                              "not to the padding argument. On the D0 test set it showed no observed adverse acceptance (0 of 1,645); "
                              "this is a descriptive record, not a guarantee."),
         prohibited_overclaim="The legacy padding is certified / validated / safe")
    edit("C04", "Status 'descriptive'; abstentions kept visible; certificate wording replaced.", status="descriptive",
         evidence_type="descriptive (census of the frozen selection)",
         proposed_claim="Which families had an admissible candidate (U <= alpha) on calibration data under the same conditional-risk ceiling",
         limitations="abstention means no candidate had U <= alpha under the ceiling, not that the feature is useless",
         permissible_wording="58 of 64 families had an admissible candidate (U <= alpha) under the same conditional-risk ceiling on "
                             "calibration data; the six saturation-fraction families at 0.1 m had none (abstained) and refer every case.")
    edit("C05", "Evidence type.", status="verification", evidence_type="verification (designed deterministic checks)",
         status_rule_provenance="verification result; no statistical status rule")
    for cell in ("D3", "D4"):
        q = pc[(pc.cell == cell) & (pc.criterion == "1m") & (pc.comparison == "A_H vs A_naive") & (pc.rule_mode == "selected_calibrated")].iloc[0]
        q01 = pc[(pc.cell == cell) & (pc.criterion == "0.1m") & (pc.comparison == "A_H vs A_naive") & (pc.rule_mode == "selected_calibrated")].iloc[0]
        disc = int(q.only_a + q.only_b)
        civ = json.loads(q.coverage_diff_ci95)
        degen = civ[0] == 0.0 and civ[1] == 0.0
        ub = cpupper(disc, int(q.n))
        unc = (f"paired bootstrap interval {'[0, 0] is degenerate (no discordant decision; it states no uncertainty)' if degen else f'[{pct(civ[0])}, {pct(civ[1])}] rests on {disc} discordant decision(s)'}; "
               f"exact one-sided 95 % upper bound on the discordance rate {pct(ub)} ({disc}/{int(q.n)})")
        edit(f"C06-{cell}", "Status 'descriptive' instead of 'not_supported' (DEV-02's post-test equivalence rule is not used for a "
                            "status; no equivalence margin was prespecified); prediction accuracy separated (C15); degenerate interval flagged.",
             status="descriptive", evidence_type="secondary (H2), descriptive screening comparison; no prespecified decision rule",
             status_rule_provenance="H2 (secondary) names the comparison but no decision rule; the v3.0 status came from a rule revised "
                                    "after the test evaluation (DEV-02); v3.1 reports the comparison descriptively",
             proposed_claim=f"In {cell}, screening decisions of the best admissible frozen A_H and A_naive candidates at 1 m",
             uncertainty=unc,
             permissible_wording=(f"In {cell}, A_H and A_naive made near-identical screening decisions: {disc} discordant of {int(q.n)} "
                                  f"test cases at 1 m and {int(q01.only_a + q01.only_b)} at 0.1 m (exact one-sided 95 % upper bound on "
                                  f"the 1 m discordance rate {pct(ub)}). No equivalence margin was prespecified, so this is descriptive. "
                                  "Prediction accuracy is reported separately (C15), including a case where A_H underpredicts."),
             limitations="one hold period per cell; PD controller; near-identical decisions do not make A_naive the correct derivative",
             prohibited_overclaim="A_H and A_naive are equivalent (no margin was prespecified); A_naive is a valid derivative; A_H gives a screening gain")
    for cell in ("D1", "D2", "D3", "D4", "D5"):
        cid = f"C07-{cell}"
        m = "A_H" if cell in ("D3", "D4") else "A"
        t1 = tm[(tm.target_cell == cell) & (tm.criterion == "1m") & (tm.target_method == m) & tm["mode"].isin(["T1", "T1_heuristic"])].iloc[0]
        st = ("observed_no_adverse" if int(t1.conservative_adverse_accepted) == 0 else "observed_adverse") if int(t1.n_accepted) > 0 else "unresolved"
        how = (f"{t1['mode']}: the D0 cutoff applied to {m}, a different feature definition" if cell in ("D3", "D4")
               else f"{t1['mode']}: the D0 rule applied unchanged")
        edit(cid, "T1/T2 transfer is observational: status from the observed adverse count.", status=st,
             evidence_type=f"observational ({how}; no certificate in the target cell); T3 inherits the target cell's own selection statement",
             status_rule_provenance="H3 prespecified as descriptive for T1/T2; " + BUILDCLAIMSPROV,
             proposed_claim=f"Observational: the unchanged D0 cutoff applied to the {cell} test set",
             prohibited_overclaim=rappend(rows[cid]["prohibited_overclaim"], "the D0 certificate extends to the target cell"))
    edit("C07-D5", "D5 T3 adverse acceptances kept visible in the wording (see C17).",
         permissible_wording=rows["C07-D5"]["permissible_wording"] + " The D5-calibrated rule's 3 adverse acceptances are reported in C17.")
    for cid, et in (("C08", "verification (implementation check)"), ("C10", "descriptive (archived development set, designed families)"),
                    ("C11", "development (in-sample, descriptive)"), ("C14", "secondary, descriptive"),
                    ("C16", "exploratory (paired transfer-diagnostic cohort, development draws)")):
        edit(cid, "Evidence type.", evidence_type=et, status=("verification" if cid == "C08" else "descriptive"))
    n = nr["n_cases"]
    r1, r01 = nr["resolution_counts"]["1m"], nr["resolution_counts"]["0.1m"]
    comp = nrt[(nrt.truth_prod_v31 == "completed_valid") & (nrt.truth_refined == "completed_valid")]
    edit("C09", "Rebuilt from the corrected audit (C04): exits are labelled exit, 'resolved' needs completed status, a finite "
                "uncertainty and convergence; the sample is rule-based, not a priori.",
         proposed_claim="In a rule-based stratified development audit, the 1 m and 0.1 m truth labels of completed cases are numerically resolved",
         status=("supported" if r1.get("resolved", 0) == len(comp) and r01.get("resolved", 0) == len(comp) else "partly_supported"),
         dataset_split=f"{n} development cases (pilot + dev), stratified by a rule first committed after the last development record (not shown to be a priori); not iid",
         analysis_id="P02 corrected (C04)", raw_source="v31/results/numerical31.csv",
         numerical_result=(f"completed in production and refinement: {len(comp)} of {n}; resolved at 1 m {r1.get('resolved', 0)}, at 0.1 m "
                           f"{r01.get('resolved', 0)}; exits {r1.get('exit', 0)} (labelled exit, not resolved/violation); max |refined - production| "
                           f"{nr['max_abs_numerical_difference_completed_m']:.1e} m; max uncertainty/margin {nr['max_uncertainty_over_margin_resolved']['1m']:.1e} "
                           f"(1 m), {nr['max_uncertainty_over_margin_resolved']['0.1m']:.1e} (0.1 m); frozen-feature numerical change max "
                           + ", ".join(f"{k} {v:.1e}" for k, v in nr["feature_frozen_def_num_err_rel_max_by_family"].items()) +
                           f" (relative); decisions at frozen thresholds: {dec['n_flips']} flips and {dec['n_ties']} exact tie (a development "
                           f"case whose feature is the threshold) of {dec['n_decisions']}; arrival agrees in "
                           f"{nr['arrival_comparison']['eci'].get('agree', 0)} of {nr['n_completed_both']} completed cases"),
         uncertainty="robust_peak lower bound with refinement-change uncertainty; not a certified continuous-time error bound",
         evidence_type="assurance audit (development sample; not a population estimate)",
         status_rule_provenance="v3.1 rule: resolved iff completed_valid in production and refinement, converged search, finite "
                                "uncertainty and uncertainty < 0.1 x margin (CORRECTION_REPORT C04)",
         permissible_wording="In a rule-based stratified audit of development cases, every completed case's 1 m and 0.1 m label was "
                             "resolved by the refined recomputation; exit cases are reported as exits.",
         prohibited_overclaim="certified continuous-time error; a-priori sample; population rate; flight validation")
    edit("C12", "Cost wording: held rows exclude referral work.", evidence_type="descriptive (implementation-specific timings)",
         status="descriptive",
         status_rule_provenance="post-test rule in buildclaims.py (saving <= 0) on implementation-specific timings; no prespecified decision rule",
         permissible_wording="Screening did not save computation: in D0 the screen's expected online CPU per case (referrals included) "
                             "exceeded direct nonlinear relative propagation for each timed feature (A, B, C; sep and sat were not timed). "
                             "The held-cell rows (D3, D4) are screen-only costs that exclude referral work, i.e. lower bounds on the "
                             "screen's cost; each comes from one 40-case batch per held cell, summarised as the median of three "
                             "repetitions (no dispersion archived), and "
                             "was slightly above the direct comparator (0.268 vs 0.237 s, 0.135 vs 0.125 s per case).",
         prohibited_overclaim="Computational saving; held rows are expected costs; the held comparison is precise")
    edit("C13", "Census unchanged; 'interrupted' is now a contract label (C02).", evidence_type="census", status="verification",
         status_rule_provenance="literal status in buildclaims.py (no rule); a census fact checked against the ledger",
         numerical_result=rows["C13"]["numerical_result"] + "; v3.1 contract relabels the 43 worker_crash records 'interrupted' (was numerical_failure); no other stored record changes")
    edit("C15", "Separated from the screening claim C06; counterexample kept; status 'descriptive' (H2 sets no decision rule).",
         evidence_type="secondary (H2), descriptive prediction accuracy", status="descriptive",
         status_rule_provenance="literal 'supported' in buildclaims.py (no rule); PROSPECTIVE_PLAN H2 sets no decision rule")
    d5 = rc[(rc.kind == "selected_calibrated") & (rc.cell == "D5") & (rc.method == "A") & (rc.criterion == "1m")].iloc[0]
    rows["C17"] = dict(claim_id="C17", proposed_claim="In D5 (CW LQR) the best admissible frozen A candidate at 1 m had adverse acceptances on the test set",
                       status="observed_adverse", dataset_split="D5 calibration (5,000) and test (2,500)", analysis_id="P04/P05 T3",
                       raw_source="results/riskcoverage.csv; results/transfermatrix.csv",
                       numerical_result=f"threshold {d5.threshold:.6g} m; accepted {int(d5.n_accepted)}; adverse {int(d5.conservative_adverse_accepted)} ({pct(d5.cond_violation_rate_conservative)})",
                       uncertainty=f"two-sided CP95 [{pct(d5.cp95_two_sided_conservative_lo)}, {pct(d5.cp95_two_sided_conservative_hi)}]",
                       limitations="the observed rate is below the 1 % ceiling; the ceiling is a statement about the selection procedure, not a promise of zero adverse cases",
                       manuscript_anchor="transfer subsection; tab:transfer",
                       permissible_wording=f"In D5 the best admissible frozen A candidate (1 m) accepted {int(d5.n_accepted)} test cases, {int(d5.conservative_adverse_accepted)} of them adverse.",
                       prohibited_overclaim="no adverse acceptances in any cell", evidence_type="secondary (prespecified family; single test evaluation)",
                       status_rule_provenance="count read from the test evaluation", status_v30="(not a separate row; inside C07-D5)",
                       change_note="New row so the D5 adverse acceptances stay visible.")
    rows["C18"] = dict(claim_id="C18", proposed_claim="Correction C01: held-engine exit storage biased exit-case discrepancy diagnostics; no label or decision changed",
                       status="correction_verified", dataset_split="all 341 held records with a physical-domain exit (D3/D4, all splits)",
                       analysis_id="C01 targeted replay", raw_source="v31/results/exitimpact.csv",
                       numerical_result=(f"dense state at the exit time: v3.0 {ex['floor_gap_v30_m']['eci']['min']:.3f}-{ex['floor_gap_v30_m']['eci']['max']:.1f} m above "
                                         f"the floor, v3.1 within {ex['floor_gap_v31_abs_max_m']['eci']:.1e} m; cw-eci peak change "
                                         f"{ex['cw_vs_eci_peak_delta_m']['all']['min']:.1f} to +{ex['cw_vs_eci_peak_delta_m']['all']['max']:.1f} m; statuses, exit "
                                         f"times and labels unchanged; features unchanged except a 1-ulp C_H_peak_m difference in "
                                         f"{ex['n_v31_unexpected_changes']} records, identical in the v3.0-engine replay (re-batching) "
                                         f"({ex['n_records']} records)"),
                       uncertainty="bit-exact comparison for unchanged fields (3 records differ by 1 ulp in C_H_peak_m identically in the v3.0-engine replay: re-batching)",
                       limitations="dev records (12) were produced by an engine whose source was not archived (25643c68)",
                       manuscript_anchor="numerics appendix", permissible_wording="A storage defect in the held engine's exit step biased exit-case discrepancy diagnostics by up to 154 m; exit cases are labelled exit, so no label, selection or decision changed.",
                       prohibited_overclaim="the defect affected decisions", evidence_type="verification (targeted replay)",
                       status_rule_provenance="replay comparison", status_v30="(new)", change_note="New row for the v3.1 correction.")
    rows["C19"] = dict(claim_id="C19", proposed_claim="Post-test sensitivity: D1 calibration with the 43 interrupted runs completed",
                       status="post_test_sensitivity", dataset_split="D1 calibration (5,000)", analysis_id="C06 addendum",
                       raw_source="v31/results/D1addendum.csv",
                       numerical_result=(f"selections changed {d1['n_families_selection_changed']} of 10; admissibility flips "
                                         f"{d1['n_candidates_admissibility_flipped']} of {d1['n_candidates_total']} candidates; candidate rows gaining "
                                         f"adverse cases {d1['n_candidate_rows_gaining_adverse_cases']} (all inadmissible before and after)"),
                       uncertainty="same alpha, delta and M; descriptive", limitations="re-executed after the test evaluation; outcome-independence of the 43 is not established",
                       manuscript_anchor="protocol deviations", permissible_wording="Completing the 43 interrupted D1 calibration runs after the test evaluation changes no selection.",
                       prohibited_overclaim="the interrupted runs were outcome-independent; the completed values replace the frozen selection",
                       evidence_type="post-test sensitivity", status_rule_provenance="comparison of selections", status_v30="(new)",
                       change_note="New row (C06).")
    rows["C20"] = dict(claim_id="C20", proposed_claim="Correction C02: conservative record contract applied to every stored record",
                       status="correction_verified", dataset_split=f"all {ci['n_records']} stored result records", analysis_id="C02 contract scan",
                       raw_source="v31/results/contractimpact.csv",
                       numerical_result=f"records changed {ci['n_records_changed']} (the worker_crash records: numerical_failure -> interrupted); features, arrivals and discrepancies unchanged",
                       uncertainty="exact", limitations="the contract hardens the code against malformed records; it does not change the frozen protocol",
                       manuscript_anchor="methods (labels)", permissible_wording="Operational interruptions are reported as 'interrupted', separate from numerical failures; they remain adverse if accepted.",
                       prohibited_overclaim="", evidence_type="verification", status_rule_provenance="scan", status_v30="(new)", change_note="New row (C02).")
    order = list(old.claim_id) + ["C17", "C18", "C19", "C20"]
    cols = list(old.columns) + NEWCOLS
    out = pd.DataFrame([rows[c] for c in order])[cols]
    os.makedirs(P.guarddocs(outdir), exist_ok=True)
    out.to_csv(P.guarddocs(os.path.join(outdir, "claimmap31.csv")), index=False)
    out[["claim_id", "status_v30", "status", "evidence_type", "status_rule_provenance", "change_note"]].to_csv(
        P.guarddocs(os.path.join(outdir, "claimstatus.csv")), index=False)
    lines = ["# Claim map v3.0 -> v3.1: wording and status diff", "",
             "Generated by code/experiments/v31/c05claims.py from claims/claimmap.csv (v3.0, unchanged) and the v3.1 "
             "outputs. Only changed fields are listed. Numbers that v3.1 does not recompute are copied from v3.0. "
             "v3.1.1 regeneration: coverage differences in percentage points; held timing provenance stated as one 40-case "
             "batch per held cell, median of three repetitions.", ""]
    oldm = {r.claim_id: r for _, r in old.iterrows()}
    for c in order:
        r = rows[c]
        lines.append(f"## {c}")
        if c not in oldm:
            lines += ["New row.", "", f"- status: `{r['status']}`", f"- claim: {r['proposed_claim']}", f"- wording: {r['permissible_wording']}", ""]
            continue
        o = oldm[c]
        anychange = False
        for f in old.columns:
            ov, nv = str(o[f]), str(r[f])
            if ov != nv:
                anychange = True
                lines += [f"- **{f}**", f"  - v3.0: {ov}", f"  - v3.1: {nv}"]
        lines += [f"- evidence type: {r['evidence_type'] or '(unchanged)'}", f"- note: {r['change_note'] or 'no change'}", ""]
        if not anychange:
            lines.insert(len(lines) - 2, "- no field changed")
    with open(P.guarddocs(os.path.join(outdir, "CLAIM_WORDING_DIFF.md")), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(out[["claim_id", "status_v30", "status"]].to_string())


def rappend(a, b):
    a = "" if a is None or (isinstance(a, float) and math.isnan(a)) else str(a)
    have = [x.strip() for x in a.replace(" / ", "; ").split(";") if x.strip()]
    new = [x.strip() for x in b.split(";") if x.strip() and x.strip().lower() not in {h.lower() for h in have}]
    return "; ".join([a] + new) if a else "; ".join(new)


if __name__ == "__main__":
    main()
