"""v3.1 documentation tables (no computation on campaign data beyond reading):

  v3_1_correction/DEPENDENCY_IMPACT_TABLE.csv      the C00 dependency map (118 rows, written before any fix from code
                                                   reading) with the verified v3.1 outcome and evidence file per row
  v3_1_correction/EXACT_INPUT_INDEX.csv            every v3.1 output -> producing script, git commit, exact inputs + SHA-256
  v3_1_correction/PROTOCOL_DEVIATIONS_v3.1_addendum.json   addendum to PROTOCOL_DEVIATIONS.json (which is not modified)
  v3_1_correction/results/transfer_cohort_manifest_check.json  1:1 check of the cohort manifests against D0_dev

The dependency map source is the C00 table produced during the audit reproduction (passed with --dep-table);
it is copied verbatim and only the two outcome columns are added.
v3.1.1: the outputs go to --out (required), never into the released v3.1 tree (guard_docs). The v3.1.1 corrected
DEPENDENCY_IMPACT_TABLE.csv is produced by experiments/v3_1_1/c14_corrected_docs.py from the released table; this
script was not rerun in v3.1.1 (its EXACT_INPUT_INDEX and deviation addendum describe the v3.1 release).
Usage: python code/experiments/v3_1/c08_docs.py --dep-table <C00 dependency_table.csv> --out <dir>
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess

import v31paths as P


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*a):
    r = subprocess.run(["git", "-C", P.REPO, *a], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout.strip()


# (defect, substring of item) -> (v3.1 outcome, evidence); first match wins, then the defect default
OUTCOME = [
    ("C01", "HeldBatch.run", "fixed (commit baa84e9): exiting rows keep the trial-step endpoint; stored end == start in 341/341 v3.0 replays, 0/341 v3.1", "results/exit_correction_summary.json; tests/test_v31_exit.py"),
    ("C01", "_locate_exit", "confirmed unchanged: t_end bit-identical in 341/341; fallback now flagged (0 occurrences); exit at t = 0 now detected", "results/exit_correction_impact.csv"),
    ("C01", "case_derivatives / dense", "fixed: dense() masks a branch after its t_end; dense radius at t_end within 4.7e-8 m of the floor (v3.0: 0.02-565 m above)", "results/exit_correction_summary.json"),
    ("C01", "held_metrics.py", "fixed: NaN guard in peak_discrepancy; exit_step/exit_fallback reported; peak now at the located exit time in 341/341 (v3.0: 35)", "results/exit_correction_summary.json"),
    ("C01", "campaign.py work_held", "targeted replay of the 341 exit records only (staged v3.1 worker); records keyed by the legacy run_id", "raw/results/held_exit_replay_v31.jsonl; raw/REPLAY_LEDGER.jsonl"),
    ("C01", "replay_case.py", "two R5 traces regenerated with the corrected engine (masked after t_end); v3.0 traces kept", "raw/counterexamples/"),
    ("C01", "p02_numerical_resolution.py refine_held", "superseded by experiments/v3_1/c04_numerical_audit.py on the corrected engine", "results/numerical_resolution_v31.csv"),
    ("C01", "verify_held_engine.py", "exit-path regressions added in tests/test_v31_exit.py (15 tests: interior, node, first-step and t = 0 exits, branches exiting in different steps, companion row, step halving incl. a saturated gain, dense state and event time vs adaptive solution, masking, common-interval discrepancy, numerical-failure freezing, failed root bracket, D4 dev 56 reproducer); verify_held_engine.py unchanged except the v3.0-tree guard", "tests_logs/"),
    ("C01", "_common_end_s", "confirmed unchanged bit for bit (341/341)", "results/exit_correction_impact.csv"),
    ("C01", "_t_end_s", "confirmed unchanged bit for bit (341/341)", "results/exit_correction_impact.csv"),
    ("C01", "_status", "confirmed unchanged (341/341)", "results/exit_correction_impact.csv"),
    ("C01", "_peak_m", "changed in the v3.1 replay records only (cw-eci -52.0 to +154.5 m; nerm-eci <= 1.9e-6 m); v3.0 records unchanged", "results/exit_correction_impact.csv"),
    ("C01", "_t_peak_s", "changed: peak time is the located exit time in 341/341", "results/exit_correction_impact.csv"),
    ("C01", "_peak_grid_m", "changed with the corrected exit segment", "results/exit_correction_impact.csv"),
    ("C01", "cw_*, A_H_*", "confirmed unchanged exactly, except C_H_peak_m in 3 records (1 ulp), identical in the v3.0-engine replay (re-batching); the final engine reproduces a full replay chunk (31 cases) by physics hash", "results/exit_correction_impact.csv; results/worker_equivalence_check.csv"),
    ("C01", "truth_record", "disc_m/u_num_m changed for exit records; truth status and labels unchanged (exit)", "results/exit_correction_impact.csv"),
    ("C01", "atlas_cases", "rebuilt: 327 rows (disc_m, u_num_m, ratio_A; t_peak 292); categories unchanged", "rebuilt/raw/atlas_cases.csv; results/CHANGED_OUTPUTS_MANIFEST.csv"),
    ("C01", "failure_atlas", "rebuilt: 2 rows (R5 truth_disc_m); counts and quantiles identical", "rebuilt/results/failure_atlas.csv"),
    ("C01", "counterexamples", "regenerated with the corrected engine", "raw/counterexamples/"),
    ("C01", "test_decisions", "rebuilt: disc_m/u_num_m of 101 exit rows; all decision columns identical", "rebuilt/raw/test_decisions.csv"),
    ("C01", "transfer_cohort_diagnostics", "rebuilt: 14 rows", "rebuilt/results/transfer_cohort_diagnostics.csv"),
    ("C01", "transfer_cohort_summary", "confirmed byte-identical", "results/CHANGED_OUTPUTS_MANIFEST.csv"),
    ("C01", "numerical_resolution.csv", "superseded by the corrected audit (v3.0 file kept)", "results/numerical_resolution_v31.csv"),
    ("C01", "TabV3_Numerics", "superseded by TabV31_Numerics (v3.0 file kept; the rebuilt copy is identical only because make_assets reads the uncorrected v3.0 audit)", "manuscript_assets/TabV31_Numerics.tex; results/CHANGED_OUTPUTS_MANIFEST.csv"),
    ("C01", "FigV3_S1", "superseded by FigV31_S1_Numerics (v3.0 file kept)", "manuscript_assets/FigV31_S1_Numerics.png"),
    ("C01", "pilot_dev", "rebuilt: 12 development exit rows; D3/D4 u_num p95/max", "rebuilt/results/dev/"),
    ("C01", "risk_coverage", "confirmed byte-identical after the rebuild (risk_coverage, paired_comparisons, transfer_matrix, hybrid_screening, controller_transfer)", "results/CHANGED_OUTPUTS_MANIFEST.csv"),
    ("C01", "C09", "rebuilt from the corrected audit", "claims/claim_evidence_map_v31.csv"),
    ("C01", "C13 / C16", "unchanged", "claims/claim_evidence_map_v31.csv"),
    ("C01", "manuscript_anchor", "wording in the v3.1 handoff", "reports/NONCOMPUTATIONAL_HANDOFF_v3.1.md"),
    ("C02", "", "contract v3.1-contract-2 in cwvalid3/contracts.py (whitelists, coverage through T, planned-horizon check, Boolean arrivals, independent features, 'interrupted', partition identity, v3.1 evaluate_rule) and experiments/v3_1/v31data.py; scan of all 47,983 stored records: only the 43 worker_crash records change (numerical_failure -> interrupted); staged workers (feature checkpoint, held CW-only fallback, separate A_naive pass) reproduce stored records exactly on the ok path; fault injection tested", "results/existing_record_contract_impact.csv; results/worker_equivalence_check.csv; tests/test_v31_contracts.py"),
    ("C03", "", "canonical serialization + record/physics (versioned) and legacy-comparable SHA-256 (cwvalid3/serialization.py); legacy hashes reconciled in a sidecar: 30,324 direct, 1 reconstructed ledger hash, 17,615 by rule nan-restore/1, 43 without legacy hash, 0 unreconciled; ledger not edited; IDs preserved and tested", "raw/legacy_hash_reconciliation.csv; tests/test_v31_serialization.py"),
    ("C04", "refine_held.epk", "fixed: held frozen feature on the 0.5 s grid including t = T, supremum by robust_peak with T included; "
                                     "scope correction: 37 held audit rows (D3 19, D4 18), not 36", "results/numerical_resolution_v31.csv"),
    ("C04", "", "corrected audit on the same 108 cases: 92 completed cases resolved at 1 m and 0.1 m, 16 exits labelled exit (partial-horizon columns), arrival 92/92 on completed cases, 505 of 506 decisions at frozen thresholds stable, 0 flips, 1 exact tie (development case), sample documented as rule-based (not shown to be a priori)", "results/numerical_resolution_v31*.csv/json"),
    ("C05", "", "v3.1 claim map with evidence type and status provenance; wording diff; v3.0 map kept", "claims/"),
    ("C06", "", "post-test completed-data addendum: 0 selection changes, 0 admissibility flips, 38 inadmissible candidate rows gain adverse cases; no outcome-independence assertion", "results/calibration_completed_D1_addendum.*; PROTOCOL_DEVIATIONS_v3.1_addendum.json"),
    ("C07", "make_package", "git failures raise; v3.0 package protected; v3.1 package script with verified inventory", "code/experiments/v3_1/make_package_v31.py"),
    ("C07", "make_assets", "rebuilt under the harness: all 6 figures' PNG and plot data byte-identical; PDF/EPS differ only in CreationDate; TabV3_Numerics and FigV3_S1 superseded", "results/CHANGED_OUTPUTS_MANIFEST.csv"),
    ("C07", "p09_cost", "timings not re-run; cost basis made explicit (held rows exclude referral work)", "results/cost_break_even_v31.csv"),
    ("C07", "cost_", "cost_break_even_v31.csv adds cost_basis; held values qualified as screen-only costs of one 40-case batch per held cell, median of 3 repetitions (no dispersion archived); only A, B, C timed; negative cost result kept", "results/cost_break_even_v31.csv"),
    ("C07", "", "versioned v3.1 outputs, changed-output manifest, review ZIP and full-release inventory", "results/CHANGED_OUTPUTS_MANIFEST.csv; OUTPUT_INVENTORY_v31.csv"),
]


def outcome(defect, item):
    for d, key, text, ev in OUTCOME:
        if d == defect and key and key in item:
            return text, ev
    for d, key, text, ev in OUTCOME:
        if d == defect and not key:
            return text, ev
    return "", ""


SCRIPTS = {
    "code/experiments/v3_1/c00_baseline.py": (["raw/original_v3_checksums.csv", "PROSPECTIVE_PLAN.json", "raw/calibration_freeze.json",
                                              "raw/test_evaluation_record.json"], ["raw/baseline_verification_*.{csv,json}", "raw/environment_record.json"]),
    "code/experiments/v3_1/c01_exit_replay.py": (["raw/manifests/D3_*.jsonl", "raw/manifests/D4_*.jsonl", "raw/results/D3_*.jsonl", "raw/results/D4_*.jsonl"],
                                                 ["raw/results/held_exit_replay_v31.jsonl", "raw/results/held_exit_replay_v30_engine.jsonl",
                                                  "raw/REPLAY_LEDGER.jsonl", "results/exit_correction_impact.csv", "results/exit_correction_summary.json"]),
    "code/experiments/v3/replay_case.py": (["raw/manifests/D3_cal.jsonl", "raw/manifests/D4_test.jsonl", "raw/results/D3_cal.jsonl", "raw/results/D4_test.jsonl"],
                                           ["raw/counterexamples/Cec9aa9e05ba7b685.*", "raw/counterexamples/Cd8c8a1e89454a319.*"]),
    "code/experiments/v3_1/c02_contract_impact.py": (["raw/results/*.jsonl"], ["results/existing_record_contract_*"]),
    "code/experiments/v3_1/c02_worker_equivalence.py": (["raw/manifests/D*_test.jsonl", "raw/results/D*_test.jsonl"], ["results/worker_equivalence_check.*"]),
    "code/experiments/v3_1/c03_hash_reconciliation.py": (["RUN_LEDGER.jsonl", "raw/results/*.jsonl", "raw/manifests/*_summary.json"],
                                                         ["raw/legacy_hash_reconciliation.*"]),
    "code/experiments/v3_1/c04_numerical_audit.py": (["results/numerical_resolution.csv", "raw/calibration_freeze.json", "raw/manifests/D*_pilot.jsonl",
                                                      "raw/manifests/D*_dev.jsonl", "raw/results/D*_pilot.jsonl", "raw/results/D*_dev.jsonl",
                                                      "v3_1_correction/raw/results/held_exit_replay_v31.jsonl"],
                                                     ["results/numerical_resolution_v31*", "results/numerical_resolution_label_changes.csv",
                                                      "manuscript_assets/FigV31_S1_Numerics.*", "manuscript_assets/TabV31_Numerics.tex"]),
    "code/experiments/v3_1/c05_claims_v31.py": (["claims/claim_evidence_map.csv", "results/paired_comparisons.csv", "results/risk_coverage.csv",
                                                 "results/transfer_matrix.csv"], ["claims/*"]),
    "code/experiments/v3_1/c06_d1_completed_addendum.py": (["PROSPECTIVE_PLAN.json", "raw/calibration_freeze.json", "raw/manifests/D1_cal.jsonl",
                                                            "raw/results/D1_cal.jsonl", "raw/results/D1_cal_reruns.jsonl", "RUN_LEDGER.jsonl",
                                                            "results/calibration_candidates.csv", "raw/rerun_check_D1_cal.json"],
                                                           ["results/calibration_completed_D1_addendum.*"]),
    "code/experiments/v3_1/c07_rebuild.py": (["raw/manifests/*.jsonl", "raw/results/*.jsonl", "raw/calibration_freeze.json", "PROSPECTIVE_PLAN.json",
                                              "results/*.csv", "v3_1_correction/raw/results/held_exit_replay_v31.jsonl"],
                                             ["rebuilt/**", "results/CHANGED_OUTPUTS_MANIFEST.csv", "results/rebuild_log.json"]),
    "code/experiments/v3_1/c07_cost_wording.py": (["results/cost_break_even.csv"], ["results/cost_break_even_v31.csv"]),
    "code/experiments/v3_1/c07_prune.py": (["v3_1_correction/results/CHANGED_OUTPUTS_MANIFEST.csv"], ["results/CHANGED_OUTPUTS_MANIFEST.csv"]),
}


def expand(pattern):
    base = P.V3
    pat = pattern.replace("{csv,json}", "*")
    hits = sorted(glob.glob(os.path.join(base, pat.replace("/", os.sep)), recursive=True))
    return [h for h in hits if os.path.isfile(h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dep-table", required=True)
    ap.add_argument("--out", required=True, help="output folder outside the v3.0 and v3.1 trees (v3.1.1)")
    a = ap.parse_args()
    os.makedirs(P.guard_docs(os.path.join(a.out, "results")), exist_ok=True)
    head = git("rev-parse", "HEAD")
    # 1. dependency / impact table
    rows = list(csv.DictReader(open(a.dep_table, encoding="utf-8")))
    for r in rows:
        r["v31_outcome"], r["v31_evidence"] = outcome(r["defect_id"], r["item"])
    with open(P.guard_docs(os.path.join(a.out, "DEPENDENCY_IMPACT_TABLE.csv")), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # 2. exact-input index
    idx = []
    for script, (ins, outs) in SCRIPTS.items():
        commit = git("log", "-1", "--format=%h", "--", script.replace("code/", "", 1))
        for o in outs:
            for of in (expand(os.path.join("v3_1_correction", o)) or [None]):
                for ipat in ins:
                    for f in expand(ipat):
                        idx.append({"output": os.path.relpath(of, P.V3).replace("\\", "/") if of else f"v3_1_correction/{o} (not present)",
                                    "script": script, "script_last_commit": commit,
                                    "input": os.path.relpath(f, P.V3).replace("\\", "/"), "input_sha256": sha(f)})
    with open(P.guard_docs(os.path.join(a.out, "EXACT_INPUT_INDEX.csv")), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["output", "script", "script_last_commit", "input", "input_sha256"])
        w.writeheader()
        w.writerows(idx)
    # 3. cohort manifests vs D0_dev
    dev = {m["case_id"]: m for m in map(json.loads, open(os.path.join(P.MAN, "D0_dev.jsonl"), encoding="utf-8"))}
    chk = {}
    for c in ("D3", "D4", "D5"):
        man = [json.loads(x) for x in open(os.path.join(P.MAN, f"{c}_transfer_cohort.jsonl"), encoding="utf-8")]
        src = [m["source_case_id"] for m in man]
        same = all(dev[m["source_case_id"]]["draw"][k] == m["draw"][k] for m in man for k in ("rel0", "a_max", "wn_ratio"))
        s = json.load(open(os.path.join(P.MAN, f"{c}_transfer_cohort_summary.json"), encoding="utf-8"))
        chk[c] = {"n": len(man), "distinct_sources": len(set(src)), "all_sources_in_D0_dev": all(x in dev for x in src),
                  "draws_identical_rel0_amax_wn": same, "written": s["written"], "manifest_sha256": sha(os.path.join(P.MAN, f"{c}_transfer_cohort.jsonl")),
                  "manifest_sha256_in_summary": s["manifest_sha256"]}
    plan = json.load(open(os.path.join(P.V3, "PROSPECTIVE_PLAN.json"), encoding="utf-8"))
    chk["plan_frozen_at_local"] = plan["frozen_at_local"]
    chk["cohort_manifests_hash_protected_by_plan"] = any("transfer_cohort" in k for k in plan["manifests"])
    with open(P.guard_docs(os.path.join(a.out, "results", "transfer_cohort_manifest_check.json")), "w", encoding="utf-8") as fh:
        json.dump(chk, fh, indent=1)
    # 4. protocol deviation addendum
    frz = json.load(open(os.path.join(P.V3, "raw", "calibration_freeze.json"), encoding="utf-8"))
    ter = json.load(open(os.path.join(P.V3, "raw", "test_evaluation_record.json"), encoding="utf-8"))
    rer = sorted(json.loads(x)["completed_at"] for x in open(os.path.join(P.RES, "D1_cal_reruns.jsonl"), encoding="utf-8"))
    d1 = json.load(open(os.path.join(P.V31_RES, "calibration_completed_D1_addendum.json"), encoding="utf-8"))
    rec = json.load(open(os.path.join(P.V31_RAW, "legacy_hash_reconciliation.json"), encoding="utf-8"))
    add = {"addendum_to": "v3_extension/PROTOCOL_DEVIATIONS.json (unchanged)", "version": "v3.1-correction-1",
           "written": "2026-10-14", "code_head": head,
           "timeline_local": {"plan_frozen": plan["frozen_at_local"], "calibration_selected": frz["selected_at_local"],
                              "test_first_evaluated": ter["first_evaluated_local"], "D1_reruns": [rer[0], rer[-1]]},
           "deviations": [
               {"id": "DEV-01-A", "amends": "DEV-01",
                "what": "Corrections to the DEV-01 text. (1) 'the remaining 4,957 D1 calibration cases are an outcome-independent "
                        "subset ... so they remain iid draws' is an assertion the record does not support: the 43 cases were those in "
                        "flight or queued when the driver was stopped, and no test of independence from outcome was made. v3.1 does not "
                        "rely on it. (2) The re-execution ran after the test evaluation (reruns " + rer[0] + " to " + rer[-1] +
                        "; test evaluation " + ter["first_evaluated_local"] + "), so the completed-data analysis is a post-test "
                        "sensitivity. (3) Completing the 43 runs changes calibration counts: e.g. D1|A|1m accepts " +
                        str(next(x for x in d1["families"] if x["family_id"] == "D1|A|1m")["completed_selected"]["n_accepted"]) +
                        " instead of " + str(next(x for x in d1["families"] if x["family_id"] == "D1|A|1m")["original_selected"]["n_accepted"]) +
                        "; " + str(d1["n_candidate_rows_gaining_adverse_cases"]) + " candidate rows gain adverse cases (all inadmissible before "
                        "and after); no selection changes (" + str(d1["n_families_selection_changed"]) + " of 10) and no admissibility flips. "
                        "(4) The v3.1 record contract labels the 43 records 'interrupted' (operational), not numerical_failure. (5) The 43 "
                        "'rerun' ledger events carry no result hash; v3_1_correction/raw/legacy_hash_reconciliation.csv records the SHA-256 "
                        "of each stored line.",
                "effect_on_protocol": "none on the frozen selection; descriptive addendum results/calibration_completed_D1_addendum.*",
                "informed_by_labels": "the completed-data analysis was run after test labels existed; it changes no rule"},
               {"id": "DEV-02-A", "amends": "DEV-02",
                "what": "v3.1 does not use the post-test equivalence rule of DEV-02 for a claim status. The A_H vs A_naive screening "
                        "comparison (C06) is reported descriptively: near-identical decisions, no prespecified equivalence margin, "
                        "degenerate [0, 0] bootstrap intervals replaced by exact one-sided bounds on the discordance rate.",
                "effect_on_protocol": "wording/status only", "informed_by_labels": "yes (status provenance recorded)"},
               {"id": "DEV-03", "what": "v3.1 correction cycle (post-test code changes). Changed: cwvalid3/sampled.py and held_metrics.py "
                                        "(C01 exit storage, dense masking, t = 0 exit), experiments/v3/campaign.py (staged workers, canonical "
                                        "records, v3.0 tree closed, path-based engine identity), new cwvalid3/contracts.py and serialization.py, "
                                        "docstrings/guards in build_claims.py, p02_numerical_resolution.py, replay_case.py, make_package.py, "
                                        "a run-only-inside-the-rebuild-harness guard (experiments/v3/v30_closed.py) in 14 v3.0 output scripts "
                                        "(including the plan-listed evaluate_test.py and analyze_dev.py, whose code otherwise is unchanged), "
                                        "and the experiments/v3_1 scripts; changes after the independent review are rows 14-20 of the change log. "
                                        "Reason and scope per item in CHANGELOG_v3.1.md.",
                "effect_on_protocol": "no threshold, candidate set, alpha, delta, M, selection rule or test decision changed; no "
                                      "estimator was redesigned and no new threshold was chosen; the frozen sources listed in the plan "
                                      "now differ from the plan's code hashes (calibrate.py's frozen-code check refuses to re-run, as "
                                      "intended); v3.0 run_ids are not recomputed: replays are keyed by the legacy run_id and carry the "
                                      "v3.1 engine identity separately",
                "informed_by_labels": "the defects were found by an audit after the test evaluation; fixes were specified by the audit, not by label values"},
               {"id": "DEV-04", "what": "Status-rule provenance: experiments/v3/build_claims.py (claim statuses and wording) was written after "
                                        "the test evaluation (first committed 2026-10-14 06:51 +0400). Its docstring said the rules were fixed "
                                        "before the test was read; corrected. The v3.1 claim map records evidence type and status provenance.",
                "effect_on_protocol": "documentation", "informed_by_labels": "yes (claim statuses)"},
               {"id": "DEV-05", "what": "The P02 numerical-audit sample was described as 'a-priori stratified'. Its selection rule was first "
                                        "committed 2026-10-13 22:02 +0400 (e376cec), after the last pilot/development record (21:57:16); no earlier "
                                        "record of the rule exists, so it cannot be shown to precede the development labels. It "
                                        "is a rule-based stratified development audit.",
                "effect_on_protocol": "documentation; the corrected audit uses the same 108 cases", "informed_by_labels": "development labels only"},
               {"id": "DEV-06", "what": "The transfer-cohort manifests were written at " + chk["D3"]["written"] + " to " + chk["D5"]["written"] +
                                        ", after the plan was frozen (" + plan["frozen_at_local"] + "), and the plan does not hash-protect them "
                                        "(it lists the 12 calibration/test manifests). Integrity check: every cohort record maps one-to-one to a "
                                        "distinct D0 development draw with identical rel0, a_max and wn_ratio (results/transfer_cohort_manifest_check.json).",
                "effect_on_protocol": "none (the cohort is exploratory and never pooled with iid splits)", "informed_by_labels": "no"},
               {"id": "DEV-07", "what": "Engine identity of development records: " + str(sum(e["n"] for e in rec["engine_census"]
                                        if e["engine_sha_record"] == "25643c68cae0ff3b")) + " pilot/dev records were produced by engine "
                                        "25643c68cae0ff3b, whose source was not archived; the D2/D5 development records (600) were produced by "
                                        "ad3ad5cb611c3786 while their run_ids bind 25643c68cae0ff3b (" + str(rec["n_records_whose_run_id_binds_another_engine"]) +
                                        " records). The only change between ad3ad5cb and the frozen ae0c780 is a numerically inert helper.",
                "effect_on_protocol": "development records only; no calibration or test record affected", "informed_by_labels": "no"},
               {"id": "DEV-08", "what": "Record conflict: IMPLEMENTATION_REPORT section 6 lists p08_failure_atlas.py and rerun_interrupted.py as "
                                        "written or changed after the test evaluation, but their file modification times (2026-10-14 03:33:04 and "
                                        "03:32:34) precede the test evaluation (" + ter["first_evaluated_local"] + ") and follow the calibration "
                                        "selection (" + frz["selected_at_local"] + "). Modification times are weak evidence; the conflict is stated, "
                                        "not resolved.", "effect_on_protocol": "none", "informed_by_labels": "unknown"},
           ]}
    with open(P.guard_docs(os.path.join(a.out, "PROTOCOL_DEVIATIONS_v3.1_addendum.json")), "w", encoding="utf-8") as fh:
        json.dump(add, fh, indent=1)
    print(f"dependency rows {len(rows)} (without outcome: {sum(1 for r in rows if not r['v31_outcome'])}); index rows {len(idx)}; cohort {chk}")


if __name__ == "__main__":
    main()
