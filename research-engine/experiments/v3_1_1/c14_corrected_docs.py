"""v3.1.1 R03: corrected copies of the hand-written v3.1 documents.

The released v3.1 documents (v3_1_correction/reports/*.md, DEPENDENCY_IMPACT_TABLE.csv) are read, never edited. Each
corrected copy is written to v3_1_1_followup/corrected_docs with a header naming its source and every replacement
asserted to occur exactly once. CORRECTED_DOCS_DIFF.md lists each replacement (old -> new) with its reason. Relative
links are rewritten so that they resolve from the new location (to the corrected copies where one exists, else to the
v3.1 release tree). The generated claim map, cost table and snippets come from the patched generators (c05, c07,
make_snippets_v31); the PDF from report_pdf.py.

Usage: python c14_corrected_docs.py
"""

import json
import os

import v311paths as P

REL31 = "../../../v3_1_correction/"          # from corrected_docs/<sub>/ to the v3.1 release tree
NR = json.load(open(os.path.join(P.V31, "results", "numerical_resolution_v31_summary.json"), encoding="utf-8"))
UM = NR["max_uncertainty_over_margin_resolved"]
NT = open(os.path.join(P.REPO, "tests", "test_v311_failure_paths.py"), encoding="utf-8").read().count("\ndef test_")
REC = json.load(open(os.path.join(P.V311_RES, "recovery_path_summary.json"), encoding="utf-8"))
HELD_REL = max(v["max_rel_diff"] for k, v in REC["per_feature"].items() if k.startswith("held:") and v.get("max_rel_diff") is not None)
HELD_ABS = max(v["max_abs_diff"] for k, v in REC["per_feature"].items() if k.startswith("held:") and v.get("max_abs_diff") is not None)

STAGING_HELD = (
    "Held cells: the nominal and reference branches are propagated together in one combined batch, so held features are "
    "not checkpointed before reference propagation. After that batch the primary features (CW branch, A_H, B_H, C_H, "
    "saturation) are computed, written into the record and checkpointed, before the optional A_naive pass and before any "
    "reference metric. Each held feature is isolated: a failure of one estimator, of the saturation summary or of the "
    "optional A_naive comparator (its pass, case view, extraction or diagnostic) removes only that feature or diagnostic "
    "(v3.1.1 R01). If a worker process is lost, the pool breaks and every unfinished task fails with it; `cmd_run` "
    "re-executes each such task alone in a fresh process, so a task that only failed with the pool completes normally. If a "
    "task is lost again and its features were not checkpointed, `cmd_run` recomputes them in a fresh process without any "
    "reference branch (`feature_recovery = driver_nominal_recovery`); this is post-failure recovery, not pre-reference "
    "checkpointing, and the case's truth stays `interrupted`, adverse if accepted (v3.1.1 R02). A failed checkpoint write "
    "is recorded and never fails the case.")


def header(src, what):
    return (f"> **v3.1.1 corrected copy.** Source: `{REL31}{src}` (released with v3.1, unchanged). Corrections from the "
            f"follow-up brief \"CLOSE TWO REMAINING v3.1 FAILURE-PATH GAPS\" (14 Oct 2026): {what}. Each replacement is "
            f"listed in `../CORRECTED_DOCS_DIFF.md`; the closure report is `../../CLOSURE_REPORT_v3.1.1.md`. Paths in the body "
            f"are relative to the v3.1 release folder `v3_1_correction/` unless written as links from this file.\n\n")


DOCS = {
    "reports/CORRECTION_REPORT.md": ("reports/CORRECTION_REPORT_v3.1.1.md",
                                     "R01/R02 staging statements, R03.1 tolerance qualifier, R03.3 timing provenance, R03.4 percentage points", [
        ("# CEAS v3 extension, correction release v3.1: reconciliation report\n",
         "# CEAS v3 extension, correction release v3.1: reconciliation report (v3.1.1 corrected copy)\n", "title"),
        ("- **All seven mandatory items are closed.** Every audit finding reproduced. None turned out to change a headline result.",
         "- **All seven mandatory items are closed.** Every audit finding reproduced. None turned out to change a headline result. "
         "(v3.1.1: a follow-up review found two C02 failure-path gaps, R01 and R02; both are repaired in v3.1.1, with no change to "
         "any stored record or result.)", "R01/R02 status"),
        ("- `cwvalid3/contracts.py` (v3.1-contract-2) adds:",
         "- `cwvalid3/contracts.py` (v3.1-contract-2; v3.1.1: v3.1-contract-3, which adds the driver-level nominal recovery to the "
         "recovered-feature condition and is identical on every stored record) adds:", "R02 contract version"),
        ("- `campaign.py` stages the workers. Features are written into the record, and to a per-case checkpoint file that a worker "
         "crash cannot erase, before any reference work starts. Held cells get their features from a CW + estimator batch if the "
         "combined batch fails, and the A_naive pass can fail on its own.",
         "- `campaign.py` stages the workers. Continuous cells: the features are written into the record, and to a per-case "
         "checkpoint file, before any reference work starts. " + STAGING_HELD + " Held cells get their features from a CW + "
         "estimator batch if the combined batch fails.", "R03.5: no unconditional held 'before any reference work'"),
        ("  - feature checkpoint recovery after a crash;",
         "  - feature checkpoint recovery after a crash;\n"
         f"  - v3.1.1 (`tests/test_v311_failure_paths.py`, {NT} tests): failures of the A_naive pass, case view, extraction and "
         "diagnostic, of saturation, of a primary estimator and of the CW branch; real worker-process loss (os._exit) after the "
         "combined batch and after the primary features, re-execution of a task broken only by another task's loss, driver-level "
         "recovery, recovery failure, stale and wrong-run checkpoints, failed checkpoint writes, and the normal path through "
         "`cmd_run`;", "R01/R02 regressions"),
        ("  - 92 completed cases resolve at 1 m and 0.1 m; the largest uncertainty-to-margin ratio is 9.9e-7.",
         f"  - 92 completed cases resolve at 1 m and 0.1 m; the largest uncertainty-to-margin ratio among resolved cases is "
         f"{UM['1m']:.1e} at the 1 m tolerance and {UM['0.1m']:.1e} at 0.1 m.", "R03.1 tolerance qualifier"),
        ("  - The held rows exclude referral work and are single medians.",
         "  - The held rows exclude referral work; each comes from one 40-case batch per held cell, summarised as the median of "
         "three repetitions, with no dispersion archived.", "R03.3 timing provenance"),
        ("  - The held rows are screen-only costs that exclude referral work, so they are lower bounds. Each is a single median of 3 "
         "repetitions of one 40-case batch, and no dispersion was archived.",
         "  - The held rows are screen-only costs that exclude referral work, so they are lower bounds. Each comes from one 40-case "
         "batch per held cell, summarised as the median of three repetitions; no dispersion was archived.", "R03.3 timing provenance"),
        ("- the primary contrast: D0 at 1 m, A 1,645 vs B 1,421 accepted, +8.96 points, 95 % interval 7.80–10.12, 0 vs 7 adverse.",
         "- the primary contrast: D0 at 1 m, A 1,645 vs B 1,421 accepted, +8.96 percentage points, 95 % paired bootstrap interval "
         "7.80–10.12 percentage points, 0 vs 7 adverse.", "R03.4 percentage points"),
        ("7. **No external validation.** No new external validation (GMAT, flight data) was done, and none is claimed.",
         "7. **No external validation.** No new external validation (GMAT, flight data) was done, and none is claimed.\n"
         "8. **Held failure-path guarantee (v3.1.1).** Held features are checkpointed after the combined nominal + reference "
         "propagation, not before it. A worker-process loss during that propagation is covered by post-failure nominal-only "
         f"recovery, whose held feature values can differ from the combined batch in the last digits (at most {HELD_REL:.1e} "
         f"relative, {HELD_ABS:.1e} m, on 50 held cases; no frozen decision changed). Neither failure type occurred in the archived "
         "campaign; the repairs are tested on injected failures.\n"
         "9. **Corrections are post-test (v3.1.1).** Like the v3.1 corrections, the v3.1.1 code changes were made after the test "
         "evaluation (`../../PROTOCOL_DEVIATIONS_v3.1.1_addendum.json`). They change no stored record, threshold or decision.",
         "R03.6 limitations"),
        ("That's roughly 2.3 CPU-h in total, for a cumulative 45–46 CPU-h, within the ceiling. No paid or cloud compute was used.",
         "That's roughly 2.3 CPU-h in total, for a cumulative 45–46 CPU-h, within the ceiling. No paid or cloud compute was used. "
         "The v3.1.1 follow-up added well under 1 CPU-h (see the closure report).", "compute note"),
    ]),
    "reports/NONCOMPUTATIONAL_HANDOFF_v3.1.md": ("reports/NONCOMPUTATIONAL_HANDOFF_v3.1.1.md",
                                                 "R03.2 evidence file, R03.3 timing provenance, R03.4 percentage points, held staging wording, links", [
        ("# Noncomputational handoff, v3.1 correction release\n", "# Noncomputational handoff, v3.1 correction release (v3.1.1 corrected copy)\n", "title"),
        ("- the primary D0 1 m contrast (A 1,645 vs B 1,421 accepted, 0 vs 7 adverse, +8.96 points, 95 % paired interval 7.80–10.12);",
         "- the primary D0 1 m contrast (A 1,645 vs B 1,421 accepted, 0 vs 7 adverse, +8.96 percentage points, 95 % paired bootstrap "
         "interval 7.80–10.12 percentage points);", "R03.4 percentage points"),
        ("The rebuilt test-result tables are byte-identical; see `../results/CHANGED_OUTPUTS_MANIFEST.csv`.",
         f"The rebuilt test-result tables are byte-identical; see `{REL31}results/CHANGED_OUTPUTS_MANIFEST.csv`.", "link"),
        ("Use `handoff_snippets_v3.1.tex` in place of the v3.0 result snippets [R1]–[R4].",
         "Use `handoff_snippets_v3.1.1.tex` (v3.1.1: held timing provenance in [R2]) in place of the v3.0 result snippets [R1]–[R4].",
         "R03.3 snippets"),
        ("Held cells: \"in a single timing of one 40-case batch, the screen alone (referrals excluded) took slightly longer than direct "
         "propagation (0.268 vs 0.237 s and 0.135 vs 0.125 s per case)\"; no dispersion was archived, so don't present it as a precise "
         "comparison (`../results/cost_break_even_v31.csv`)",
         "Held cells: \"in one 40-case batch per held cell, timed as the median of three repetitions, the screen alone (referrals "
         "excluded) took slightly longer than direct propagation (0.268 vs 0.237 s and 0.135 vs 0.125 s per case)\"; no dispersion was "
         "archived, so don't present it as a precise comparison (`../results/cost_break_even_v31.csv`)", "R03.3 timing provenance"),
        ("- `../results/numerical_resolution_v31.csv`;\n- `../results/numerical_resolution_decisions_v31.json`;\n"
         "- `../manuscript_assets/FigV31_S1_Numerics.*` and `TabV31_Numerics.tex`, which replace FigV3_S1 and TabV3_Numerics;\n"
         "- `C04_NUMERICAL_AUDIT_NOTES.md`.",
         f"- `{REL31}results/numerical_resolution_v31.csv` (per case);\n"
         f"- `{REL31}results/numerical_resolution_decisions_v31.csv` (decisions at the frozen thresholds) and "
         f"`{REL31}results/numerical_resolution_v31_summary.json` (summary, including the uncertainty-to-margin ratios);\n"
         f"- `{REL31}manuscript_assets/FigV31_S1_Numerics.*` and `TabV31_Numerics.tex`, which replace FigV3_S1 and TabV3_Numerics;\n"
         f"- `{REL31}reports/C04_NUMERICAL_AUDIT_NOTES.md`.", "R03.2 evidence file"),
        ("Evidence: `../results/exit_correction_summary.json` and claim C18. If held exit discrepancies are shown anywhere (failure atlas, "
         "traces), use the rebuilt files in `../rebuilt/` and the regenerated traces in `../raw/counterexamples/`.",
         f"Evidence: `{REL31}results/exit_correction_summary.json` and claim C18. If held exit discrepancies are shown anywhere (failure "
         f"atlas, traces), use the rebuilt files in `{REL31}rebuilt/` and the regenerated traces in `{REL31}raw/counterexamples/`.", "links"),
        ("- **Deviations.** Cite the v3.1 deviation addendum (`../PROTOCOL_DEVIATIONS_v3.1_addendum.json`) alongside DEV-01 and DEV-02:",
         f"- **Deviations.** Cite the v3.1 deviation addendum (`{REL31}PROTOCOL_DEVIATIONS_v3.1_addendum.json`) and the v3.1.1 addendum "
         "(`../../PROTOCOL_DEVIATIONS_v3.1.1_addendum.json`, post-test failure-path code changes) alongside DEV-01 and DEV-02:", "links"),
        ("Use `../claims/claim_evidence_map_v31.csv`.",
         "Use `../claims/claim_evidence_map_v31.csv` (v3.1.1 regeneration: coverage differences in percentage points, held timing "
         f"provenance; otherwise identical to `{REL31}claims/claim_evidence_map_v31.csv`).", "R03.4 claim map"),
        ("- All code and analysis came from one AI-assisted workflow. This was already listed in v3.0; it applies equally to the corrections.",
         "- All code and analysis came from one AI-assisted workflow. This was already listed in v3.0; it applies equally to the corrections.\n"
         "- Worker staging (only if the manuscript or supplement describes it): continuous workers store and checkpoint the screening "
         "features before any reference propagation. Held workers propagate the nominal and reference branches together, then store and "
         "checkpoint the features before the optional comparator and any reference metric; if a worker process is lost before that, the "
         "features are recomputed afterwards without reference branches and the case stays 'interrupted' (adverse if accepted). Don't "
         "write that held features are computed before any reference work.\n"
         "- The v3.1 and v3.1.1 code corrections were made after the test evaluation; they change no stored record, threshold or decision.",
         "R03.5/R03.6"),
        ("In addition, decide whether to mention the correction release in the Online Resource. If it is mentioned, cite "
         "`CORRECTION_REPORT.md` and the review ZIP.",
         "In addition, decide whether to mention the correction release in the Online Resource. If it is mentioned, cite "
         "`CORRECTION_REPORT_v3.1.1.md` (or the released v3.1 report with the v3.1.1 closure report) and the review ZIPs.", "links"),
    ]),
    "reports/RECORD_CONTRACT_AND_HASHING_v3.1.md": ("reports/RECORD_CONTRACT_AND_HASHING_v3.1.1.md",
                                                    "R01/R02 staging and recovery, contract v3.1-contract-3, fixture-backed tests", [
        ("# v3.1 record contract (C02) and canonical hashing (C03)\n",
         "# v3.1 record contract (C02) and canonical hashing (C03) (v3.1.1 corrected copy)\n", "title"),
        ("- `code/cwvalid3/contracts.py`: the contract (`v3.1-contract-2`).",
         "- `code/cwvalid3/contracts.py`: the contract (`v3.1-contract-3` since v3.1.1; `v3.1-contract-2` in v3.1).", "R02 contract version"),
        ("- `code/experiments/v3/campaign.py`: staged workers, feature checkpoints, the canonical writer and `status`.",
         "- `code/experiments/v3/campaign.py`: staged workers, feature checkpoints, the driver-level nominal recovery (v3.1.1), the "
         "canonical writer and `status`.", "R02"),
        ("Tests: `code/tests/test_v31_contracts.py` and `code/tests/test_v31_serialization.py`. The run log is in "
         "`tests_logs/pytest_v3_and_v31.txt`.",
         "Tests: `code/tests/test_v31_contracts.py`, `code/tests/test_v31_serialization.py` and (v3.1.1) "
         "`code/tests/test_v311_failure_paths.py` and `code/tests/test_v31_fixtures.py`; since v3.1.1 the data-backed tests read exact "
         "archive lines from `code/tests/fixtures/v31` and never skip. Run logs: `../../tests_logs/` (v3.1.1) and "
         f"`{REL31}tests_logs/pytest_v3_and_v31.txt` (v3.1).", "R04 tests"),
        ("| unavailable, unless the crash record carries the case's feature checkpoint (`feature_checkpoint_recovered`) |",
         "| unavailable, unless its features were recovered: merged from a checkpoint of the same run session and engine "
         "(`feature_checkpoint_recovered`), or recomputed by the driver-level nominal-only recovery without reference branches "
         "(`feature_recovery = driver_nominal_recovery`, v3.1.1) |", "R02 contract"),
        ("**Feature availability** depends only on three things:",
         "Per-feature failures inside a completed worker (v3.1.1) are carried by the method's own status field "
         "(`feature_extraction_failed`, `naive_feature_failed`, `naive_pass_failed`) and listed in `feature_errors`, which is diagnostic and "
         "does not change truth; the ok-with-error rule above applies to `error`, `reference_error` and `failed_stage`.\n\n"
         "**Feature availability** depends only on three things:", "R01 contract"),
        ("The nominal CW branch and the screening features are computed **before** any reference or cross-check work. They are written "
         "into the record and into a per-case checkpoint file in `<out>/results/feature_checkpoints/`.",
         "Continuous cells compute the nominal CW branch and the screening features **before** any reference or cross-check work. "
         + STAGING_HELD + " Features are written into the record and into a per-case checkpoint file in "
         "`<out>/results/feature_checkpoints/`; the checkpoint carries the run session and the engine identity.", "R03.5 staging"),
        ("- **Worker process dies.** `cmd_run` merges the case's checkpoint into its `worker_crash` record. The checkpoint is deleted once "
         "the full record is stored.",
         "- **Worker process dies.** The pool breaks and every unfinished task fails with it, also tasks that never started. After the "
         "main pool, `cmd_run` re-executes each such task alone in a fresh process (`task_retry`, volatile); only a task lost again "
         "gets `worker_crash` records (v3.1.1). `cmd_run` merges the case's checkpoint into its `worker_crash` record only if it names "
         "the case's run_id, is complete, and was written in the same run session by the same engine; anything else is ignored "
         "(`stale_checkpoint_ignored`). Cases without a valid checkpoint get the nominal-only recovery in a fresh process "
         "(`feature_recovery = driver_nominal_recovery`, or `failed`). A failed checkpoint write is recorded in "
         "`feature_checkpoint_error` (volatile) and never fails the case. The checkpoint is deleted once the full record is stored.",
         "R02 recovery"),
        ("  - The A_naive pass is a separate CW-only batch. If it fails, only A_naive is unavailable (`naive_pass_failed`).",
         "  - The primary features are computed per case with each part isolated and checkpointed before the A_naive pass (v3.1.1).\n"
         "  - The A_naive pass is a separate CW-only batch. If it, its case view or its extraction fails, only A_naive is unavailable "
         "(`naive_pass_failed`, `naive_feature_failed`); if the naive-pass diagnostic fails, only the diagnostic is blank (v3.1.1).\n"
         "  - The reference metrics are computed separately from the features (`_held_reference_fields`), so a feature failure cannot "
         "fail them (v3.1.1).", "R01"),
        ("  - The feature checkpoint is written before the reference stage and recovered into a crash record.",
         "  - Continuous: the feature checkpoint is written before the reference stage and recovered into a crash record.\n"
         "  - v3.1.1, held (`test_v311_failure_paths.py`): A_naive pass, case-view, extraction and diagnostic failures keep A_H, B_H, C_H, "
         "sep, sat, the frozen decisions and the truth; a saturation failure removes only sat; a primary A_H failure refers A_H only; a CW "
         "failure refers everything; a real worker-process loss after the combined batch gets driver-level recovery (features available, "
         "truth interrupted, an accepted case adverse); a loss after the primary features merges this run's checkpoint; a failed recovery "
         "leaves features unavailable; stale, wrong-run, other-engine, incomplete and unreadable checkpoints are never merged; a task "
         "broken only by another task's loss completes normally when re-executed; failed checkpoint writes leave the case ok; the normal "
         "path through `cmd_run` reproduces the stored record (to round-off, with the same frozen decisions).", "R01/R02 tests"),
        ("the partition identity, and agreement with v3.0 on 2,000 stored records.",
         "the partition identity, and agreement with v3.0 on stored calibration records (v3.1.1: the fixture set, 40 per cell, all 43 "
         "D1 interruptions, the first 10 D1 and D4 exits and the first 10 D1 1 m violations; v3.1 used the first 400 records per cell; "
         f"the full scan of all 47,983 is `{REL31}results/existing_record_contract_impact.csv`).", "R04 fixture-backed test"),
        ("One complete C01 replay chunk (31 D3 exit cases, same membership and order) re-executed on the final engine reproduces the stored "
         "replay records' physics hashes.",
         "One complete C01 replay chunk (31 D3 exit cases, same membership and order) re-executed on the final engine reproduces the stored "
         "replay records' physics hashes. v3.1.1: the v3.1 and v3.1.1 workers give identical records (every non-volatile field) on the same "
         "sample plus D4 test 0-1 and D4 dev 56 (`../../results/normal_path_old_new.csv`), and contract-3 equals contract-2 on all 48,665 "
         "stored records (`../../results/contract3_vs_contract2_scan.json`).", "normal-path equivalence"),
    ]),
}


def main():
    diff = ["# Corrected documents: replacement list (v3.1.1)", "",
            "Generated by `code/experiments/v3_1_1/c14_corrected_docs.py`. Each corrected copy is the released v3.1 document with the "
            "replacements below (each asserted to occur exactly once) and a header. Generated documents (claim map, cost table, snippets) "
            "come from the patched generators; their diffs against the v3.1 copies are in `GENERATED_DOCS_DIFF.md`.", ""]
    for src, (dst, what, reps) in DOCS.items():
        s = open(os.path.join(P.V31, src.replace("/", os.sep)), encoding="utf-8").read().replace("\r\n", "\n")
        diff += [f"## {dst}", f"source `{REL31}{src}`", ""]
        for old, new, why in reps:
            assert s.count(old) == 1, (src, old[:100])
            s = s.replace(old, new)
            diff += [f"- **{why}**", f"  - v3.1: {old.strip()}", f"  - v3.1.1: {new.strip()}"]
        diff.append("")
        lines = s.split("\n", 1)
        out = lines[0] + "\n\n" + header(src, what) + lines[1].lstrip("\n")
        p = os.path.join(P.V311_DOCS, dst.replace("/", os.sep))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(P.guard(p), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(out)
        print("wrote", p)
    # DEPENDENCY_IMPACT_TABLE.csv: the C07 cost row's outcome text (same wording as the patched c08 source)
    s = open(os.path.join(P.V31, "DEPENDENCY_IMPACT_TABLE.csv"), encoding="utf-8", newline="").read()
    old = "held values qualified as single medians (no dispersion archived)"
    new = "held values qualified as screen-only costs of one 40-case batch per held cell, median of 3 repetitions (no dispersion archived)"
    assert s.count(old) == 1
    p = os.path.join(P.V311_DOCS, "DEPENDENCY_IMPACT_TABLE.csv")
    with open(P.guard(p), "w", encoding="utf-8", newline="") as fh:
        fh.write(s.replace(old, new))
    diff += ["## DEPENDENCY_IMPACT_TABLE.csv", f"source `{REL31[3:]}DEPENDENCY_IMPACT_TABLE.csv`", "",
             "- **R03.3 timing provenance** (C07 row, outcome column)", f"  - v3.1: {old}", f"  - v3.1.1: {new}", ""]
    with open(P.guard(os.path.join(P.V311_DOCS, "CORRECTED_DOCS_DIFF.md")), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(diff) + "\n")
    print("wrote CORRECTED_DOCS_DIFF.md")
    # generated documents (from the patched generators): unified diffs against the released v3.1 copies
    import difflib
    g = ["# Generated documents: diffs against the released v3.1 copies (v3.1.1)", "",
         "The files below were regenerated by the patched generators (c05_claims_v31.py, c07_cost_wording.py, make_snippets_v31.py, "
         "review_log.py with the v3.1 review journal) "
         "into corrected_docs/. Unified diffs, line endings normalised; every other line is identical to the v3.1 copy.", ""]
    for old, new in (("claims/claim_evidence_map_v31.csv", "claims/claim_evidence_map_v31.csv"),
                     ("claims/claim_status_provenance.csv", "claims/claim_status_provenance.csv"),
                     ("claims/CLAIM_WORDING_DIFF.md", "claims/CLAIM_WORDING_DIFF.md"),
                     ("results/cost_break_even_v31.csv", "results/cost_break_even_v31.csv"),
                     ("reports/handoff_snippets_v3.1.tex", "reports/handoff_snippets_v3.1.1.tex"),
                     ("reports/INDEPENDENT_REVIEW_LOG.md", "reports/INDEPENDENT_REVIEW_LOG.md")):
        a = open(os.path.join(P.V31, old.replace("/", os.sep)), encoding="utf-8").read().replace("\r\n", "\n").splitlines()
        b = open(os.path.join(P.V311_DOCS, new.replace("/", os.sep)), encoding="utf-8").read().replace("\r\n", "\n").splitlines()
        d = list(difflib.unified_diff(a, b, f"v3_1_correction/{old}", f"corrected_docs/{new}", n=0, lineterm=""))
        g += [f"## {new}", "", f"{sum(1 for x in d if x.startswith('-') and not x.startswith('---'))} line(s) replaced", "", "```diff"] + d + ["```", ""]
    with open(P.guard(os.path.join(P.V311_DOCS, "GENERATED_DOCS_DIFF.md")), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(g) + "\n")
    print("wrote GENERATED_DOCS_DIFF.md")


if __name__ == "__main__":
    main()
