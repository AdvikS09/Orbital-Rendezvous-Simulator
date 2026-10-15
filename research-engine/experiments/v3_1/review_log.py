"""Write INDEPENDENT_REVIEW_LOG.md from the journal of the adversarial review workflow (5 reviewers, one per item
group, each followed by an independent verifier asked to refute every finding). The disposition column records what
v3.1 did with each finding; it is maintained here, next to the finding ids. v3.1.1: three dispositions carry a
follow-up note (held checkpoint timing, held timing provenance), and the log is written into --out (default
v3_1_1_followup/corrected_docs/reports; the released v3.1 log is kept unchanged).
Usage: python code/experiments/v3_1/review_log.py --journal <journal.jsonl> [--out DIR]
"""

import argparse
import json
import os

import v31paths as P

DISPOSITION = {
    "C01-E1": "guarded anyway: every v3.0 writer now runs only inside the rebuild harness (experiments/v3/v30_closed.py)",
    "C01-E2": "fixed: same guard for p09_cost, verify_*, compare_hybrid_map, similarity_checks, rerun_interrupted, replay_baseline",
    "C01-E3": "fixed: tests for numerical-failure freezing, failed root bracket, saturated-gain exit under step halving",
    "C01-E4": "fixed: test of the dense state at the event against an adaptive solution; docstring corrected",
    "C01-E5": "fixed: report and claim C18 name the 1-ulp C_H re-batching difference in 3 records",
    "C01-E6": "changed anyway: exit at t = 0 has its own sentinel (EXIT_AT_T0 = -2); status stays authoritative",
    "C02-F1": "addressed: contracts.evaluate_rule (partition-checked) for v3.1 analyses; the frozen calibration.evaluate_rule is kept as the definition behind the frozen results, and this is stated",
    "C02-F2": "addressed: experiments/v3_1/v31data.py loads any root with the v3.1 contract (tested with a reference_failed record)",
    "C02-F3": "fixed: held features come from a CW + estimator batch when the combined batch or its case view fails; the A_naive pass fails separately "
              "(v3.1.1: a failure of the A_naive case view, extraction or diagnostic, or of one primary feature, could still discard the other "
              "features; fixed in v3.1.1 R01)",
    "C02-F4": "fixed: horizon checked against the planned horizon; 'ok' records with an error, failed stage or branch failure flag are unresolved (contract v3.1-contract-2; stored records unchanged)",
    "C02-F5": "fixed: equivalence check re-run at the committed code (0 differing fields, hashes equal)",
    "C02-F6": "fixed: feature-stage checkpoint merged into worker_crash records; continuous workers write it before reference work "
              "(v3.1.1: held workers write it only after the combined nominal + reference batch, so a process loss during that batch left no "
              "checkpoint; v3.1.1 R02 adds a driver-level nominal-only recovery and validates checkpoints by run session and engine)",
    "C02-F7": "documented: worker 'exception' stays a numerical failure as in the frozen specification; failed_stage records where it occurred",
    "C03-F1": "fixed: equivalence evidence regenerated at the committed code; physics hash versioned (PHYSICS_HASH_VERSION)",
    "C03-F2": "fixed: case-normalised real-path guards everywhere, with mixed-case tests",
    "C03-F3": "fixed: legacy_comparable_sha256 (v3.1-only fields removed) for comparisons with v3.0 records; wording corrected",
    "C03-F4": "recorded anyway: held records store batch_members_sha256 (volatile provenance)",
    "C03-F5": "fixed: 0-d arrays handled; str/bytes/object arrays and extended-precision floats rejected; tests",
    "C03-F6": "fixed: round trips on real record classes, legacy rule with its exceptions, case_id/physical_id/run_id formulas, relocation test",
    "C03-F7": "fixed: campaign status reconciles --out and checks every stored-line hash",
    "C03-F8": "fixed: the reconstructed ledger hash is reported separately (30,324 direct + 1 reconstructed)",
    "C04-N1": "fixed: arrival compared on completed cases only (92/92 agree; 16 exits not applicable)",
    "C04-N2": "fixed: partial-horizon columns for exits; decision margins and status per audited family in the main table",
    "C04-N3": "fixed: numerics summarised by family and estimand (A 4.5e-6, A_H 2.3e-10, A_naive 2.4e-3)",
    "C04-N4": "fixed: exact stored values; the D5 development case is an exact tie (tie_unresolved)",
    "C04-N5": "fixed: change/margin ratio with matching denominators",
    "C04-N6": "fixed: p02 docstring",
    "C04-N7": "fixed: scope note in the dependency table (37 held rows)",
    "C04-N8": "fixed: figure range",
    "C05-1": "fixed: C15 (and C12, C13) no longer carry a literal 'supported'; provenance states the absence of a rule",
    "C05-2": "wording made explicit anyway (C18)",
    "C05-3": "fixed: wording diff compares every column",
    "C05-4": "fixed: C04 and C06 wording",
    "C05-5": "fixed: T1_heuristic named for D3/D4; status from the observed adverse count",
    "C06-1": "no change (the file name, JSON and DEV-01-A label the analysis post-test)",
    "C07-1": "fixed: TabV3_Numerics / FigV3_S1 marked superseded; figure count corrected",
    "C07-2": "fixed: 'each timed feature (A, B, C)'; held timings qualified as one 40-case batch per held cell, median of three "
             "repetitions, no archived dispersion (wording of v3.1.1)",
    "C07-3": "strengthened anyway: the harness artefact is compared field by field and its regeneration entry kept in the manifest",
    "C07-4": "fixed: final baseline check run; packaging refuses to run without it",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", required=True)
    ap.add_argument("--out", default=os.path.join(P.V311_DOCS, "reports"))
    a = ap.parse_args()
    reviews, verdicts = [], {}
    for line in open(a.journal, encoding="utf-8"):
        e = json.loads(line)
        if e.get("type") != "result":
            continue
        r = e.get("result") or e.get("value")
        if isinstance(r, dict) and "findings" in r:
            reviews.append(r)
        elif isinstance(r, dict) and "verdicts" in r:
            for v in r["verdicts"]:
                verdicts[v["id"]] = v
    lines = ["# Independent review of the v3.1 corrections", "",
             "An adversarial review ran after the first v3.1 implementation (commit 3a6430f and its outputs). Five reviewers, "
             "one per item group (C01; C02; C03; C04; C05-C07), each read the code, the outputs and the brief, and ran small "
             "checks. They were told not to write anywhere in the archive and not to exceed about two CPU-minutes per check. "
             "Each reviewer's findings then went to an independent verifier, who was asked to refute each one. The "
             "disposition column says what v3.1 did; every fix is in commit 4e4a83b or later and was followed by a full "
             "regeneration of the affected outputs.", "",
             "| id | severity | verdict | finding | disposition |", "|---|---|---|---|---|"]
    n = {"confirmed": 0, "refuted": 0, "uncertain": 0}
    for r in reviews:
        for f in r["findings"]:
            v = verdicts.get(f["id"], {}).get("verdict", "not verified")
            n[v] = n.get(v, 0) + 1
            summ = f["summary"].replace("|", "/").replace("\n", " ")
            lines.append(f"| {f['id']} | {f['severity']} | {v} | {summ} | {DISPOSITION.get(f['id'], '')} |")
    lines += ["", f"Totals: {sum(len(r['findings']) for r in reviews)} findings; " + ", ".join(f"{k} {v}" for k, v in n.items() if v) + ".",
              "", "## Checked and found correct by the reviewers", ""]
    for r in reviews:
        for c in r.get("checked_ok", []):
            lines.append("- " + c.replace("\n", " "))
    os.makedirs(P.guard_docs(a.out), exist_ok=True)
    with open(P.guard_docs(os.path.join(a.out, "INDEPENDENT_REVIEW_LOG.md")), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(n, "missing dispositions:", [f["id"] for r in reviews for f in r["findings"] if f["id"] not in DISPOSITION])


if __name__ == "__main__":
    main()
