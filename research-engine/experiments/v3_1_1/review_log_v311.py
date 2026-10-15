"""Write v3_1_1_followup/reports/INDEPENDENT_REVIEW_LOG_v3.1.1.md from the journal of the adversarial review of the
v3.1.1 patch (5 reviewers, one per lens: R01, R02, contract and hashing, tests and fixtures, generators and scripts;
each followed by an independent verifier asked to refute every finding). Finding ids repeat across lenses, so every
finding is identified by lens and id. The disposition column says what v3.1.1 did; it is maintained here.
Usage: python review_log_v311.py --journal <journal.jsonl>
"""

import argparse
import json
import os

import v311paths as P

DISPOSITION = {
    ("R01", "R01-F1"): "fixed: _feature_checkpoint never raises; a failed write is recorded in feature_checkpoint_error (volatile)",
    ("R01", "R01-F2"): "fixed: every task that fails with a broken pool is re-executed alone in a fresh process; only a task lost again "
                       "gets crash records (cmd_run, _retry_alone; test_r02_tasks_broken_by_another_tasks_loss_complete_on_reexecution)",
    ("R01", "R01-F3"): "fixed: same as R01-F1 (the case stays ok; test_r01_checkpoint_write_failure_never_fails_the_case and the "
                       "continuous counterpart)",
    ("R01", "R01-F4"): "fixed: a failed A_naive pass is listed in feature_errors",
    ("R02", "R02-F1"): "fixed: as R01-F2; recoveries now run in parallel (up to `workers`), not serially",
    ("R02", "R02-F2"): "fixed: as R01-F1",
    ("R02", "R02-F3"): "fixed: as R01-F1",
    ("R02", "R02-F4"): "fixed: the tests register every variable with monkeypatch (setenv), so it is removed after the test",
    ("R02", "R02-F5"): "fixed: as R01-F4",
    ("R02", "R02-F6"): "fixed: review_log.py dispositions C02-F3, C02-F6 and C07-2 carry v3.1.1 notes; it writes to corrected_docs",
    ("CONTRACT", "F1"): "fixed: as R01-F1",
    ("CONTRACT", "F2"): "fixed: as R01-F4",
    ("TESTS", "F1"): "fixed: comparisons with stored records allow round-off (rel 1e-9) and require identical frozen decisions; exact "
                     "old/new identity stays in c11b (one machine)",
    ("TESTS", "F2"): "refuted by the verifier (the contract has no cell-specific logic); fixtures extended anyway with the first 10 "
                     "D1 and D4 exits and the first 10 D1 1 m violations",
    ("TESTS", "F3"): "fixed: the cached record is compared with a fresh, uncached propagation",
    ("TESTS", "F4"): "fixed: as R02-F4",
    ("DOCS_SCRIPTS", "F1"): "fixed: c05 writes the C01 uncertainty in percentage points",
    ("DOCS_SCRIPTS", "F2"): "fixed: the guards refuse UNC and extended-length paths (campaign.v30_tree_guard, v31paths, v311paths)",
    ("DOCS_SCRIPTS", "F3"): "fixed: as R02-F6",
    ("DOCS_SCRIPTS", "F4"): "fixed anyway (verdict uncertain): c08_docs needs --out and refuses the released v3.1 tree; it was not rerun",
    ("DOCS_SCRIPTS", "F5"): "fixed: c11b and c12 key records by (group, run_id); both rerun",
    ("DOCS_SCRIPTS", "F6"): "fixed: comment corrected",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", required=True)
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(a.journal, encoding="utf-8")]
    label = {r["key"]: r["label"] for r in rows if r.get("type") == "started"}
    found, verdicts = {}, {}
    for r in rows:
        if r.get("type") != "result":
            continue
        lab = label.get(r["key"], "")
        lens = lab.split(":", 1)[1] if ":" in lab else lab
        res = r.get("result") or {}
        if lab.startswith("find:"):
            found[lens] = res
        elif lab.startswith("verify:"):
            verdicts[lens] = {v["id"]: v for v in res.get("verdicts", [])}
    lines = ["# Independent review of the v3.1.1 patch", "",
             "An adversarial review ran on the first v3.1.1 implementation (git diff v3.1..922e9b2 and its outputs). Five reviewers, "
             "one per lens (R01 feature isolation; R02 process loss and recovery; contract and hashing; tests and fixtures; "
             "generators and analysis scripts), read the code, the outputs and the brief and ran small checks, read-only. Each "
             "reviewer's findings went to an independent verifier asked to refute each one. The disposition column says what "
             "v3.1.1 did; every fix is in commit 8d7f1cb or later, followed by a rerun of the affected evidence.", "",
             "| lens | id | severity (verified) | verdict | finding | disposition |", "|---|---|---|---|---|---|"]
    n = {}
    for lens in ("R01", "R02", "CONTRACT", "TESTS", "DOCS_SCRIPTS"):
        for f in found.get(lens, {}).get("findings", []):
            v = verdicts.get(lens, {}).get(f["id"], {})
            n[v.get("verdict", "not verified")] = n.get(v.get("verdict", "not verified"), 0) + 1
            t = f["title"].replace("|", "/").replace("\n", " ")
            lines.append(f"| {lens} | {f['id']} | {f['severity']} ({v.get('severity', '?')}) | {v.get('verdict', 'not verified')} | {t} "
                         f"| {DISPOSITION.get((lens, f['id']), '')} |")
    total = sum(n.values())
    lines += ["", f"Totals: {total} findings; " + ", ".join(f"{k} {v}" for k, v in sorted(n.items())) + ". Several findings "
              "describe the same defect from different lenses (the unguarded checkpoint rewrite was found by three reviewers, "
              "the pool-break collateral by two, the feature_errors wording by three).", "",
              "## Checked and found correct by the reviewers", ""]
    for lens in ("R01", "R02", "CONTRACT", "TESTS", "DOCS_SCRIPTS"):
        for c in found.get(lens, {}).get("checked_ok", []):
            lines.append(f"- ({lens}) " + c.replace("\n", " "))
    os.makedirs(P.V311_REP, exist_ok=True)
    with open(P.guard(os.path.join(P.V311_REP, "INDEPENDENT_REVIEW_LOG_v3.1.1.md")), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    missing = [(lens, f["id"]) for lens in found for f in found[lens].get("findings", []) if (lens, f["id"]) not in DISPOSITION]
    print(n, "missing dispositions:", missing)


if __name__ == "__main__":
    main()
