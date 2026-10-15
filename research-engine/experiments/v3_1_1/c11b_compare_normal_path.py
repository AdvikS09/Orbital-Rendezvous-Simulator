"""v3.1.1: compare the normal-path records of the v3.1 release code (tag v3.1) and the v3.1.1 code on the c11 sample.

Inputs: results/normal_path/records_v3.1_tag.jsonl and records_v3.1.1.jsonl (written by c11_normal_path.py), the
stored v3.0 records and the stored C01 replay records (read-only). Three checks per case:
  old_vs_new    every non-volatile field of the v3.1 and the v3.1.1 record, including the v3.1-only fields
                (physics_sha256 and the list of differing fields);
  new_vs_stored the v3.1.1 record against the stored v3.0 record (legacy_comparable_sha256: volatile and v3.1-only
                fields excluded, as in experiments/v3_1/c02_worker_equivalence.py); held exit cases are compared
                against the C01 replay record instead (physics_sha256), because v3.1 changed exits by design;
  decisions     accept/refer of every frozen selection (calibration_freeze.json) for both records.
Writes results/normal_path_old_new.csv and normal_path_old_new_summary.json.
"""

import csv
import json
import os

import v311paths as P
from cwvalid3 import contracts as C
from cwvalid3 import serialization as SER

D = os.path.join(P.V311_RES, "normal_path")


def load(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def diff_fields(a, b, skip=()):
    a, b = SER.normalize(a), SER.normalize(b)
    ka = {k for k in a if not SER.is_volatile(k) and k not in skip}
    kb = {k for k in b if not SER.is_volatile(k) and k not in skip}
    return sorted(k for k in ka & kb if a[k] != b[k]), sorted(ka - kb), sorted(kb - ka)


def main():
    old = {(r["c11_group"], r["run_id"]): r for r in load(os.path.join(D, "records_v3.1_tag.jsonl"))}   # D4 test 0-1 occur in two groups
    new = load(os.path.join(D, "records_v3.1.1.jsonl"))
    meta = {k: json.load(open(os.path.join(D, f"records_{k}.jsonl.meta.json"))) for k in ("v3.1_tag", "v3.1.1")}
    sel = json.load(open(os.path.join(P.V3, "raw", "calibration_freeze.json"), encoding="utf-8"))["selection"]
    stored = {}
    for c in ("D0", "D1", "D2", "D3", "D4", "D5"):
        for s in ("pilot", "dev", "cal", "test", "transfer_cohort"):
            p = os.path.join(P.RES, f"{c}_{s}.jsonl")
            if os.path.exists(p):
                want = {r["run_id"] for r in new if r["cell"] == c and r["split"] == s}
                if want:
                    for r in load(p):
                        if r["run_id"] in want:
                            stored[r["run_id"]] = r
    replay = {r["replay"]["of_run_id"]: r for r in load(os.path.join(P.V31, "raw", "results", "held_exit_replay_v31.jsonl"))}
    rows = []
    for n in new:
        o = old[(n["c11_group"], n["run_id"])]
        kind = C.kind_of({"cell": n["cell"]})
        d_on, only_new, only_old = diff_fields(o, n, skip=("c11_group",))
        s = stored[n["run_id"]]
        exit_case = s.get("eci_status") in ("domain_exit", "terminated") and kind == "held"
        if exit_case and n["run_id"] in replay:
            ref, how = replay[n["run_id"]], "C01 replay record (physics_sha256)"
            d_ns, _, miss = diff_fields(n, ref, skip=("c11_group",))
            eq = SER.physics_sha256({k: v for k, v in n.items() if k != "c11_group"}) == SER.physics_sha256(ref)
        else:
            ref, how = s, "stored v3.0 record (legacy_comparable_sha256)"
            d_ns, _, miss = diff_fields(n, ref, skip=("c11_group",) + tuple(SER.V31_ONLY_FIELDS))
            eq = SER.legacy_comparable_sha256({k: v for k, v in n.items() if k != "c11_group"}) == SER.legacy_comparable_sha256(ref)
        flips = 0
        for key, x in sel.items():
            cell, meth, crit = key.split("|")
            if cell != n["cell"] or x.get("threshold") is None:
                continue
            va, sa, _ = C.feature_availability(o, meth, kind)
            vb, sb, _ = C.feature_availability(n, meth, kind)
            flips += (sa == "ok" and va <= x["threshold"]) != (sb == "ok" and vb <= x["threshold"])
        rows.append({"group": n["c11_group"], "cell": n["cell"], "split": n["split"], "index": n["index"], "case_id": n["case_id"],
                     "worker_status_old": o["worker_status"], "worker_status_new": n["worker_status"],
                     "old_new_physics_equal": SER.physics_sha256({k: v for k, v in o.items() if k != "c11_group"}) ==
                     SER.physics_sha256({k: v for k, v in n.items() if k != "c11_group"}),
                     "old_new_diff_fields": ";".join(d_on), "only_new_fields": ";".join(only_new), "only_old_fields": ";".join(only_old),
                     "new_vs": how, "new_vs_equal": eq, "new_vs_diff_fields": ";".join(d_ns), "new_vs_missing_fields": ";".join(miss),
                     "decision_flips_old_new": flips})
    with open(P.guard(os.path.join(P.V311_RES, "normal_path_old_new.csv")), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = {"n_cases": len(rows), "n_old_new_physics_equal": sum(r["old_new_physics_equal"] for r in rows),
            "n_old_new_with_field_diff": sum(bool(r["old_new_diff_fields"]) for r in rows),
            "n_only_new_or_old_fields": sum(bool(r["only_new_fields"] or r["only_old_fields"]) for r in rows),
            "n_new_vs_reference_equal": sum(r["new_vs_equal"] for r in rows),
            "n_compared_to_stored_v30": sum(r["new_vs"].startswith("stored") for r in rows),
            "n_compared_to_c01_replay": sum(r["new_vs"].startswith("C01") for r in rows),
            "n_decision_flips_old_new": sum(r["decision_flips_old_new"] for r in rows),
            "all_ok": all(r["worker_status_new"] == "ok" == r["worker_status_old"] for r in rows),
            "engine_old": meta["v3.1_tag"]["engine_sha"], "engine_new": meta["v3.1.1"]["engine_sha"],
            "runs": meta, "groups": sorted({r["group"] for r in rows})}
    with open(P.guard(os.path.join(P.V311_RES, "normal_path_old_new_summary.json")), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "runs"}, indent=1))
    for r in rows:
        if not (r["old_new_physics_equal"] and r["new_vs_equal"]) or r["only_new_fields"] or r["only_old_fields"]:
            print(r)


if __name__ == "__main__":
    main()
