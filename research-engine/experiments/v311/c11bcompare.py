import csv
import json
import os

import v311paths as P
from cwvalid3 import contracts as C
from cwvalid3 import serialization as SER

D = os.path.join(P.V311RES, "normalpath")


def load(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def difffields(a, b, skip=()):
    a, b = SER.normalize(a), SER.normalize(b)
    ka = {k for k in a if not SER.volat(k) and k not in skip}
    kb = {k for k in b if not SER.volat(k) and k not in skip}
    return sorted(k for k in ka & kb if a[k] != b[k]), sorted(ka - kb), sorted(kb - ka)


def main():
    old = {(r["c11_group"], r["run_id"]): r for r in load(os.path.join(D, "records31.jsonl"))}
    new = load(os.path.join(D, "records311.jsonl"))
    meta = {k: json.load(open(os.path.join(D, f"{f}.jsonl.meta.json"))) for k, f in (("v3.1_tag", "records31"), ("v3.1.1", "records311"))}
    sel = json.load(open(os.path.join(P.V3, "raw", "freeze.json"), encoding="utf-8"))["selection"]
    stored = {}
    for c in ("D0", "D1", "D2", "D3", "D4", "D5"):
        for s in ("pilot", "dev", "cal", "test", "transfer_cohort"):
            p = os.path.join(P.RES, f"{c}{s.replace('_', '')}.jsonl")
            if os.path.exists(p):
                want = {r["run_id"] for r in new if r["cell"] == c and r["split"] == s}
                if want:
                    for r in load(p):
                        if r["run_id"] in want:
                            stored[r["run_id"]] = r
    replay = {r["replay"]["of_run_id"]: r for r in load(os.path.join(P.V31, "raw", "results", "heldexit31.jsonl"))}
    rows = []
    for n in new:
        o = old[(n["c11_group"], n["run_id"])]
        kind = C.kindof({"cell": n["cell"]})
        don, onlynew, onlyold = difffields(o, n, skip=("c11_group",))
        s = stored[n["run_id"]]
        exitcase = s.get("eci_status") in ("domain_exit", "terminated") and kind == "held"
        if exitcase and n["run_id"] in replay:
            ref, how = replay[n["run_id"]], "C01 replay record (physics_sha256)"
            dns, _, miss = difffields(n, ref, skip=("c11_group",))
            eq = SER.physics_sha256({k: v for k, v in n.items() if k != "c11_group"}) == SER.physics_sha256(ref)
        else:
            ref, how = s, "stored v3.0 record (legacy_comparable_sha256)"
            dns, _, miss = difffields(n, ref, skip=("c11_group",) + tuple(SER.V31ONLYFIELDS))
            eq = SER.legacycomp({k: v for k, v in n.items() if k != "c11_group"}) == SER.legacycomp(ref)
        flips = 0
        for key, x in sel.items():
            cell, meth, crit = key.split("|")
            if cell != n["cell"] or x.get("threshold") is None:
                continue
            va, sa, _ = C.featavail(o, meth, kind)
            vb, sb, _ = C.featavail(n, meth, kind)
            flips += (sa == "ok" and va <= x["threshold"]) != (sb == "ok" and vb <= x["threshold"])
        rows.append({"group": n["c11_group"], "cell": n["cell"], "split": n["split"], "index": n["index"], "case_id": n["case_id"],
                     "worker_status_old": o["worker_status"], "worker_status_new": n["worker_status"],
                     "old_new_physics_equal": SER.physics_sha256({k: v for k, v in o.items() if k != "c11_group"}) ==
                     SER.physics_sha256({k: v for k, v in n.items() if k != "c11_group"}),
                     "old_new_diff_fields": ";".join(don), "only_new_fields": ";".join(onlynew), "only_old_fields": ";".join(onlyold),
                     "new_vs": how, "new_vs_equal": eq, "new_vs_diff_fields": ";".join(dns), "new_vs_missing_fields": ";".join(miss),
                     "decision_flips_old_new": flips})
    with open(P.guard(os.path.join(P.V311RES, "normalcompare.csv")), "w", newline="", encoding="utf-8") as fh:
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
    with open(P.guard(os.path.join(P.V311RES, "normalcomparesummary.json")), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "runs"}, indent=1))
    for r in rows:
        if not (r["old_new_physics_equal"] and r["new_vs_equal"]) or r["only_new_fields"] or r["only_old_fields"]:
            print(r)


if __name__ == "__main__":
    main()
