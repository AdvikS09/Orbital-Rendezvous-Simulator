"""C02/C03: the staged v3.1 workers must reproduce stored v3.0 records exactly on the 'ok' path.

Re-executes a fixed sample of planned cases with the v3.1 campaign workers and compares every stored
field (after the canonical normalisation, NaN == null) with the v3.0 record. Volatile fields (timing,
software, completed_at, batch_size) and fields new in v3.1 are excluded from the comparison and listed.
Held cases are taken from records without a physical-domain exit (exits change by design, C01).
The sample is fixed in advance: the first N_CONT planned indices of every continuous test split and
the first N_HELD non-exit planned indices of every held test split (no outcome-based choice).

Writes v3_1_correction/results/worker_equivalence_check.csv and .json.
Usage: python code/experiments/v3_1/c02_worker_equivalence.py [--n-cont 6] [--n-held 8]
"""

import argparse
import csv
import json
import os
import sys
import time

import v31paths as P

sys.path.insert(0, os.path.join(P.REPO, "experiments", "v3"))
import campaign as CP  # noqa: E402
from cwvalid3 import serialization as SER  # noqa: E402

NEW_V31 = set(SER.V31_ONLY_FIELDS)


def load(cell, split):
    man = [json.loads(x) for x in open(os.path.join(P.MAN, f"{cell}_{split}.jsonl"), encoding="utf-8") if x.strip()]
    res = {r["run_id"]: r for r in (json.loads(x) for x in open(os.path.join(P.RES, f"{cell}_{split}.jsonl"), encoding="utf-8") if x.strip())}
    return man, res


def compare(new, old):
    a, b = SER.normalize(new), SER.normalize(old)
    vol = SER.is_volatile
    ka = {k for k in a if not vol(k) and k not in NEW_V31}
    kb = {k for k in b if not vol(k)}
    diff = sorted(k for k in ka & kb if a[k] != b[k])
    return diff, sorted(ka - kb), sorted(kb - ka)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cont", type=int, default=6)
    ap.add_argument("--n-held", type=int, default=8)
    ap.add_argument("--replay-chunk", type=int, default=3, help="C01 replay chunk to re-execute (3 = the 31-case D3 chunk)")
    a = ap.parse_args()
    rows = []
    t0 = time.perf_counter()
    for cell in ("D0", "D1", "D2", "D5"):
        man, res = load(cell, "test")
        for m in man[:a.n_cont]:
            o = CP.work_continuous(m)[0]
            d, extra, missing = compare(o, res[m["run_id"]])
            rows.append({"cell": cell, "split": "test", "index": m["index"], "case_id": m["case_id"], "worker_status": o["worker_status"],
                         "n_fields_compared": len(set(SER.normalize(o)) & set(res[m["run_id"]])), "n_diff": len(d),
                         "diff_fields": ";".join(d), "only_new": ";".join(extra), "only_old": ";".join(missing),
                         "legacy_comparable_sha256_new": SER.legacy_comparable_sha256(o),
                         "legacy_comparable_sha256_stored": SER.legacy_comparable_sha256(res[m["run_id"]]),
                         "hash_equal": SER.legacy_comparable_sha256(o) == SER.legacy_comparable_sha256(res[m["run_id"]])})
    for cell in ("D3", "D4"):
        man, res = load(cell, "test")
        pick = [m for m in man if res[m["run_id"]].get("eci_status") == "completed" and res[m["run_id"]].get("nerm_status") == "completed"][:a.n_held]
        outs = CP.work_held(pick)
        for m, o in zip(pick, outs):
            d, extra, missing = compare(o, res[m["run_id"]])
            rows.append({"cell": cell, "split": "test", "index": m["index"], "case_id": m["case_id"], "worker_status": o["worker_status"],
                         "n_fields_compared": len(set(SER.normalize(o)) & set(res[m["run_id"]])), "n_diff": len(d),
                         "diff_fields": ";".join(d), "only_new": ";".join(extra), "only_old": ";".join(missing),
                         "legacy_comparable_sha256_new": SER.legacy_comparable_sha256(o),
                         "legacy_comparable_sha256_stored": SER.legacy_comparable_sha256(res[m["run_id"]]),
                         "hash_equal": SER.legacy_comparable_sha256(o) == SER.legacy_comparable_sha256(res[m["run_id"]])})
    # 3. the final engine reproduces a C01 replay chunk exactly (same chunk membership and order)
    rep = [json.loads(x) for x in open(os.path.join(P.V31_RAW, "results", "held_exit_replay_v31.jsonl"), encoding="utf-8")]
    chunk = [r for r in rep if r["replay"]["chunk"] == a.replay_chunk]
    want = [r["replay"]["of_run_id"] for r in chunk]
    cell = chunk[0]["cell"]
    mans = {}
    for sp in ("pilot", "dev", "cal", "test", "transfer_cohort"):
        mp = os.path.join(P.MAN, f"{cell}_{sp}.jsonl")
        for x in open(mp, encoding="utf-8"):
            m = json.loads(x)
            if m["run_id"] in want:
                mans[m["run_id"]] = m
    outs = CP.work_held([mans[rid] for rid in want])
    for r0, o in zip(chunk, outs):
        d, extra, missing = compare(o, r0)
        rows.append({"cell": cell, "split": r0["split"], "index": r0["index"], "case_id": r0["case_id"],
                     "worker_status": o["worker_status"], "n_fields_compared": len(set(SER.normalize(o)) & set(r0)),
                     "n_diff": len(d), "diff_fields": ";".join(d), "only_new": ";".join(extra), "only_old": ";".join(missing),
                     "legacy_comparable_sha256_new": SER.physics_sha256(o), "legacy_comparable_sha256_stored": SER.physics_sha256(r0),
                     "hash_equal": SER.physics_sha256(o) == SER.physics_sha256(r0), "check": f"replay chunk {a.replay_chunk} (physics_sha256)"})
    os.makedirs(P.V31_RES, exist_ok=True)
    with open(P.guard_v30(os.path.join(P.V31_RES, "worker_equivalence_check.csv")), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[-1]), restval="sample vs stored v3.0 (legacy_comparable_sha256)")
        w.writeheader()
        w.writerows(rows)
    summ = {"n_cases": len(rows), "n_with_diff": sum(r["n_diff"] > 0 for r in rows),
            "all_ok": all(r["worker_status"] == "ok" for r in rows), "engine_sha_v31": CP.engine_version(),
            "excluded_volatile": sorted(SER.VOLATILE_KEYS) + [p + "*" for p in SER.VOLATILE_PREFIXES] +
                                 ["*" + x for x in SER.VOLATILE_SUFFIXES],
            "excluded_new_v31_fields": sorted(NEW_V31), "wall_s": time.perf_counter() - t0,
            "physics_hash_version": SER.PHYSICS_HASH_VERSION, "git_head": os.popen(f'git -C "{P.REPO}" rev-parse HEAD').read().strip(),
            "n_hash_equal": sum(bool(r["hash_equal"]) for r in rows),
            "n_sample_vs_v30": sum(1 for r in rows if "check" not in r), "n_replay_chunk_rows": sum(1 for r in rows if "check" in r),
            "sample_rule": "first n planned indices per continuous test split; first n non-exit planned indices per held test split; "
                           "plus every case of one C01 replay chunk, re-executed in the same membership and order"}
    with open(P.guard_v30(os.path.join(P.V31_RES, "worker_equivalence_check.json")), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ, indent=1))
    for r in rows:
        if r["n_diff"] or r["only_old"]:
            print(r)


if __name__ == "__main__":
    main()
