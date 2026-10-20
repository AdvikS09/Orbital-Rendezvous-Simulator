import collections
import csv
import hashlib
import json
import os

import v31paths as P
from cwvalid3 import ledger as L
from cwvalid3 import serialization as SER

RULE = ("nan-restore/1: a JSON null in a stored v3.0 result record was float NaN in memory, except "
        "{cw,eci,nerm}_sat_events_disabled_at_s (continuous cells), d1_failure_reason (D0-D2) and cpu_s/wall_s "
        "of worker_crash records, which were Python None")
CONT, PD = ("D0", "D1", "D2", "D5"), ("D0", "D1", "D2")
KNOWNENG = {"ae0c780aaa672ad4": "commits 1a16fb1 / 2aa8c44 (frozen implementation)",
            "ad3ad5cb611c3786": "commit e376cec (pre-freeze); differs from ae0c780 only by the numerically inert ledger.text_sha256 helper",
            "25643c68cae0ff3b": "matches no committed snapshot (working-tree state before e376cec; source not archived)"}


def nonefields(r):
    s = set()
    if r.get("cell") in CONT:
        s |= {f"{b}_sat_events_disabled_at_s" for b in ("cw", "eci", "nerm")}
    if r.get("cell") in PD:
        s.add("d1_failure_reason")
    if r.get("worker_status") == "worker_crash":
        s |= {"cpu_s", "wall_s"}
    return s


def main():
    led = {}
    nohash = []
    evcount = collections.Counter()
    with open(os.path.join(P.V3, "ledger.jsonl"), encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            e = json.loads(line)
            evcount[e["event"]] += 1
            if e["event"] in ("completed", "completed_reconciled", "rerun"):
                led.setdefault(e["run_id"], []).append((e["event"], e.get("result_sha256_16"), n, e.get("engine_sha")))
                if not e.get("result_sha256_16"):
                    nohash.append({"line": n, "event": e["event"], "run_id": e["run_id"]})
    bound = {}
    for f in os.listdir(P.MAN):
        if f.endswith("summary.json"):
            s = json.load(open(os.path.join(P.MAN, f), encoding="utf-8"))
            bound[f"{s['cell']}_{s['split']}"] = s["run_config"].get("engine_sha")
    rows = []
    restored = collections.Counter()
    kept = collections.Counter()
    engines = collections.Counter()
    for cell, split, path in P.resfiles():
        fname = os.path.basename(path)
        key = f"{cell}_{'cal' if split == 'cal_reruns' else split}"
        with open(path, "rb") as fh:
            for ln, raw in enumerate(fh, 1):
                line = raw.rstrip(b"\r\n")
                r = json.loads(line)
                pay = {k: v for k, v in r.items() if k not in ("software", "completed_at")}
                evs = led.get(r["run_id"], [])
                if split == "cal_reruns":
                    evs = [e for e in evs if e[0] == "rerun"]
                else:
                    evs = [e for e in evs if e[0] in ("completed", "completed_reconciled")]
                ev = evs[0] if len(evs) == 1 else (None, None, None, None)
                legacy = ev[1]
                direct = L.stablehash(pay, 32)
                nf = nonefields(r)
                rest = sorted(k for k, v in pay.items() if v is None and k not in nf)
                q = dict(pay)
                for k in rest:
                    q[k] = float("nan")
                ruled = L.stablehash(q, 32)
                for k, v in pay.items():
                    if v is None:
                        (kept if k in nf else restored)[k] += 1
                if legacy is None:
                    status = "no_legacy_hash" if evs else "no_ledger_event"
                elif direct == legacy and ev[0] == "completed_reconciled":
                    status = "reconstructed_ledger_hash"
                elif direct == legacy:
                    status = "direct_match"
                elif ruled == legacy:
                    status = "reconciled_by_rule"
                else:
                    status = "unreconciled"
                eng = (r.get("software") or {}).get("engine_sha")
                engines[(fname, eng, bound.get(key))] += 1
                b, h = SER.canline(r, None)
                rows.append({"results_file": fname, "line": ln, "run_id": r["run_id"], "case_id": r.get("case_id"),
                             "cell": cell, "split": split, "worker_status": r.get("worker_status"),
                             "ledger_event": ev[0], "ledger_line": ev[2], "legacy_result_sha256_32hex": legacy,
                             "legacy_recomputed_direct": direct, "legacy_recomputed_after_rule": ruled,
                             "reconciliation": status, "n_nulls_restored_to_nan": len(rest),
                             "restored_fields": ";".join(rest), "stored_line_sha256": hashlib.sha256(line).hexdigest(),
                             "v31_canonical_sha256_of_stored_content": h, "v31_physics_sha256": SER.physics_sha256(r),
                             "stored_line_is_v31_canonical": b == line, "engine_sha_record": eng,
                             "engine_sha_bound_in_run_id": bound.get(key)})
    out = os.path.join(P.V31RAW, "legacyhashes.csv")
    with open(P.guardv30(out), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    st = collections.Counter(r["reconciliation"] for r in rows)
    summ = {"rule": RULE, "n_records": len(rows), "reconciliation": dict(st),
            "by_file": {f: dict(collections.Counter(r["reconciliation"] for r in rows if r["results_file"] == f))
                        for f in sorted({r["results_file"] for r in rows})},
            "ledger_event_counts": dict(evcount), "ledger_events_without_result_hash": len(nohash),
            "ledger_events_without_result_hash_detail": nohash,
            "nulls_restored_to_nan_by_field": dict(restored.most_common()), "nulls_kept_none_by_field": dict(kept.most_common()),
            "engine_census": [{"results_file": f, "engine_sha_record": e, "engine_sha_bound_in_run_id": b, "n": n,
                               "producing_engine_note": KNOWNENG.get(e, "unknown"),
                               "run_id_binds_producing_engine": e == b}
                              for (f, e, b), n in sorted(engines.items(), key=lambda x: (x[0][0], str(x[0][1])))],
            "n_records_whose_run_id_binds_another_engine": sum(n for (f, e, b), n in engines.items() if e != b),
            "any_stored_line_v31_canonical": any(r["stored_line_is_v31_canonical"] for r in rows),
            "policy": "ledger.jsonl and results/*.jsonl are not modified; this sidecar maps each stored record to "
                      "its legacy hash, the reconciliation rule and the v3.1 hashes of the stored content",
            "output_sha256": L.filesha256(out)}
    with open(P.guardv30(os.path.join(P.V31RAW, "legacyhashes.json")), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps({k: summ[k] for k in ("n_records", "reconciliation", "ledger_events_without_result_hash",
                                           "n_records_whose_run_id_binds_another_engine", "any_stored_line_v31_canonical")}, indent=1))


if __name__ == "__main__":
    main()
