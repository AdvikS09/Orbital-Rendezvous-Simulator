import collections
import csv
import json
import math
import os

import v31paths as P
from cwvalid3 import calibration as CAL
from cwvalid3 import contracts as C

METHODS = {"continuous": ("A", "B", "C", "sep", "sat"), "held": ("A_H", "B_H", "A_naive", "C_H", "sep", "sat")}


def same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def main():
    rows, changes = [], []
    patterns = collections.Counter()
    for cell, split, path in P.resfiles():
        kind = C.kindof({"cell": cell})
        n = 0
        cnt = collections.Counter()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                n += 1
                t0, t1 = CAL.truth_record(r), C.truth_record(r, kind)
                items = [("truth_status", t0["truth_status"], t1["truth_status"])]
                for tol, nm in ((1.0, "label_1m"), (0.1, "label_0.1m")):
                    items.append((nm, CAL.critlabel(t0, tol), C.critlabel(t1, tol)))
                items.append(("arrival_status", CAL.arrlabel(r, t0)["arrival_status"],
                              C.arrlabel(r, t1, kind)["arrival_status"]))
                if t0["truth_status"] == "completed_valid" and t1["truth_status"] == "completed_valid":
                    items.append(("disc_m", t0["disc_m"], t1["disc_m"]))
                    items.append(("u_num_m", t0["u_num_m"], t1["u_num_m"]))
                for m in METHODS[kind]:
                    items.append((f"f_{m}", CAL.feature_value(r, m, kind), C.feature_value(r, m, kind)))
                for item, a, b in items:
                    if not same(a, b):
                        cnt[item] += 1
                        patterns[(cell, split, item, str(a), str(b), r.get("worker_status"))] += 1
                        changes.append({"cell": cell, "split": split, "index": r.get("index"), "case_id": r.get("case_id"),
                                        "run_id": r.get("run_id"), "worker_status": r.get("worker_status"), "item": item,
                                        "v30": a, "v31": b, "v31_reason": t1.get("reason", "")})
        rows.append({"cell": cell, "split": split, "kind": kind, "n_records": n, "n_records_changed":
                     len({(c["run_id"]) for c in changes if c["cell"] == cell and c["split"] == split}),
                     **{f"changed_{k}": v for k, v in sorted(cnt.items())}})
    os.makedirs(P.V31RES, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (not k in ("cell", "split", "kind", "n_records", "n_records_changed"), k))
    base = ["cell", "split", "kind", "n_records", "n_records_changed"]
    keys = base + sorted(k for k in keys if k not in base)
    with open(P.guardv30(os.path.join(P.V31RES, "contractimpact.csv")), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval=0)
        w.writeheader()
        w.writerows(rows)
    with open(P.guardv30(os.path.join(P.V31RES, "contractchanges.csv")), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["cell", "split", "index", "case_id", "run_id", "worker_status", "item", "v30", "v31", "v31_reason"])
        w.writeheader()
        w.writerows(changes)
    summary = {"contract": C.CONTRVER, "n_records": sum(r["n_records"] for r in rows),
               "n_records_changed": sum(r["n_records_changed"] for r in rows), "n_item_changes": len(changes),
               "patterns": [{"cell": k[0], "split": k[1], "item": k[2], "v30": k[3], "v31": k[4], "worker_status": k[5], "n": v}
                            for k, v in sorted(patterns.items())]}
    with open(P.guardv30(os.path.join(P.V31RES, "contractimpact.json")), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "patterns"}))
    for p in summary["patterns"]:
        print(p)


if __name__ == "__main__":
    main()
