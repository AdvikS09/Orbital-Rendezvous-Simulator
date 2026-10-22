import os

import numpy as np
import pandas as pd

import v31paths as P
from cwvalid3 import contracts as C
from cwvalid3 import ledger as L

METHODS = {"continuous": ("A", "B", "C", "sep", "sat"), "held": ("A_H", "B_H", "A_naive", "C_H", "sep", "sat")}


def paths(cell, split, root):
    man = os.path.join(root, "manifests", f"{cell}{split.replace('_', '')}.jsonl") if root else None
    if not (man and os.path.exists(man)):
        man = os.path.join(P.MAN, f"{cell}{split.replace('_', '')}.jsonl")
    res = os.path.join(root, "results", f"{cell}{split.replace('_', '')}.jsonl") if root else os.path.join(P.RES, f"{cell}{split.replace('_', '')}.jsonl")
    return man, res


def loadsplit(cell, split, root=None, reqfull=False):
    manp, resp = paths(cell, split, root)
    man = L.readjsonlstrict(manp, key="run_id")
    res = L.readjsonlstrict(resp, key="run_id", allow_missing=True)
    rc = L.reconcile([m["run_id"] for m in man], res, "run_id")
    if reqfull and (rc["n_missing"] or rc["n_extra"] or rc["n_duplicates"]):
        raise SystemExit(f"{cell}/{split}: results do not reconcile with the manifest: {rc}")
    by = {r["run_id"]: r for r in res}
    kind = C.kindof({"cell": cell})
    rows = []
    for m in man:
        r = by.get(m["run_id"], {"run_id": m["run_id"], "worker_status": "missing"})
        tr = C.truth_record(r, kind)
        d = {"run_id": m["run_id"], "case_id": m["case_id"], "physical_id": m["physical_id"], "cell": cell, "split": split,
             "index": m["index"], "worker_status": r.get("worker_status"), "failed_stage": r.get("failed_stage"),
             "truth_status": tr["truth_status"], "truth_reason": tr["reason"], "disc_m": tr["disc_m"], "u_num_m": tr["u_num_m"],
             "label_1m": C.critlabel(tr, 1.0), "label_0.1m": C.critlabel(tr, 0.1),
             **{f"arr_{k}": v for k, v in C.arrlabel(r, tr, kind).items()}, "contract": C.CONTRVER}
        for meth in METHODS[kind]:
            v, st, why = C.featavail(r, meth, kind)
            d[f"f_{meth}"], d[f"f_{meth}_status"], d[f"f_{meth}_reason"] = v, st, why
        rows.append(d)
    return pd.DataFrame(rows), rc


def adverse(labels):
    return np.array([C.isadverse(x) for x in labels], bool)


def evaluate(df, method, criterion, threshold):
    return C.evalrule(df[f"f_{method}"].to_numpy(float), list(df[f"label_{criterion}"]), threshold)
