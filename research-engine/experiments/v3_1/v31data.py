"""v3.1 case loader: experiments/v3/v3data.load_split with the v3.1 record contract (cwvalid3.contracts) instead of
the frozen v3.0 labelling, and with an optional v3.1 output root (results written by `campaign.py run --out`).

Per planned case: truth status, labels (1 m, 0.1 m; LABELS incl. 'interrupted'), arrival status, and for every method
the feature value with its availability status and reason. A 'reference_failed' record keeps its feature (accepted
if below threshold) with 'unresolved' truth, as PROSPECTIVE_PLAN.json specifies; a planned case without a record is
'missing' (operational, 'interrupted'). evaluate() applies contracts.evaluate_rule, which enforces the label-partition
identity. The frozen v3.0 analyses keep using v3data/calibration; this loader is for v3.1 analyses and new runs.
Usage (library): from v31data import load_split, evaluate
"""

import os

import numpy as np
import pandas as pd

import v31paths as P
from cwvalid3 import contracts as C
from cwvalid3 import ledger as L

METHODS = {"continuous": ("A", "B", "C", "sep", "sat"), "held": ("A_H", "B_H", "A_naive", "C_H", "sep", "sat")}


def _paths(cell, split, root):
    man = os.path.join(root, "manifests", f"{cell}_{split}.jsonl") if root else None
    if not (man and os.path.exists(man)):
        man = os.path.join(P.MAN, f"{cell}_{split}.jsonl")
    res = os.path.join(root, "results", f"{cell}_{split}.jsonl") if root else os.path.join(P.RES, f"{cell}_{split}.jsonl")
    return man, res


def load_split(cell, split, root=None, require_complete=False):
    """DataFrame of planned cases with v3.1 labels and features (root: a v3.1 output root, else the v3.0 tree)."""
    man_p, res_p = _paths(cell, split, root)
    man = L.read_jsonl_strict(man_p, key="run_id")
    res = L.read_jsonl_strict(res_p, key="run_id", allow_missing=True)
    rc = L.reconcile([m["run_id"] for m in man], res, "run_id")
    if require_complete and (rc["n_missing"] or rc["n_extra"] or rc["n_duplicates"]):
        raise SystemExit(f"{cell}/{split}: results do not reconcile with the manifest: {rc}")
    by = {r["run_id"]: r for r in res}
    kind = C.kind_of({"cell": cell})
    rows = []
    for m in man:
        r = by.get(m["run_id"], {"run_id": m["run_id"], "worker_status": "missing"})
        tr = C.truth_record(r, kind)
        d = {"run_id": m["run_id"], "case_id": m["case_id"], "physical_id": m["physical_id"], "cell": cell, "split": split,
             "index": m["index"], "worker_status": r.get("worker_status"), "failed_stage": r.get("failed_stage"),
             "truth_status": tr["truth_status"], "truth_reason": tr["reason"], "disc_m": tr["disc_m"], "u_num_m": tr["u_num_m"],
             "label_1m": C.criterion_label(tr, 1.0), "label_0.1m": C.criterion_label(tr, 0.1),
             **{f"arr_{k}": v for k, v in C.arrival_label(r, tr, kind).items()}, "contract": C.CONTRACT_VERSION}
        for meth in METHODS[kind]:
            v, st, why = C.feature_availability(r, meth, kind)
            d[f"f_{meth}"], d[f"f_{meth}_status"], d[f"f_{meth}_reason"] = v, st, why
        rows.append(d)
    return pd.DataFrame(rows), rc


def adverse(labels):
    return np.array([C.is_adverse(x) for x in labels], bool)


def evaluate(df, method, criterion, threshold):
    """contracts.evaluate_rule on one family of a loaded split."""
    return C.evaluate_rule(df[f"f_{method}"].to_numpy(float), list(df[f"label_{criterion}"]), threshold)


__all__ = ["load_split", "adverse", "evaluate", "METHODS"]
