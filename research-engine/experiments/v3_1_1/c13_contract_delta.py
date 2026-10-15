"""v3.1.1: contract v3.1-contract-3 vs v3.1-contract-2 on every stored record.

contract-3 only adds the driver-level nominal-only recovery (feature_recovery 'driver_nominal_recovery') to the
recovered-feature condition of feature_availability. This scan applies both versions to every stored v3.0 record
(all result files, including the 43 D1 calibration reruns) and every stored v3.1 record under
v3_1_correction/raw/results, and lists every difference in truth status and reason, disc/u_num, criterion labels
(1 m, 0.1 m), arrival status and feature value/availability/reason for every method. contract-2 is loaded from the
v3.1 release commit (git show v3.1:cwvalid3/contracts.py), never from the working tree.

Writes v3_1_1_followup/results/contract3_vs_contract2_scan.json (and _changes.csv if any change).
Usage: python c13_contract_delta.py
"""

import csv
import glob
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import tempfile
import time

import v311paths as P
from cwvalid3 import contracts as C3

METHODS = {"continuous": ("A", "B", "C", "sep", "sat"), "held": ("A_H", "B_H", "A_naive", "C_H", "sep", "sat")}


def load_contract2():
    src = subprocess.run(["git", "-C", P.REPO, "show", "v3.1:cwvalid3/contracts.py"], capture_output=True, check=True).stdout
    d = tempfile.mkdtemp(prefix="c13_")
    p = os.path.join(d, "contracts_v31.py")
    open(p, "wb").write(src)
    spec = importlib.util.spec_from_file_location("contracts_v31", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, hashlib.sha256(src.replace(b"\r\n", b"\n")).hexdigest()


def same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def items(C, r, kind):
    tr = C.truth_record(r, kind)
    out = [("truth_status", tr["truth_status"]), ("truth_reason", tr["reason"]), ("disc_m", tr["disc_m"]),
           ("u_num_m", tr["u_num_m"]), ("label_1m", C.criterion_label(tr, 1.0)), ("label_0.1m", C.criterion_label(tr, 0.1)),
           ("arrival_status", C.arrival_label(r, tr, kind)["arrival_status"])]
    for m in METHODS[kind]:
        v, st, why = C.feature_availability(r, m, kind)
        out += [(f"f_{m}", v), (f"f_{m}_status", st), (f"f_{m}_reason", why)]
    return out


def main():
    C2, c2sha = load_contract2()
    assert C2.CONTRACT_VERSION == "v3.1-contract-2" and C3.CONTRACT_VERSION == "v3.1-contract-3"
    files = [(c, s, p, "v3.0") for c, s, p in P.P31.result_files()]
    files += [(None, os.path.basename(p), p, "v3.1") for p in sorted(glob.glob(os.path.join(P.V31, "raw", "results", "*.jsonl")))]
    t0 = time.perf_counter()
    per, changes = [], []
    n_total = 0
    for cell, split, path, tree in files:
        n = nch = n_rec = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                kind = C3.kind_of({"cell": r.get("cell", cell)})
                n += 1
                n_rec += "feature_recovery" in r
                a, b = items(C2, r, kind), items(C3, r, kind)
                diff = [(k, x, y) for (k, x), (_, y) in zip(a, b) if not same(x, y)]
                if diff:
                    nch += 1
                    for k, x, y in diff:
                        changes.append({"file": os.path.relpath(path, P.ARCH).replace("\\", "/"), "run_id": r.get("run_id"),
                                        "item": k, "contract2": x, "contract3": y})
        n_total += n
        per.append({"tree": tree, "file": os.path.relpath(path, P.ARCH).replace("\\", "/"), "n_records": n,
                    "n_records_changed": nch, "n_with_feature_recovery": n_rec})
    summ = {"contract_old": C2.CONTRACT_VERSION, "contract_old_source": "git show v3.1:cwvalid3/contracts.py",
            "contract_old_sha256_lf": c2sha, "contract_new": C3.CONTRACT_VERSION, "n_files": len(per), "n_records": n_total,
            "n_records_changed": sum(x["n_records_changed"] for x in per), "n_item_changes": len(changes),
            "n_records_with_feature_recovery": sum(x["n_with_feature_recovery"] for x in per),
            "items_compared": "truth status and reason, disc_m, u_num_m, labels 1 m / 0.1 m, arrival status, and per method "
                              "the feature value, availability status and reason", "files": per,
            "wall_s": time.perf_counter() - t0}
    os.makedirs(P.V311_RES, exist_ok=True)
    with open(P.guard(os.path.join(P.V311_RES, "contract3_vs_contract2_scan.json")), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    if changes:
        with open(P.guard(os.path.join(P.V311_RES, "contract3_vs_contract2_changes.csv")), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(changes[0]))
            w.writeheader()
            w.writerows(changes)
    print(json.dumps({k: v for k, v in summ.items() if k != "files"}, indent=1))


if __name__ == "__main__":
    main()
