import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
ARCH = os.path.dirname(REPO)
V3 = os.path.join(ARCH, "v3")
RES = os.path.join(V3, "raw", "results")
OUT = os.path.join(V3, "results", "dev")

from cwvalid3 import calibration as CAL
from cwvalid3 import ledger as L
from cwvalid3.domains import CELLS

DEVSPLITS = ("pilot", "dev")
HIST = json.load(open(os.path.join(ARCH, "results/v2/raw/frozenv2/estimatorv2.json")))["rules"]


def meth2(cell):
    return ("A", "B", "C", "sep", "sat") if CELLS[cell]["cmd_mode"] == "continuous" else ("A_H", "B_H", "A_naive", "C_H", "sep", "sat")


def histcut(method, tol):
    pref = "traj1" if tol == 1.0 else "traj01"
    feat = {"A": "d1", "A_H": "d1", "A_naive": "d1", "B": "d1u", "B_H": "d1u", "C": "cres", "C_H": "cres",
            "sep": "sep", "sat": "sat"}[method]
    r = HIST.get(f"{pref}_{feat}")
    return r["threshold"] if r and r.get("status") == "ok" else None


def casetable(cell, split):
    if split not in DEVSPLITS:
        raise SystemExit("development analysis only")
    rows = L.readjsonlstrict(os.path.join(RES, f"{cell}{split.replace('_', '')}.jsonl"), key="run_id")
    kind = "continuous" if CELLS[cell]["cmd_mode"] == "continuous" else "held"
    recs = []
    for r in rows:
        tr = CAL.truth_record(r)
        d = {"run_id": r["run_id"], "case_id": r["case_id"], "cell": cell, "split": split, "index": r["index"],
             "worker_status": r.get("worker_status"), "sep_m": r.get("sep_m"), "nu": r.get("nu"),
             "wn_ratio": r.get("wn_ratio"), "a_max": r.get("a_max"), "cpu_s": r.get("cpu_s"), **tr,
             "label_1m": CAL.critlabel(tr, 1.0), "label_01m": CAL.critlabel(tr, 0.1),
             **{f"arr_{k}": v for k, v in CAL.arrlabel(r, tr).items()}}
        for m in meth2(cell):
            d[f"f_{m}"] = CAL.feature_value(r, m, kind)
        if kind == "held":
            d["n_near_switch_updates"] = r.get("cw_n_near_switch_updates")
            d["sat_fraction"] = r.get("cw_sat_fraction")
            d["cw_naive_pass_max_state_diff_m"] = r.get("cw_naive_pass_max_state_diff_m")
        else:
            d["sat_fraction"] = r.get("feature_sat_fraction")
            d["n_sat_switch"] = r.get("cw_n_sat_switch_events")
        recs.append(d)
    return pd.DataFrame(recs)


def summar(df, cell):
    s = {"cell": cell, "n": int(len(df)), "worker_status": df.worker_status.value_counts().to_dict(),
         "truth_status": df.truth_status.value_counts().to_dict(),
         "label_1m": df.label_1m.value_counts().to_dict(), "label_01m": df.label_01m.value_counts().to_dict(),
         "arrival_status": df.arr_arrival_status.value_counts().to_dict(),
         "cpu_s_per_case": {"mean": float(df.cpu_s.mean()), "median": float(df.cpu_s.median()),
                            "p95": float(df.cpu_s.quantile(0.95)), "max": float(df.cpu_s.max())},
         "u_num_m": {"median": float(df.u_num_m.median()), "p95": float(df.u_num_m.quantile(0.95)),
                     "max": float(df.u_num_m.max())}}
    ok = df.truth_status == "completed_valid"
    feats = {}
    for m in meth2(cell):
        f = df[f"f_{m}"].to_numpy(float)
        ent = {"n_unavailable": int((~np.isfinite(f)).sum())}
        for tol, lab in ((1.0, "label_1m"), (0.1, "label_01m")):
            hc = histcut(m, tol)
            if hc is not None:
                acc = CAL.acceptmask(f, hc)
                bad = df[lab].isin(["violation", "exit", "failure", "unresolved"]).to_numpy()
                ent[f"hist_cutoff_{tol:g}m"] = {"threshold": hc, "accepted": int(acc.sum()),
                                                 "accepted_adverse": int((acc & bad).sum())}
        if m not in ("sep", "sat"):
            sel = ok.to_numpy() & np.isfinite(f) & (df.disc_m.to_numpy(float) > 1e-6)
            if sel.any():
                ratio = f[sel] / df.disc_m.to_numpy(float)[sel]
                near = sel & (df.disc_m.to_numpy(float) > 0.01) & (df.disc_m.to_numpy(float) < 100)
                ent["ratio_feature_over_truth"] = {"n": int(sel.sum()), "median": float(np.median(ratio)),
                                                   "min": float(ratio.min()), "max": float(ratio.max()),
                                                   "n_0.01_100m": int(near.sum()),
                                                   "min_0.01_100m": float((f[near] / df.disc_m.to_numpy(float)[near]).min()) if near.any() else None}
        feats[m] = ent
    s["features"] = feats
    return s


def main():
    import v30closed
    v30closed.refuseoutsharness(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["pilot"])
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or "".join(a.splits)
    os.makedirs(OUT, exist_ok=True)
    fram, summ = [], []
    for cell in CELLS:
        parts = [casetable(cell, sp) for sp in a.splits if os.path.exists(os.path.join(RES, f"{cell}{sp.replace('_', '')}.jsonl"))]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        fram.append(df)
        summ.append(summar(df, cell))
    allc = pd.concat(fram, ignore_index=True)
    allc.to_csv(os.path.join(OUT, f"{tag}cases.csv"), index=False)
    json.dump(L.jsonable(summ), open(os.path.join(OUT, f"{tag}summary.json"), "w"), indent=1)
    for s in summ:
        print(json.dumps(L.jsonable(s)))


if __name__ == "__main__":
    main()
