"""P04/P05/P06/P07 single held-out evaluation of the frozen rules.

Requires PROSPECTIVE_PLAN.json, raw/calibration_freeze.json and complete test splits. Rules are read
from the freeze file only; nothing here selects or changes a threshold. Writes
  results/risk_coverage.csv          selected rules, legacy rules, matched-workload rules; descriptive test curves
  results/paired_comparisons.csv     paired method contrasts on identical test cases
  results/transfer_matrix.csv        T1 / T2 / T3 (and T1-heuristic) per target cell
  results/hybrid_screening.csv       held-command cells: rules and prediction diagnostics
  results/controller_transfer.csv    D5 (matrix LQR) slices
  raw/test_decisions.csv             per-case accept/refer decisions for every frozen rule
  raw/test_evaluation_record.json    first-evaluation time, input hashes, output hashes
Usage: python code/experiments/v3/evaluate_test.py
"""

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v3data as VD  # noqa: E402
from cwvalid3 import calibration as CAL  # noqa: E402
from cwvalid3 import ledger as L  # noqa: E402
from cwvalid3.domains import CELLS  # noqa: E402

PLAN = os.path.join(VD.V3, "PROSPECTIVE_PLAN.json")
FREEZE = os.path.join(VD.V3, "raw", "calibration_freeze.json")
RECORD = os.path.join(VD.V3, "raw", "test_evaluation_record.json")
OUTR = os.path.join(VD.V3, "results")
R_E = 6378137.0
GOOD = "ok"
PAIRS = {"continuous": [("A", "B"), ("A", "C"), ("A", "sep"), ("A", "sat"), ("B", "C")],
         "held": [("A_H", "B_H"), ("A_H", "A_naive"), ("A_H", "C_H"), ("A_H", "sep"), ("A_H", "sat"), ("A_naive", "B_H")]}


def rule_eval(df, method, crit, thr):
    v = df[f"f_{method}"].to_numpy(float)
    lab = df[f"label_{crit}"].to_numpy(object)
    return CAL.evaluate_rule(v, lab, thr), CAL.accept_mask(v, thr) if thr is not None else np.zeros(v.size, bool)


def arrival_among(df, acc):
    sub = df[acc]
    st = sub.arr_arrival_status
    res = sub[st.isin(["arrived", "completed_non_arrival"])]
    ff = int(((res.arr_cw_arrived == True) & (res.arr_ref_arrived == False)).sum())  # noqa: E712
    fu = int(((res.arr_cw_arrived == False) & (res.arr_ref_arrived == True)).sum())  # noqa: E712
    return {"arr_resolved": int(len(res)), "arr_false_favourable": ff, "arr_false_unfavourable": fu,
            "arr_unresolved": int((st == "unresolved").sum()), "arr_not_applicable": int((st == "not_applicable").sum())}


def prediction_stats(df, method):
    f = df[f"f_{method}"].to_numpy(float)
    d = df.disc_m.to_numpy(float)
    ok = (df.truth_status == "completed_valid").to_numpy() & np.isfinite(f) & np.isfinite(d)
    sel = ok & (d > 1e-3)
    out = {"n_valid": int(ok.sum()), "n_ratio": int(sel.sum())}
    if sel.sum() > 2:
        r = f[sel] / d[sel]
        out.update({"ratio_q05": float(np.quantile(r, 0.05)), "ratio_q50": float(np.quantile(r, 0.5)),
                    "ratio_q95": float(np.quantile(r, 0.95)), "frac_under_10pct": float(np.mean(r < 0.9)),
                    "frac_over_10pct": float(np.mean(r > 1.1)),
                    "spearman_feature_truth": float(spearmanr(f[ok], d[ok]).correlation)})
        near = sel & (d > 0.01) & (d < 100)
        if near.any():
            rn = f[near] / d[near]
            out.update({"n_0.01_100m": int(near.sum()), "ratio_min_0.01_100m": float(rn.min()),
                        "ratio_max_0.01_100m": float(rn.max()), "frac_under_1pct_0.01_100m": float(np.mean(rn < 0.99))})
    small = ok & (d <= 1e-3)
    if small.any():
        out["abs_err_max_truth_below_1mm_m"] = float(np.max(np.abs(f[small] - d[small])))
    viol = df["label_1m"].isin(["violation", "exit"]).to_numpy()
    good = (df["label_1m"] == "ok").to_numpy()
    m = np.isfinite(f) & (viol | good)
    if m.sum() and viol[m].any() and good[m].any():
        from scipy.stats import mannwhitneyu
        u = mannwhitneyu(f[m & viol], f[m & good], alternative="two-sided").statistic
        out["auc_violation_1m"] = float(u / (viol[m].sum() * good[m].sum()))
    return out


def main():
    import v30_closed  # v3.1: preserved v3.0 outputs; runs only inside the rebuild harness
    v30_closed.refuse_outside_harness(__file__)
    for p in (PLAN, FREEZE):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    plan, frz = json.load(open(PLAN)), json.load(open(FREEZE))
    if frz["plan_sha256"] != L.file_sha256(PLAN):
        sys.exit("plan changed after the calibration freeze")
    sel = frz["selection"]
    fams = {f["family_id"]: f for f in plan["families"]}
    data = {c: VD.load_split(c, "test")[0] for c in plan["domains"]}
    R0 = R_E + CELLS["D0"]["altitude"]
    rc, pc, tm, hs, ct, dec = [], [], [], [], [], {}
    for c, df in data.items():
        dec[c] = df[["case_id", "physical_id", "cell", "index", "truth_status", "disc_m", "u_num_m", "label_1m", "label_0.1m",
                     "arr_arrival_status"] + [f"f_{m}" for m in VD.methods_for(c)]].copy()
    # ---- selected, legacy and matched-workload rules -------------------------------------------
    for fid, s in sel.items():
        c, m, crit = s["cell"], s["method"], s["criterion"]
        df = data[c]
        ev, acc = rule_eval(df, m, crit, s["threshold"])
        row = {"kind": "selected_calibrated", "family_id": fid, "cell": c, "method": m, "criterion": crit,
               "threshold": s["threshold"], "rule_source": s["source"], "selection_status": s["status"],
               "certificate": f"U<={plan['multiplicity']['alpha']} at level delta/M on calibration" if s["threshold"] is not None else "abstain",
               "cal_accepted": s["cal_accepted"], "cal_adverse": s["cal_adverse"], "cal_ucb": s["cal_ucb"], **ev}
        if crit == "1m":
            row.update(arrival_among(df, acc))
        rc.append(row)
        dec[c][f"acc_sel_{m}_{crit}"] = acc
        f = fams[fid]
        hist = f.get("historical_cutoff")
        if hist is not None:
            ev2, acc2 = rule_eval(df, m, crit, hist)
            rc.append({"kind": "legacy_padded_rule", "family_id": fid, "cell": c, "method": m, "criterion": crit,
                       "threshold": hist, "rule_source": "historical v2 frozen (K=0.8 x smallest violating dev feature)",
                       "certificate": "none", **ev2, **(arrival_among(df, acc2) if crit == "1m" else {})})
            dec[c][f"acc_legacy_{m}_{crit}"] = acc2
        for cand in f["candidates"]:
            if cand["source"] in ("dev_q0.25", "dev_q0.50", "dev_q0.75"):
                ev3, _ = rule_eval(df, m, crit, cand["threshold"])
                rc.append({"kind": f"matched_workload_{cand['source']}", "family_id": fid, "cell": c, "method": m,
                           "criterion": crit, "threshold": cand["threshold"], "rule_source": cand["source"],
                           "certificate": "none", **ev3})
            ev4, _ = rule_eval(df, m, crit, cand["threshold"])
            rc.append({"kind": "test_curve_descriptive", "family_id": fid, "cell": c, "method": m, "criterion": crit,
                       "threshold": cand["threshold"], "rule_source": cand["source"], "certificate": "none",
                       "note": "descriptive test curve; not used for any selection", **ev4})
    # ---- paired comparisons --------------------------------------------------------------------
    for c, df in data.items():
        kind = "continuous" if CELLS[c]["cmd_mode"] == "continuous" else "held"
        for crit in ("1m", "0.1m"):
            good = (df[f"label_{crit}"] == GOOD).to_numpy()
            for a, b in PAIRS[kind]:
                for mode in ("selected_calibrated", "dev_q0.25", "dev_q0.50", "dev_q0.75"):
                    if mode == "selected_calibrated":
                        ta, tb = sel[f"{c}|{a}|{crit}"]["threshold"], sel[f"{c}|{b}|{crit}"]["threshold"]
                    else:
                        ca = [x for x in fams[f"{c}|{a}|{crit}"]["candidates"] if x["source"] == mode]
                        cb = [x for x in fams[f"{c}|{b}|{crit}"]["candidates"] if x["source"] == mode]
                        if not ca or not cb:
                            continue
                        ta, tb = ca[0]["threshold"], cb[0]["threshold"]
                    _, aa = rule_eval(df, a, crit, ta)
                    _, ab = rule_eval(df, b, crit, tb)
                    bt = CAL.paired_bootstrap_diff(aa, ab, good, n_boot=10000, seed=20261013)
                    b01, b10 = int((aa & ~ab).sum()), int((~aa & ab).sum())
                    pc.append({"cell": c, "criterion": crit, "comparison": f"{a} vs {b}", "rule_mode": mode,
                               "primary": (c == "D0" and crit == "1m" and (a, b) == ("A", "B") and mode == "selected_calibrated"),
                               "paired_cases": True, "n": int(len(df)), "threshold_a": ta, "threshold_b": tb,
                               "accepted_a": int(aa.sum()), "accepted_b": int(ab.sum()),
                               "adverse_accepted_a": int((aa & ~good).sum()), "adverse_accepted_b": int((ab & ~good).sum()),
                               "only_a": b01, "only_b": b10, "only_a_adverse": int((aa & ~ab & ~good).sum()),
                               "only_b_adverse": int((~aa & ab & ~good).sum()),
                               "mcnemar_exact_p": CAL.mcnemar_exact(b01, b10), **bt})
    # ---- transfer matrix -----------------------------------------------------------------------
    for c, df in data.items():
        Rt = R_E + CELLS[c]["altitude"]
        held = CELLS[c]["cmd_mode"] == "sampled"
        for crit in ("1m", "0.1m"):
            for src_m, tgt_ms in ((("A", ("A_H", "A_naive")) if held else ("A", ("A",))),
                                  (("B", ("B_H",)) if held else ("B", ("B",)))):
                s0 = sel[f"D0|{src_m}|{crit}"]
                for tgt_m in tgt_ms:
                    if s0["threshold"] is not None:
                        mode = "T1" if not held else "T1_heuristic"
                        ev, _ = rule_eval(df, tgt_m, crit, s0["threshold"])
                        tm.append({"target_cell": c, "criterion": crit, "source_method": src_m, "target_method": tgt_m,
                                   "mode": mode, "threshold": s0["threshold"], "certificate": "none (no target calibration)",
                                   "note": ("D0 cutoff attached to a different feature definition" if held else
                                            "D0 cutoff unchanged (metres)") if c != "D0" else "source domain itself", **ev})
                        if c in ("D1", "D2"):
                            t2 = s0["threshold"] * Rt / R0
                            ev, _ = rule_eval(df, tgt_m, crit, t2)
                            tm.append({"target_cell": c, "criterion": crit, "source_method": src_m, "target_method": tgt_m,
                                       "mode": "T2", "threshold": t2, "certificate": "none",
                                       "note": "similarity image c R_t/R_0, frozen before target data", **ev})
                    st = sel[f"{c}|{tgt_m}|{crit}"]
                    ev, _ = rule_eval(df, tgt_m, crit, st["threshold"])
                    tm.append({"target_cell": c, "criterion": crit, "source_method": src_m, "target_method": tgt_m,
                               "mode": "T3", "threshold": st["threshold"], "selection_status": st["status"],
                               "certificate": "own calibration family" if st["threshold"] is not None else "abstain",
                               "note": "target-recalibrated; not evidence of out-of-domain validity of the D0 rule", **ev})
    # ---- hybrid screening and controller slices ------------------------------------------------
    for c in ("D3", "D4"):
        df = data[c]
        for m in ("A_H", "B_H", "A_naive", "C_H"):
            row = {"cell": c, "Ts_s": CELLS[c]["Ts"], "method": m, "population": "test (iid)", **prediction_stats(df, m)}
            for crit in ("1m", "0.1m"):
                s = sel[f"{c}|{m}|{crit}"]
                ev, _ = rule_eval(df, m, crit, s["threshold"])
                row.update({f"{crit}_threshold": s["threshold"], f"{crit}_selection": s["status"],
                            f"{crit}_coverage": ev["coverage"], f"{crit}_accepted": ev["n_accepted"],
                            f"{crit}_known_adverse": ev["known_adverse_accepted"],
                            f"{crit}_conservative_adverse": ev["conservative_adverse_accepted"]})
            hs.append(row)
        f1, f2 = df.f_A_H.to_numpy(float), df.f_A_naive.to_numpy(float)
        ok = np.isfinite(f1) & np.isfinite(f2) & (f1 > 0)
        rel = np.abs(f1 - f2) / f1
        x = {"wn_Ts": (df.wn_ratio * df.n_radps * CELLS[c]["Ts"]).to_numpy(float),
             "sat_fraction": df.sat_fraction.to_numpy(float), "n_status_changes": df.n_status_changes.to_numpy(float)}
        hs.append({"cell": c, "Ts_s": CELLS[c]["Ts"], "method": "A_H vs A_naive", "population": "test (iid)",
                   "rel_diff_q50": float(np.median(rel[ok])), "rel_diff_q95": float(np.quantile(rel[ok], 0.95)),
                   "rel_diff_max": float(rel[ok].max()),
                   **{f"spearman_reldiff_{k}": float(spearmanr(rel[ok], v[ok]).correlation) for k, v in x.items()},
                   "n_near_switch_cases": int((df.n_near_switch_updates.fillna(0) > 0).sum())})
    df = data["D5"]
    for m in ("A", "B", "C", "sep", "sat"):
        row = {"cell": "D5", "controller": "CW LQR (Bryson scaling)", "method": m, **prediction_stats(df, m)}
        for crit in ("1m", "0.1m"):
            s = sel[f"D5|{m}|{crit}"]
            ev, _ = rule_eval(df, m, crit, s["threshold"])
            row.update({f"{crit}_T3_threshold": s["threshold"], f"{crit}_T3_coverage": ev["coverage"],
                        f"{crit}_T3_known_adverse": ev["known_adverse_accepted"],
                        f"{crit}_T3_conservative_adverse": ev["conservative_adverse_accepted"]})
            s0 = sel[f"D0|{m}|{crit}"]
            ev0, _ = rule_eval(df, m, crit, s0["threshold"])
            row.update({f"{crit}_T1_threshold": s0["threshold"], f"{crit}_T1_coverage": ev0["coverage"],
                        f"{crit}_T1_known_adverse": ev0["known_adverse_accepted"],
                        f"{crit}_T1_conservative_adverse": ev0["conservative_adverse_accepted"]})
        ct.append(row)
    for c in ("D0",):
        for m in ("A", "B", "C"):
            ct.append({"cell": c, "controller": "scalar PD (reference)", "method": m, **prediction_stats(data[c], m)})
    outs = {"risk_coverage.csv": rc, "paired_comparisons.csv": pc, "transfer_matrix.csv": tm,
            "hybrid_screening.csv": hs, "controller_transfer.csv": ct}
    hashes = {}
    for name, rows in outs.items():
        p = os.path.join(OUTR, name)
        pd.DataFrame(rows).to_csv(p, index=False)
        hashes[name] = L.file_sha256(p)
    dpath = os.path.join(VD.V3, "raw", "test_decisions.csv")
    pd.concat(dec.values(), ignore_index=True).to_csv(dpath, index=False)
    hashes["test_decisions.csv"] = L.file_sha256(dpath)
    rec = json.load(open(RECORD)) if os.path.exists(RECORD) else {"first_evaluated_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                                  "regenerations": []}
    if "outputs" in rec:
        rec["regenerations"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                     "identical_outputs": rec["outputs"] == hashes})
    rec.update({"plan_sha256": L.file_sha256(PLAN), "freeze_sha256": L.file_sha256(FREEZE),
                "test_results_sha256": {c: L.file_sha256(os.path.join(VD.RES, f"{c}_test.jsonl")) for c in data}})
    rec.setdefault("outputs", hashes)
    json.dump(rec, open(RECORD, "w"), indent=1)
    prim = pd.DataFrame(pc)
    print(prim[prim.primary].T.to_string())


if __name__ == "__main__":
    main()
