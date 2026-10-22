import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v3data as VD
from cwvalid3 import ledger as L


def timed(fn, reps):
    fn()
    cpu, wall = [], []
    out = None
    for _ in range(reps):
        c0, w0 = time.process_time(), time.perf_counter()
        out = fn()
        cpu.append(time.process_time() - c0)
        wall.append(time.perf_counter() - w0)
    return out, float(np.median(cpu)), float(np.median(wall))


def benchcont(rec, reps):
    from cwvalid.estimate import firstorderdisc
    from cwvalid.estimate2 import residonlydiag, unsatjacest
    from cwvalid.experiment import RunCfg, evaluate, makelaw
    from cwvalid.scenario import Scenario
    from cwvalid.simulate import EciCfg, SolverCfg, simulate
    s = Scenario.from_dict(rec["scenario"])
    law = makelaw(s)
    from cwvalid.control import CmdSpec
    cmd = CmdSpec("continuous", law=law)
    sol = SolverCfg()
    row = {"case_id": rec["case_id"], "cell": rec["cell"]}
    sims = {}
    for m in ("cw", "nerm", "eci"):
        sims[m], c, w = timed(lambda m=m: simulate(m, s.rel0, cmd, s.horizon, n=s.n, R=s.R, solver=sol,
                                                   eci=EciCfg() if m == "eci" else None), reps)
        row[f"cpu_prop_{m}"], row[f"wall_prop_{m}"] = c, w
    for m in ("cw", "nerm", "eci"):
        cfg = RunCfg(models=(m,), reference=m, crosscheck=None)
        _, c, w = timed(lambda m=m, cfg=cfg: evaluate(s, cfg, {m: sims[m]}), reps)
        row[f"cpu_eval_{m}"], row[f"wall_eval_{m}"] = c, w
    for k, fn in (("A", lambda: firstorderdisc(sims["cw"], law, s.n, s.R, s.horizon)),
                  ("B", lambda: unsatjacest(sims["cw"], law, s.n, s.R, s.horizon)),
                  ("C", lambda: residonlydiag(sims["cw"], s.n, s.R, s.horizon))):
        _, c, w = timed(fn, reps)
        row[f"cpu_feat_{k}"], row[f"wall_feat_{k}"] = c, w
    _, c, w = timed(lambda: Scenario.from_dict(rec["scenario"]), reps)
    row["cpu_scenario"], row["wall_scenario"] = c, w
    return row


def benchheld(recs, reps):
    from cwvalid3.heldmet import CaseView, estimfeat, termarr
    from cwvalid3.sampled import HeldBatch
    from cwvalid3.domains import CELLS
    cell = recs[0]["cell"]
    Ts, alt = CELLS[cell]["Ts"], CELLS[cell]["altitude"]
    rel0 = [r["scenario"]["rel0"] for r in recs]
    K = np.array([r["K"] for r in recs])
    am = [r["scenario"]["a_max"] for r in recs]
    N = len(recs)

    def screen():
        hb = HeldBatch(rel0, K, am, alt, Ts, h=min(Ts, 2.0), models=("cw",), ests=("A_H",)).run()
        for i in range(N):
            cv = CaseView(hb, i)
            estimfeat(cv, "A_H")
            termarr(cv, "cw")
        return hb

    def direct():
        hb = HeldBatch(rel0, K, am, alt, Ts, h=min(Ts, 2.0), models=("nerm",), ests=()).run()
        for i in range(N):
            termarr(CaseView(hb, i), "nerm")
        return hb
    _, cs, ws = timed(screen, reps)
    _, cd, wd = timed(direct, reps)
    return {"cell": cell, "Ts_s": Ts, "batch_N": N, "cpu_per_case_screen_cw_AH": cs / N, "wall_per_case_screen_cw_AH": ws / N,
            "cpu_per_case_direct_nerm": cd / N, "wall_per_case_direct_nerm": wd / N}


def main():
    import v30closed
    v30closed.refuseoutsharness(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-cont", type=int, default=60)
    ap.add_argument("--n-held", type=int, default=40)
    a = ap.parse_args()
    man = lambda c: L.readjsonlstrict(os.path.join(VD.MAN, f"{c}dev.jsonl"), key="run_id")
    t0 = time.perf_counter()
    rows = [benchcont(r, a.reps) for r in man("D0")[:a.n_cont]]
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(VD.V3, "raw", "costtimings.csv"), index=False)
    held = [benchheld(man(c)[:a.n_held], a.reps) for c in ("D3", "D4")]
    frz = json.load(open(os.path.join(VD.V3, "raw", "freeze.json")))["selection"]
    dev, _ = VD.loadsplit("D0", "dev")
    test, _ = VD.loadsplit("D0", "test")
    T = T.merge(dev[["case_id", "f_A", "f_B", "f_C"]], on="case_id", how="left")
    out = []
    directnrm = T.cpu_prop_nerm + T.cpu_eval_nerm
    directeci = T.cpu_prop_eci + T.cpu_eval_eci
    for k in ("A", "B", "C"):
        thr = frz[f"D0|{k}|1m"]["threshold"]
        accbench = (T[f"f_{k}"] <= thr).to_numpy() if thr is not None else np.zeros(len(T), bool)
        preftest = float(1 - np.mean(np.isfinite(test[f"f_{k}"]) & (test[f"f_{k}"] <= thr))) if thr is not None else 1.0
        acccold = T.cpu_prop_cw + T[f"cpu_feat_{k}"] + T.cpu_eval_cw
        refcold = T.cpu_prop_cw + T[f"cpu_feat_{k}"] + T.cpu_prop_nerm + T.cpu_eval_nerm
        accwarm = T[f"cpu_feat_{k}"]
        refwarm = T[f"cpu_feat_{k}"] + T.cpu_prop_nerm + T.cpu_eval_nerm
        meanacc = lambda s: float(s[accbench].mean()) if accbench.any() else float(s.mean())
        meanref = lambda s: float(s[~accbench].mean()) if (~accbench).any() else float(s.mean())
        for mode, ca, cr in (("cold", acccold, refcold), ("warm_cw_available", accwarm, refwarm)):
            escreen = (1 - preftest) * meanacc(ca) + preftest * meanref(cr)
            for refname, direct in (("direct_NRM", directnrm), ("direct_ECI", directeci)):
                edirect = float(direct.mean())
                diff = edirect - escreen
                out.append({"cell": "D0", "feature": k, "rule_threshold": thr, "workflow": f"screen_{k}_{mode}",
                            "comparator": refname, "p_referral_test": preftest,
                            "expected_cpu_s_per_case_screen": escreen, "expected_cpu_s_per_case_comparator": edirect,
                            "online_saving_cpu_s_per_case": diff,
                            "screen_cost_q25_q50_q75": json.dumps([float(np.quantile(ca, q)) for q in (0.25, 0.5, 0.75)]),
                            "comparator_cost_q25_q50_q75": json.dumps([float(np.quantile(direct, q)) for q in (0.25, 0.5, 0.75)]),
                            "n_benchmark_cases": len(T), "reps": a.reps})
    led = L.readjsonlstrict(os.path.join(VD.V3, "ledger.jsonl"), key="ledger_id")
    off = sum((e.get("cpu_s") or 0.0) for e in led if e["event"] == "completed" and e["cell"] == "D0"
              and e["split"] in ("pilot", "dev", "cal"))
    for r in out:
        r["offline_calibration_cpu_s_D0"] = off
        r["break_even_cases"] = (off / r["online_saving_cpu_s_per_case"]) if r["online_saving_cpu_s_per_case"] > 0 else None
        r["break_even_note"] = ("finite: online saving positive" if r["online_saving_cpu_s_per_case"] > 0 else
                                "no positive finite break-even: screening costs more online than the comparator")
    for h in held:
        out.append({"cell": h["cell"], "feature": "A_H", "workflow": "screen_A_H_batch_amortised", "comparator": "direct_NRM_batch",
                    "expected_cpu_s_per_case_screen": h["cpu_per_case_screen_cw_AH"],
                    "expected_cpu_s_per_case_comparator": h["cpu_per_case_direct_nerm"],
                    "online_saving_cpu_s_per_case": h["cpu_per_case_direct_nerm"] - h["cpu_per_case_screen_cw_AH"],
                    "note": f"batch of {h['batch_N']} cases, per-case amortised; screen cost excludes referrals",
                    "reps": a.reps})
    df = pd.DataFrame(out)
    df.insert(0, "hardware", f"{os.cpu_count()} logical CPUs, single process, BLAS threads 1; machine not otherwise idle")
    df.to_csv(os.path.join(VD.V3, "results", "breakeven.csv"), index=False)
    comp = T[[c for c in T.columns if c.startswith("cpu_")]].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
    comp.to_csv(os.path.join(VD.V3, "results", "costcomponents.csv"))
    print(comp.to_string())
    print(df[["cell", "workflow", "comparator", "expected_cpu_s_per_case_screen", "expected_cpu_s_per_case_comparator",
              "online_saving_cpu_s_per_case"]].to_string())
    print(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
