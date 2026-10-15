"""P09 practical cost: repeated single-process timings of complete workflows and break-even analysis.

Timing protocol: one process, BLAS threads pinned to 1, one warm-up evaluation per case, then R
repetitions; both CPU time (process_time) and wall time (perf_counter) are recorded per component;
distributions (median, quartiles, p95) are reported, not a single favourable number. Cases: the first
60 D0 development cases and the first 40 D3 and D4 development cases (development data: no
calibration or test labels are needed for timing).

Components (continuous D0, production code): CW, NERM and ECI propagation; A, B, C features;
evaluation of one branch's outputs (trajectory on the 0.5 s grid, arrival label) for CW and for NERM.
Workflows per case, each producing the same outputs (a trajectory and an arrival label at the 1 m/0.01 m/s
criteria; the screen's accepted output is the CW trajectory, a referral is answered by NERM):
  direct_NRM        NERM propagation + evaluation                         (always nonlinear)
  direct_ECI        ECI propagation + evaluation                          (slower reference)
  screen_<F>_cold   CW + feature F + decision; accepted -> CW evaluation; referred -> NERM + evaluation
  screen_<F>_warm   the CW trajectory and its evaluation exist already (another use): feature F,
                    and NERM + evaluation for referred cases only
Held cells (batch engine, N = 40 cases per batch, amortised per case): CW + A_H batch vs NERM batch.
Expected online cost uses the referral fraction of the frozen selected rule on the TEST set and the
case-dependent costs measured here, split by each benchmark case's own decision under that rule.
Offline cost = CPU time of the development and calibration labelling runs of the cell (ledger).
Break-even N* = offline / (direct - screen) when the online difference is positive; otherwise there
is no positive finite break-even.

Usage: python code/experiments/v3/p09_cost.py [--reps 3]
Output: results/cost_break_even.csv, raw/cost_timings.csv, manuscript_assets/FigV3_5_Cost.pdf (if informative)
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v3data as VD  # noqa: E402
from cwvalid3 import ledger as L  # noqa: E402


def timed(fn, reps):
    fn()  # warm-up
    cpu, wall = [], []
    out = None
    for _ in range(reps):
        c0, w0 = time.process_time(), time.perf_counter()
        out = fn()
        cpu.append(time.process_time() - c0)
        wall.append(time.perf_counter() - w0)
    return out, float(np.median(cpu)), float(np.median(wall))


def bench_continuous(rec, reps):
    from cwvalid.estimate import first_order_discrepancy
    from cwvalid.estimate_v2 import residual_only_diagnostic, unsaturated_jacobian_estimate
    from cwvalid.experiment import RunConfig, evaluate, make_law
    from cwvalid.scenario import Scenario
    from cwvalid.simulate import EciConfig, SolverConfig, simulate
    s = Scenario.from_dict(rec["scenario"])
    law = make_law(s)
    from cwvalid.control import CommandSpec
    cmd = CommandSpec("continuous", law=law)
    sol = SolverConfig()
    row = {"case_id": rec["case_id"], "cell": rec["cell"]}
    sims = {}
    for m in ("cw", "nerm", "eci"):
        sims[m], c, w = timed(lambda m=m: simulate(m, s.rel0, cmd, s.horizon, n=s.n, R=s.R, solver=sol,
                                                   eci=EciConfig() if m == "eci" else None), reps)
        row[f"cpu_prop_{m}"], row[f"wall_prop_{m}"] = c, w
    for m in ("cw", "nerm", "eci"):
        cfg = RunConfig(models=(m,), reference=m, crosscheck=None)
        _, c, w = timed(lambda m=m, cfg=cfg: evaluate(s, cfg, {m: sims[m]}), reps)
        row[f"cpu_eval_{m}"], row[f"wall_eval_{m}"] = c, w
    for k, fn in (("A", lambda: first_order_discrepancy(sims["cw"], law, s.n, s.R, s.horizon)),
                  ("B", lambda: unsaturated_jacobian_estimate(sims["cw"], law, s.n, s.R, s.horizon)),
                  ("C", lambda: residual_only_diagnostic(sims["cw"], s.n, s.R, s.horizon))):
        _, c, w = timed(fn, reps)
        row[f"cpu_feat_{k}"], row[f"wall_feat_{k}"] = c, w
    _, c, w = timed(lambda: Scenario.from_dict(rec["scenario"]), reps)
    row["cpu_scenario"], row["wall_scenario"] = c, w
    return row


def bench_held(recs, reps):
    from cwvalid3.held_metrics import CaseView, estimator_feature, terminal_arrival
    from cwvalid3.sampled import HeldBatch
    from cwvalid3.domains import CELLS
    cell = recs[0]["cell"]
    Ts, alt = CELLS[cell]["Ts"], CELLS[cell]["altitude"]
    rel0 = [r["scenario"]["rel0"] for r in recs]
    K = np.array([r["K"] for r in recs])
    am = [r["scenario"]["a_max"] for r in recs]
    N = len(recs)

    def screen():
        hb = HeldBatch(rel0, K, am, alt, Ts, h=min(Ts, 2.0), models=("cw",), estimators=("A_H",)).run()
        for i in range(N):
            cv = CaseView(hb, i)
            estimator_feature(cv, "A_H")
            terminal_arrival(cv, "cw")
        return hb

    def direct():
        hb = HeldBatch(rel0, K, am, alt, Ts, h=min(Ts, 2.0), models=("nerm",), estimators=()).run()
        for i in range(N):
            terminal_arrival(CaseView(hb, i), "nerm")
        return hb
    _, cs, ws = timed(screen, reps)
    _, cd, wd = timed(direct, reps)
    return {"cell": cell, "Ts_s": Ts, "batch_N": N, "cpu_per_case_screen_cw_AH": cs / N, "wall_per_case_screen_cw_AH": ws / N,
            "cpu_per_case_direct_nerm": cd / N, "wall_per_case_direct_nerm": wd / N}


def main():
    import v30_closed  # v3.1: preserved v3.0 outputs; runs only inside the rebuild harness
    v30_closed.refuse_outside_harness(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-cont", type=int, default=60)
    ap.add_argument("--n-held", type=int, default=40)
    a = ap.parse_args()
    man = lambda c: L.read_jsonl_strict(os.path.join(VD.MAN, f"{c}_dev.jsonl"), key="run_id")
    t0 = time.perf_counter()
    rows = [bench_continuous(r, a.reps) for r in man("D0")[:a.n_cont]]
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(VD.V3, "raw", "cost_timings.csv"), index=False)
    held = [bench_held(man(c)[:a.n_held], a.reps) for c in ("D3", "D4")]
    # ---- workflows ---------------------------------------------------------------------------
    frz = json.load(open(os.path.join(VD.V3, "raw", "calibration_freeze.json")))["selection"]
    dev, _ = VD.load_split("D0", "dev")
    test, _ = VD.load_split("D0", "test")
    T = T.merge(dev[["case_id", "f_A", "f_B", "f_C"]], on="case_id", how="left")
    out = []
    direct_nrm = T.cpu_prop_nerm + T.cpu_eval_nerm
    direct_eci = T.cpu_prop_eci + T.cpu_eval_eci
    for k in ("A", "B", "C"):
        thr = frz[f"D0|{k}|1m"]["threshold"]
        acc_bench = (T[f"f_{k}"] <= thr).to_numpy() if thr is not None else np.zeros(len(T), bool)
        p_ref_test = float(1 - np.mean(np.isfinite(test[f"f_{k}"]) & (test[f"f_{k}"] <= thr))) if thr is not None else 1.0
        acc_cost_cold = T.cpu_prop_cw + T[f"cpu_feat_{k}"] + T.cpu_eval_cw
        ref_cost_cold = T.cpu_prop_cw + T[f"cpu_feat_{k}"] + T.cpu_prop_nerm + T.cpu_eval_nerm
        acc_cost_warm = T[f"cpu_feat_{k}"]
        ref_cost_warm = T[f"cpu_feat_{k}"] + T.cpu_prop_nerm + T.cpu_eval_nerm
        mean_acc = lambda s: float(s[acc_bench].mean()) if acc_bench.any() else float(s.mean())
        mean_ref = lambda s: float(s[~acc_bench].mean()) if (~acc_bench).any() else float(s.mean())
        for mode, ca, cr in (("cold", acc_cost_cold, ref_cost_cold), ("warm_cw_available", acc_cost_warm, ref_cost_warm)):
            e_screen = (1 - p_ref_test) * mean_acc(ca) + p_ref_test * mean_ref(cr)
            for ref_name, direct in (("direct_NRM", direct_nrm), ("direct_ECI", direct_eci)):
                e_direct = float(direct.mean())
                diff = e_direct - e_screen
                out.append({"cell": "D0", "feature": k, "rule_threshold": thr, "workflow": f"screen_{k}_{mode}",
                            "comparator": ref_name, "p_referral_test": p_ref_test,
                            "expected_cpu_s_per_case_screen": e_screen, "expected_cpu_s_per_case_comparator": e_direct,
                            "online_saving_cpu_s_per_case": diff,
                            "screen_cost_q25_q50_q75": json.dumps([float(np.quantile(ca, q)) for q in (0.25, 0.5, 0.75)]),
                            "comparator_cost_q25_q50_q75": json.dumps([float(np.quantile(direct, q)) for q in (0.25, 0.5, 0.75)]),
                            "n_benchmark_cases": len(T), "reps": a.reps})
    led = L.read_jsonl_strict(os.path.join(VD.V3, "RUN_LEDGER.jsonl"), key="ledger_id")
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
    df.to_csv(os.path.join(VD.V3, "results", "cost_break_even.csv"), index=False)
    comp = T[[c for c in T.columns if c.startswith("cpu_")]].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
    comp.to_csv(os.path.join(VD.V3, "results", "cost_components.csv"))
    print(comp.to_string())
    print(df[["cell", "workflow", "comparator", "expected_cpu_s_per_case_screen", "expected_cpu_s_per_case_comparator",
              "online_saving_cpu_s_per_case"]].to_string())
    print(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
