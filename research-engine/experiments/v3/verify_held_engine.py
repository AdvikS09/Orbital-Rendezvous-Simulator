"""Verify the batch held-command engine (cwvalid3.sampled) against the production propagation
(cwvalid.simulate in 'sampled' mode, adaptive DOP853 restarted at every update).

For each case and hold period: maximum difference of the relative state on the 0.5 s grid per
branch (cw, so, nerm, eci), peak CW-to-reference discrepancy, terminal-arrival label, entry time
and dwell margin, and branch status; plus timing of both implementations.

Usage: python experiments/v3/verify_held_engine.py [--ts 10 1] [--cases quick|full] [--workers 7]
Output: v3_extension/results/held_engine_verification.csv
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
OUT = os.path.join(os.path.dirname(REPO), "v3_extension", "results")


def cases(kind):
    from cwvalid.scenario import axis_scenario
    from cwvalid.scenario import Scenario
    base = [axis_scenario("v3ver", "+R", 3000.0, 6.0, 0.05), axis_scenario("v3ver", "+H", 20000.0, 6.0, 0.05),
            axis_scenario("v3ver", "-R", 12000.0, 8.0, 0.1), axis_scenario("v3ver", "+V", 500.0, 6.0, 0.05)]
    if kind == "full":
        n = math.sqrt(3.986004418e14 / 6778137.0 ** 3)
        rng = np.random.default_rng(31001)
        for k in range(4):
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            sep = float(np.exp(rng.uniform(np.log(200), np.log(30000))))
            w = rng.normal(size=3); w /= np.linalg.norm(w)
            nu = float(rng.uniform(0, 2))
            rel0 = tuple(np.concatenate((u * sep, w * nu * n * sep)))
            base.append(Scenario("v3ver", rel0, float(rng.uniform(3, 10)), float(np.exp(rng.uniform(np.log(0.005), np.log(0.2)))), 1.0,
                                 meta={"axis": "mixed", "sep_m": sep}))
        base.append(axis_scenario("v3ver", "-H", 50000.0, 4.0, 0.01))
    return base


def production(args):
    scn_d, Ts = args
    from cwvalid.experiment import RunConfig, run_scenario
    from cwvalid.scenario import Scenario
    scn = Scenario.from_dict(scn_d)
    cfg = RunConfig(models=("cw", "so", "nerm", "eci"), reference="eci", crosscheck="nerm", cmd_mode="sampled", sample_period=Ts)
    t0 = time.perf_counter()
    row, hist = run_scenario(scn, cfg, keep_histories=True, hist_stride=1)
    return row, {k: v for k, v in hist.items()}, time.perf_counter() - t0


def batch(scns, Ts):
    from cwvalid3.sampled import HeldBatch, pd_matrix
    from cwvalid3.held_metrics import case_summary
    t0 = time.perf_counter()
    K = np.stack([pd_matrix(s.wn_ratio * s.n) for s in scns])
    # same step rule as the v3 campaign (experiments/v3/campaign.py): h = min(Ts, 2 s)
    hb = HeldBatch([s.rel0 for s in scns], K, [s.a_max for s in scns], scns[0].altitude, Ts,
                   horizon_orbits=scns[0].horizon_orbits, h=min(Ts, 2.0)).run()
    wall = time.perf_counter() - t0
    summ, grids = [], []
    for i in range(len(scns)):
        sm, cv = case_summary(hb, i)
        summ.append(sm)
        grids.append({b: cv.grid(b) for b in ("cw", "so", "nerm", "eci")})
    hb.t = hb.t  # step boundaries
    hb.tm = cv.tm
    return hb, summ, grids, wall


def main():
    import v30_closed  # v3.1: preserved v3.0 outputs; runs only inside the rebuild harness
    v30_closed.refuse_outside_harness(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", type=float, nargs="+", default=[10.0, 1.0])
    ap.add_argument("--cases", default="quick")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    scns = cases(a.cases)
    rows = []
    for Ts in a.ts:
        hb, summ, grids, wall_b = batch(scns, Ts)
        with ProcessPoolExecutor(a.workers) as ex:
            prod = list(ex.map(production, [(s.to_dict(), Ts) for s in scns]))
        for i, (s, (prow, phist, wall_p)) in enumerate(zip(scns, prod)):
            r = {"sid": s.sid, "Ts_s": Ts, "sep_m": s.separation, "wn_ratio": s.wn_ratio, "a_max": s.a_max,
                 "wall_batch_per_case_s": wall_b / len(scns), "wall_production_s": wall_p}
            for b in ("cw", "so", "nerm", "eci"):
                Sp = phist[f"{b}_S"]
                Sb = grids[i][b]
                m = min(len(Sp), len(Sb))
                same_t = np.allclose(phist[f"{b}_t"][:m], hb.tm[:m], atol=1e-9)
                r[f"{b}_max_pos_diff_m"] = float(np.max(np.linalg.norm(Sp[:m, :3] - Sb[:m, :3], axis=1))) if same_t else float("nan")
                r[f"{b}_max_vel_diff_mps"] = float(np.max(np.linalg.norm(Sp[:m, 3:] - Sb[:m, 3:], axis=1))) if same_t else float("nan")
                r[f"{b}_max_range_m"] = float(np.max(np.linalg.norm(Sp[:m, :3], axis=1)))
                stp = {"ok": "completed", "terminated": "domain_exit", "failed": "numerical_failure"}[prow[f"{b}_status"]]
                r[f"{b}_status_prod"], r[f"{b}_status_batch"] = stp, summ[i][f"{b}_status"]
                r[f"{b}_arrived_prod"], r[f"{b}_arrived_batch"] = bool(prow[f"{b}_terminal_arrived"]), bool(summ[i][f"{b}_arrived"])
                r[f"{b}_entry_diff_s"] = abs(float(prow[f"{b}_terminal_arrival_time_s"] or np.nan) - summ[i][f"{b}_entry_time_s"])
                r[f"{b}_t_end_diff_s"] = abs(float(prow[f"{b}_t_final_s"]) - summ[i][f"{b}_t_end_s"])
            for a_, ref in (("cw", "eci"), ("so", "eci"), ("cw", "nerm")):
                pk_p = float(prow[f"{a_}_vs_{ref}_max_pos_disc_m"])
                pk_b = summ[i][f"{a_}_vs_{ref}_peak_m"]
                r[f"{a_}_vs_{ref}_peak_prod_m"], r[f"{a_}_vs_{ref}_peak_batch_m"] = pk_p, pk_b
                r[f"{a_}_vs_{ref}_peak_absdiff_m"] = abs(pk_p - pk_b)
            r["batch_nerm_vs_eci_peak_m"] = summ[i]["nerm_vs_eci_peak_m"]
            rows.append(r)
            print(f"Ts={Ts:g} {s.sid}: cw {r['cw_max_pos_diff_m']:.2e} eci {r['eci_max_pos_diff_m']:.2e} so {r['so_max_pos_diff_m']:.2e} "
                  f"peak diff {r['cw_vs_eci_peak_absdiff_m']:.2e} labels "
                  f"{[(r[f'{b}_arrived_prod'], r[f'{b}_arrived_batch']) for b in ('cw', 'eci', 'so')]} "
                  f"wall batch {r['wall_batch_per_case_s']:.2f}s prod {wall_p:.1f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "held_engine_verification.csv"), index=False)


if __name__ == "__main__":
    main()
