import argparse
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
OUT = os.path.join(os.path.dirname(REPO), "v3", "results")


def cases(kind):
    from cwvalid.scenario import axisscen
    from cwvalid.scenario import Scenario
    base = [axisscen("v3ver", "+R", 3000.0, 6.0, 0.05), axisscen("v3ver", "+H", 20000.0, 6.0, 0.05),
            axisscen("v3ver", "-R", 12000.0, 8.0, 0.1), axisscen("v3ver", "+V", 500.0, 6.0, 0.05)]
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
        base.append(axisscen("v3ver", "-H", 50000.0, 4.0, 0.01))
    return base


def prod2(args):
    scnd, Ts = args
    from cwvalid.experiment import RunCfg, runscen
    from cwvalid.scenario import Scenario
    scn = Scenario.from_dict(scnd)
    cfg = RunCfg(models=("cw", "so", "nerm", "eci"), reference="eci", crosscheck="nerm", cmd_mode="sampled", sample_period=Ts)
    t0 = time.perf_counter()
    row, hist = runscen(scn, cfg, keep_histories=True, hist_stride=1)
    return row, {k: v for k, v in hist.items()}, time.perf_counter() - t0


def batch(scns, Ts):
    from cwvalid3.sampled import HeldBatch, pdmatrix
    from cwvalid3.heldmet import casesumm
    t0 = time.perf_counter()
    K = np.stack([pdmatrix(s.wn_ratio * s.n) for s in scns])
    hb = HeldBatch([s.rel0 for s in scns], K, [s.a_max for s in scns], scns[0].altitude, Ts,
                   horizon_orbits=scns[0].horizon_orbits, h=min(Ts, 2.0)).run()
    wall = time.perf_counter() - t0
    summ, grids = [], []
    for i in range(len(scns)):
        sm, cv = casesumm(hb, i)
        summ.append(sm)
        grids.append({b: cv.grid(b) for b in ("cw", "so", "nerm", "eci")})
    hb.t = hb.t
    hb.tm = cv.tm
    return hb, summ, grids, wall


def main():
    import v30closed
    v30closed.refuseoutsharness(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", type=float, nargs="+", default=[10.0, 1.0])
    ap.add_argument("--cases", default="quick")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    scns = cases(a.cases)
    rows = []
    for Ts in a.ts:
        hb, summ, grids, wallb = batch(scns, Ts)
        with ProcessPoolExecutor(a.workers) as ex:
            prod = list(ex.map(prod2, [(s.to_dict(), Ts) for s in scns]))
        for i, (s, (prow, phist, wallp)) in enumerate(zip(scns, prod)):
            r = {"sid": s.sid, "Ts_s": Ts, "sep_m": s.separation, "wn_ratio": s.wn_ratio, "a_max": s.a_max,
                 "wall_batch_per_case_s": wallb / len(scns), "wall_production_s": wallp}
            for b in ("cw", "so", "nerm", "eci"):
                Sp = phist[f"{b}_S"]
                Sb = grids[i][b]
                m = min(len(Sp), len(Sb))
                samet = np.allclose(phist[f"{b}_t"][:m], hb.tm[:m], atol=1e-9)
                r[f"{b}_max_pos_diff_m"] = float(np.max(np.linalg.norm(Sp[:m, :3] - Sb[:m, :3], axis=1))) if samet else float("nan")
                r[f"{b}_max_vel_diff_mps"] = float(np.max(np.linalg.norm(Sp[:m, 3:] - Sb[:m, 3:], axis=1))) if samet else float("nan")
                r[f"{b}_max_range_m"] = float(np.max(np.linalg.norm(Sp[:m, :3], axis=1)))
                stp = {"ok": "completed", "terminated": "domain_exit", "failed": "numerical_failure"}[prow[f"{b}_status"]]
                r[f"{b}_status_prod"], r[f"{b}_status_batch"] = stp, summ[i][f"{b}_status"]
                r[f"{b}_arrived_prod"], r[f"{b}_arrived_batch"] = bool(prow[f"{b}_terminal_arrived"]), bool(summ[i][f"{b}_arrived"])
                r[f"{b}_entry_diff_s"] = abs(float(prow[f"{b}_terminal_arrival_time_s"] or np.nan) - summ[i][f"{b}_entry_time_s"])
                r[f"{b}_t_end_diff_s"] = abs(float(prow[f"{b}_t_final_s"]) - summ[i][f"{b}_t_end_s"])
            for a_, ref in (("cw", "eci"), ("so", "eci"), ("cw", "nerm")):
                pkp = float(prow[f"{a_}_vs_{ref}_max_pos_disc_m"])
                pkb = summ[i][f"{a_}_vs_{ref}_peak_m"]
                r[f"{a_}_vs_{ref}_peak_prod_m"], r[f"{a_}_vs_{ref}_peak_batch_m"] = pkp, pkb
                r[f"{a_}_vs_{ref}_peak_absdiff_m"] = abs(pkp - pkb)
            r["batch_nerm_vs_eci_peak_m"] = summ[i]["nerm_vs_eci_peak_m"]
            rows.append(r)
            print(f"Ts={Ts:g} {s.sid}: cw {r['cw_max_pos_diff_m']:.2e} eci {r['eci_max_pos_diff_m']:.2e} so {r['so_max_pos_diff_m']:.2e} "
                  f"peak diff {r['cw_vs_eci_peak_absdiff_m']:.2e} labels "
                  f"{[(r[f'{b}_arrived_prod'], r[f'{b}_arrived_batch']) for b in ('cw', 'eci', 'so')]} "
                  f"wall batch {r['wall_batch_per_case_s']:.2f}s prod {wallp:.1f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "heldengine.csv"), index=False)


if __name__ == "__main__":
    main()
