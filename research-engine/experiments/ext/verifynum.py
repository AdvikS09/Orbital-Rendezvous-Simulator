import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
OUT = os.path.join(os.path.dirname(ROOT), "results", "verification")

from cwvalid.experiment import RunCfg, evaluate, runbranch
from cwvalid.scenario import Scenario, axisscen
from cwvalid.simulate import SolverCfg


def cases():
    A = axisscen
    c = [
        A("verify", "+R", 100.0, 6, 0.05), A("verify", "-R", 100.0, 6, 0.05),
        A("verify", "+V", 500.0, 6, 0.05), A("verify", "-V", 500.0, 6, 0.05),
        A("verify", "+H", 800.0, 6, 0.05), A("verify", "-H", 800.0, 6, 0.05),
        Scenario("verify", (200.0, -300.0, 80.0, 0, 0, 0), 6, 0.05, meta={"label": "manuscript 369 m mixed"}),
        A("verify", "+R", 3000.0, 6, 0.05), A("verify", "-R", 3000.0, 6, 0.05),
        A("verify", "+V", 4500.0, 6, 0.05, extra_meta={"label": "near 0.1 m along-track"}),
        A("verify", "+H", 4800.0, 6, 0.05, extra_meta={"label": "near 0.1 m cross-track"}),
        A("verify", "+R", 5500.0, 6, 0.05, extra_meta={"label": "near 1 m radial"}),
        A("verify", "+V", 7700.0, 6, 0.05, extra_meta={"label": "near 1 m along-track"}),
        A("verify", "+H", 10000.0, 6, 0.05, extra_meta={"label": "near 1 m cross-track"}),
        A("verify", "+R", 12000.0, 6, 0.05), A("verify", "+R", 14251.0, 6, 0.05),
        A("verify", "-R", 14251.0, 6, 0.05), A("verify", "+V", 26706.0, 6, 0.05),
        A("verify", "+H", 50000.0, 6, 0.05), A("verify", "+R", 50000.0, 6, 0.05),
        Scenario("verify", (-3000.0, 4000.0, 1500.0, 0.5, -1.0, 0.3), 6, 0.05, meta={"label": "mixed with velocity"}),
        Scenario("verify", (8000.0, -6000.0, 3000.0, 0, 0, 0), 6, 0.05, meta={"label": "mixed large"}),
        A("verify", "+R", 5000.0, 6, 0.05, velocity="em1"),
        A("verify", "+R", 5000.0, 6, 0.05, velocity="em_exact"),
        A("verify", "+R", 10000.0, 6, 0.01, extra_meta={"label": "sustained saturation low cap"}),
        A("verify", "+V", 2000.0, 8, 0.10),
        A("verify", "-H", 20000.0, 4, 0.01),
    ]
    return c


def scaledsolver(rtol, max_step=20.0, sat_events=True):
    return SolverCfg(rtol=rtol, atol_rel_pos=rtol * 1e3, atol_rel_vel=rtol * 1.0,
                     atol_eci_pos=rtol * 1e6, atol_eci_vel=rtol * 1e3, atol_dv=rtol,
                     max_step=max_step, sat_events=sat_events)


PRODRTOL = 1e-13
VARIANTS = (
    [("ref", scaledsolver(2.5e-14, 2.0), 0.25)]
    + [(f"rtol{r:.0e}", scaledsolver(r), 0.5) for r in (1e-9, 1e-10, 1e-11, 1e-12, 1e-13)]
    + [("noevents_rtol1e-12", scaledsolver(1e-12, sat_events=False), 0.5),
       ("noevents_rtol1e-13", scaledsolver(1e-13, sat_events=False), 0.5)]
    + [("maxstep_inf", scaledsolver(PRODRTOL, np.inf), 0.5), ("maxstep_5", scaledsolver(PRODRTOL, 5.0), 0.5)]
    + [(f"dteval{d:g}", scaledsolver(PRODRTOL), d) for d in (2.0, 1.0, 0.25)]
)
RK4K = (1000, 2000, 4000, 8000)

KEYS = ["cw_vs_nerm_max_pos_disc_m", "cw_vs_eci_max_pos_disc_m", "eci_vs_nerm_max_pos_disc_m",
        "cw_vs_nerm_max_vel_disc_mps", "cw_vs_nerm_t_max_pos_disc_s", "cw_vs_nerm_max_pos_disc_legacy_grid_m",
        "cw_terminal_arrived", "nerm_terminal_arrived", "eci_terminal_arrived",
        "cw_terminal_arrival_time_s", "nerm_terminal_arrival_time_s", "eci_terminal_arrival_time_s",
        "cw_legacy_arrived", "nerm_legacy_arrived", "eci_legacy_arrived",
        "cw_pos_only_arrival_time_s", "nerm_pos_only_arrival_time_s",
        "cw_dv_aug_mps", "nerm_dv_aug_mps", "eci_dv_aug_mps", "nerm_dv_simpson_mps",
        "cw_sat_total_s", "nerm_sat_total_s", "eci_sat_total_s", "nerm_closest_approach_m",
        "cw_final_pos_m", "nerm_final_pos_m", "nerm_final_speed_mps",
        "cw_nfev", "nerm_nfev", "eci_nfev", "eci_n_sat_switch_events", "nerm_n_sat_switch_events", "cw_n_sat_switch_events", "nerm_termination", "eci_t_final_s", "cw_wall_time_s", "nerm_wall_time_s", "eci_wall_time_s",
        "outcome_terminal_cw", "outcome_legacy_cw", "ambiguity_flags"]


def histdiff(h1, h2, m1, m2, prefix):
    n = min(h1[f"{m1}_t"].size, h2[f"{m2}_t"].size)
    S1, S2 = h1[f"{m1}_S"][:n], h2[f"{m2}_S"][:n]
    U1, U2 = h1[f"{m1}_app"][:n], h2[f"{m2}_app"][:n]
    return {f"{prefix}_max_dpos_m": float(np.linalg.norm(S1[:, :3] - S2[:, :3], axis=1).max()),
            f"{prefix}_max_dvel_mps": float(np.linalg.norm(S1[:, 3:] - S2[:, 3:], axis=1).max()),
            f"{prefix}_max_dcmd_mps2": float(np.linalg.norm(U1 - U2, axis=1).max())}


def runcase(args):
    scnd, outdir = args
    scn = Scenario.from_dict(scnd)
    rows = []
    hists = {}
    for name, solver, dte in VARIANTS:
        cfg = RunCfg(models=("cw", "nerm", "eci"), reference="nerm", crosscheck="eci", solver=solver, dt_eval=dte)
        t0 = time.perf_counter()
        sims, _ = runbranch(scn, cfg)
        row, hist = evaluate(scn, cfg, sims, keep_histories=True)
        r = {"sid": scn.sid, "variant": name, "rtol": solver.rtol, "max_step": solver.max_step,
             "dt_eval": dte, "eval_wall_s": time.perf_counter() - t0}
        r.update({k: row.get(k) for k in KEYS})
        rows.append(r)
        hists[name] = hist
    for k in RK4K:
        dt = scn.horizon / k
        cfg = RunCfg(models=("cw", "nerm", "eci"), reference="nerm", crosscheck="eci",
                     solver=SolverCfg(method="RK4", rk4_dt=dt), dt_eval=0.5)
        t0 = time.perf_counter()
        sims, _ = runbranch(scn, cfg)
        row, hist = evaluate(scn, cfg, sims, keep_histories=True)
        r = {"sid": scn.sid, "variant": f"rk4_T/{k}", "rtol": float("nan"), "max_step": dt, "dt_eval": 0.5,
             "eval_wall_s": time.perf_counter() - t0}
        r.update({k2: row.get(k2) for k2 in KEYS})
        rows.append(r)
        hists[f"rk4_T/{k}"] = hist
    ref = hists["ref"]
    for r in rows:
        h = hists[r["variant"]]
        for m in ("cw", "nerm", "eci"):
            ta, tb = np.round(h[f"{m}_t"], 9), np.round(ref[f"{m}_t"], 9)
            common, ia, ib = np.intersect1d(ta, tb, return_indices=True)
            if common.size == 0:
                continue
            r.update({f"{m}_vs_ref_max_dpos_m": float(np.linalg.norm(h[f"{m}_S"][ia, :3] - ref[f"{m}_S"][ib, :3], axis=1).max()),
                      f"{m}_vs_ref_max_dvel_mps": float(np.linalg.norm(h[f"{m}_S"][ia, 3:] - ref[f"{m}_S"][ib, 3:], axis=1).max()),
                      f"{m}_vs_ref_max_dcmd_mps2": float(np.linalg.norm(h[f"{m}_app"][ia] - ref[f"{m}_app"][ib], axis=1).max()),
                      f"{m}_vs_ref_common_end_s": float(common[-1])})
        r.update(histdiff(h, h, "eci", "nerm", "eci_vs_nerm_hist"))
        r.update(histdiff(h, h, "cw", "nerm", "cw_vs_nerm_hist"))
    np.savez_compressed(os.path.join(outdir, "histories", f"{scn.sid}.npz"),
                        **{f"{v}__{k}": a for v in ("ref", "rtol1e-13") for k, a in hists[v].items()})
    orig = []
    try:
        from src import controller as sctl, orbit as sorb, perturbations as spert, validation as sval
        tgt0 = sorb.circorbitstate(scn.R, sorb.MUEARTH, math.radians(51.6))
        ctrl = sctl.PDCtrl.critdamped(scn.wn_ratio * scn.n, max_accel=scn.a_max)
        for k in (2001, 8001):
            tt = np.linspace(0.0, scn.horizon, k)
            t0 = time.perf_counter()
            cmp = sval.cmpfid(tgt0, np.array(scn.rel0), tt, spert.ForceMod.twobody(), scn.n, ctrl)
            cwm = sval.rdvmetrics(tt, cmp["cw_states"], cmp["cw_controls"], pos_tol=1.0).iloc[0]
            hfm = sval.rdvmetrics(tt, cmp["hf_states"], cmp["hf_controls"], pos_tol=1.0).iloc[0]
            orig.append({"sid": scn.sid, "samples": k, "wall_s": time.perf_counter() - t0,
                         "orig_max_traj_diff_m": float(cmp["summary"]["max_pos_error_m"].iloc[0]),
                         "orig_cw_conv_time_s": float(cwm["convergence_time_s"]),
                         "orig_hf_conv_time_s": float(hfm["convergence_time_s"]),
                         "orig_cw_final_pos_m": float(cwm["final_pos_error_m"]),
                         "orig_hf_final_pos_m": float(hfm["final_pos_error_m"]),
                         "orig_cw_dv_mps": float(cwm["delta_v_mps"]), "orig_hf_dv_mps": float(hfm["delta_v_mps"]),
                         "orig_hf_states_end": json.dumps([float(v) for v in cmp["hf_states"][-1]])})
    except Exception as exc:
        orig.append({"sid": scn.sid, "samples": -1, "error": repr(exc)})
    return rows, orig


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--cases-json", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.tag:
        OUT = os.path.join(OUT, a.tag)
    os.makedirs(os.path.join(OUT, "histories"), exist_ok=True)
    if a.cases_json:
        cs = [Scenario.from_dict(d) for d in json.load(open(a.cases_json))]
    else:
        cs = cases()
    pd.DataFrame([{**s.to_dict(), "sep_m": s.separation} for s in cs]).to_csv(os.path.join(OUT, "cases.csv"), index=False)
    rows, orig = [], []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r, o in ex.map(runcase, [(s.to_dict(), OUT) for s in cs]):
            rows.extend(r)
            orig.extend(o)
            print(f"  case {r[0]['sid']} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "variants.csv"), index=False)
    pd.DataFrame(orig).to_csv(os.path.join(OUT, "originalengine.csv"), index=False)
    print(f"wrote {len(df)} variant rows for {len(cs)} cases in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
