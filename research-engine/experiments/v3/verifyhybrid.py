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
ARCH = os.path.dirname(REPO)
OUT = os.path.join(ARCH, "v3", "results")

from cwvalid.constants import MUEARTH, REARTH

ALT = 400_000.0
N0 = math.sqrt(MUEARTH / (REARTH + ALT) ** 3)
LAMS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
CASES = [
    ("+R_3km_switching", [3000.0, 0, 0, 0, 0, 0], 6.0, 0.05),
    ("+H_20km_switching", [0, 0, 20000.0, 0, 0, 0], 6.0, 0.05),
    ("mixed_8km_saturated", [3000.0, -8000.0, 2000.0, 1.0, -2.0, 0.5], 5.0, 0.02),
    ("-R_12km_saturated", [-12000.0, 0, 0, 0, 0, 0], 8.0, 0.01),
    ("+V_500m_unsat", [0, 500.0, 0, 0, 0, 0], 6.0, 0.05),
    ("mixed_1.1km_vel", [700.0, -600.0, 550.0, 0.6, 0.2, -0.4], 7.0, 0.006),
]


def _K(wn_ratio):
    from cwvalid3.sampled import pdmatrix
    return pdmatrix(wn_ratio * N0)


def fdladder(args):
    label, rel0, wn, am, Ts = args
    from cwvalid3.heldmet import CaseView
    from cwvalid3.sampled import HeldBatch
    lams = tuple(x for l in LAMS for x in (l, -l))
    h = min(Ts, 2.0)
    hb = HeldBatch([rel0], _K(wn)[None], [am], ALT, Ts, h=h, models=("cw",), ests=("A_H", "B_H"),
                   lambdas=lams).run()
    cv = CaseView(hb, 0)
    eA, eB = cv.grid("A_H")[:, :3], cv.grid("B_H")[:, :3]
    scale = np.max(np.linalg.norm(eA, axis=1))
    rows = []
    for l in LAMS:
        sp = cv.grid(f"hom:{l:.12g}")[:, :3]
        sm = cv.grid(f"hom:{-l:.12g}")[:, :3]
        s0 = cv.grid("cw")[:, :3]
        fd = (sp - sm) / (2 * l)
        fwd = (sp - s0) / l
        bwd = (s0 - sm) / l
        rows.append({"test": "fd_ladder", "case": label, "Ts_s": Ts, "h_s": h, "lambda": l,
                     "max_e_A_H_m": scale, "err_central_vs_A_H_rel": float(np.max(np.linalg.norm(fd - eA, axis=1)) / scale),
                     "err_forward_vs_A_H_rel": float(np.max(np.linalg.norm(fwd - eA, axis=1)) / scale),
                     "err_backward_vs_A_H_rel": float(np.max(np.linalg.norm(bwd - eA, axis=1)) / scale),
                     "err_central_vs_B_H_rel": float(np.max(np.linalg.norm(fd - eB, axis=1)) / scale),
                     "n_sat_updates": int(hb.sat_at_update[0].sum()), "n_updates": int(hb.sat_at_update.shape[1]),
                     "n_status_changes": int(np.sum(hb.sat_at_update[0][1:] != hb.sat_at_update[0][:-1])),
                     "n_near_switch_updates": int(hb.nearswitch[0].sum()),
                     "hom_status": ";".join(sorted(set(hb.status[f'hom:{x:.12g}'][0] for x in (l, -l))))})
    return rows


def nosat(args):
    label, rel0, wn, _, Ts = args
    from cwvalid3.heldmet import CaseView
    from cwvalid3.sampled import HeldBatch
    hb = HeldBatch([rel0], _K(wn)[None], [None], ALT, Ts, h=min(Ts, 2.0), models=("cw",), ests=("A_H", "B_H")).run()
    cv = CaseView(hb, 0)
    d = np.abs(cv.grid("A_H") - cv.grid("B_H"))
    return [{"test": "no_saturation", "case": label, "Ts_s": Ts, "max_abs_diff_A_H_B_H": float(d.max()),
             "bitwise_identical": bool(np.array_equal(cv.grid("A_H"), cv.grid("B_H")))}]


def _scan(rel0, K, Ts, amaxes, horizon=1.0):
    from cwvalid3.sampled import HeldBatch
    N = len(amaxes)
    hb = HeldBatch([rel0] * N, np.repeat(K[None], N, 0), list(amaxes), ALT, Ts, h=min(Ts, 2.0), models=("cw",),
                   ests=(), horizon_orbits=horizon).run()
    sat = hb.sat_at_update
    return sat.sum(1), np.sum(sat[:, 1:] != sat[:, :-1], axis=1)


def bisect(rel0, K, Ts, alo, ahi, pred, rounds=6, N=24):
    for _ in range(rounds):
        grid = np.geomspace(alo, ahi, N)
        nsat, nchg = _scan(rel0, K, Ts, grid)
        p = np.array([pred(s, c) for s, c in zip(nsat, nchg)])
        k = int(np.argmax(p != p[0]))
        if k == 0:
            break
        alo, ahi = grid[k - 1], grid[k]
    return alo, ahi


def nearswitch(args):
    label, rel0, wn, Ts, kind = args
    from cwvalid3.heldmet import CaseView
    from cwvalid3.sampled import HeldBatch, command
    K = _K(wn)
    scan = np.geomspace(0.002, 0.2, 48)
    nsat, nchg = _scan(rel0, K, Ts, scan)
    if kind == "transversal":
        i = int(np.argmin(np.abs(scan - 0.05)))
        m = nsat[i]
        alo, ahi = bisect(rel0, K, Ts, scan[i], scan[i] * 1.2, lambda s, c: s >= m)
    else:
        idx = [k for k in range(1, scan.size) if (nchg[k - 1] >= 3) != (nchg[k] >= 3)]
        if not idx:
            return [{"test": "near_switch_grazing", "case": label, "Ts_s": Ts,
                     "note": "no re-saturation boundary found on the a_max scan; grazing case not constructed"}]
        k = idx[0]
        alo, ahi = bisect(rel0, K, Ts, scan[k - 1], scan[k], lambda s, c: c >= 3)
    rows = []
    lams = tuple(x for l in LAMS for x in (l, -l))
    for side, am in (("lower_a_max", alo), ("upper_a_max", ahi)):
        hb = HeldBatch([rel0], K[None], [am], ALT, Ts, h=min(Ts, 2.0), models=("cw",), ests=("A_H",),
                       lambdas=lams, nearswitchtol=1e-6).run()
        mags = np.array([command(K[None], np.array([am]), hb.Y[j, :, :6])[2][0] for j in hb.updatestep])
        reldist = np.abs(mags / am - 1.0)
        j = int(np.argmin(reldist))
        cv = CaseView(hb, 0)
        eA = cv.grid("A_H")[:, :3]
        scale = np.max(np.linalg.norm(eA, axis=1))
        s0 = cv.grid("cw")[:, :3]
        for l in LAMS:
            sp = cv.grid(f"hom:{l:.12g}")[:, :3]
            sm = cv.grid(f"hom:{-l:.12g}")[:, :3]
            rows.append({"test": f"near_switch_{kind}", "case": label, "Ts_s": Ts, "side": side, "lambda": l,
                         "a_max": am, "bracket_rel_width": ahi / alo - 1.0, "closest_update_index": j,
                         "closest_update_rel_distance": float(reldist[j]), "max_e_A_H_m": scale,
                         "err_forward_vs_A_H_rel": float(np.max(np.linalg.norm((sp - s0) / l - eA, axis=1)) / scale),
                         "err_backward_vs_A_H_rel": float(np.max(np.linalg.norm((s0 - sm) / l - eA, axis=1)) / scale),
                         "err_central_vs_A_H_rel": float(np.max(np.linalg.norm((sp - sm) / (2 * l) - eA, axis=1)) / scale),
                         "n_sat_updates": int(hb.sat_at_update[0].sum()),
                         "n_status_changes": int(np.sum(hb.sat_at_update[0][1:] != hb.sat_at_update[0][:-1])),
                         "n_near_switch_updates_tol1e-6": int(hb.nearswitch[0].sum())})
    return rows


def tstozero(args):
    label, rel0, wn, am = args
    from cwvalid.estimate import firstorderdisc
    from cwvalid.experiment import RunCfg, evaluate, makelaw, runbranch
    from cwvalid.scenario import Scenario
    from cwvalid3.continuous import estmat
    from cwvalid3.controllers import pdlaw
    from cwvalid3.heldmet import casesumm, estimfeat, CaseView
    from cwvalid3.sampled import HeldBatch
    s = Scenario("v3hyb", tuple(rel0), wn, am, 1.0, ALT, meta={"variant": f"tsconv:{label}"})
    cfg = RunCfg(models=("cw", "eci", "nerm"), reference="eci", crosscheck="nerm")
    sims, _ = runbranch(s, cfg)
    row, _ = evaluate(s, cfg, sims)
    law = makelaw(s)
    AC = firstorderdisc(sims["cw"], law, s.n, s.R, s.horizon, dt_out=0.5, rtol=1e-11, atol=1e-11)["d1_peak_m"]
    BC = estmat(sims["cw"], pdlaw(s.wn_ratio * s.n, am), s.n, s.R, s.horizon, "B", dt_out=0.5,
                rtol=1e-11, atol=1e-11)["peak_m"]
    rows = []
    for Ts in (10.0, 5.0, 2.0, 1.0, 0.5, 0.25):
        t0 = time.perf_counter()
        hb = HeldBatch([rel0], _K(wn)[None], [am], ALT, Ts, h=Ts / math.ceil(Ts / 2.0), models=("cw", "nerm", "eci"),
                       ests=("A_H", "B_H")).run()
        hn = HeldBatch([rel0], _K(wn)[None], [am], ALT, Ts, h=min(Ts, 0.5), models=("cw",), ests=("A_naive",)).run()
        sm, _ = casesumm(hb, 0, pairs=(("cw", "eci"),))
        fn = estimfeat(CaseView(hn, 0), "A_naive")
        rows.append({"test": "ts_to_zero", "case": label, "Ts_s": Ts, "omega_n_Ts": wn * s.n * Ts,
                     "A_H_m": sm["A_H_peak_m"], "A_naive_m": fn["peak_m"], "B_H_m": sm["B_H_peak_m"],
                     "A_continuous_m": AC, "B_continuous_m": BC,
                     "rel_A_H_vs_cont": abs(sm["A_H_peak_m"] - AC) / AC,
                     "rel_A_naive_vs_cont": abs(fn["peak_m"] - AC) / AC,
                     "rel_A_H_vs_A_naive": abs(sm["A_H_peak_m"] - fn["peak_m"]) / AC,
                     "rel_B_H_vs_cont": abs(sm["B_H_peak_m"] - BC) / BC,
                     "disc_held_m": sm["cw_vs_eci_peak_m"], "disc_continuous_m": row["cw_vs_eci_max_pos_disc_m"],
                     "rel_A_H_vs_disc_held": abs(sm["A_H_peak_m"] - sm["cw_vs_eci_peak_m"]) / sm["cw_vs_eci_peak_m"],
                     "rel_A_naive_vs_disc_held": abs(fn["peak_m"] - sm["cw_vs_eci_peak_m"]) / sm["cw_vs_eci_peak_m"],
                     "n_sat_updates": sm.get("cw_n_sat_updates"), "n_status_changes": sm.get("cw_n_status_changes"),
                     "cont_n_sat_switch": len(sims["cw"].satswitchtimes), "wall_s": time.perf_counter() - t0})
    return rows


def failpath(_):
    from cwvalid3.heldmet import CaseView, estimfeat
    from cwvalid3.sampled import HeldBatch, pdmatrix
    hb = HeldBatch([[1000.0, 0, 0, 0, 0, 0]], pdmatrix(0.5)[None], [None], ALT, 10.0, h=2.0, models=("cw", "nerm"),
                   ests=("A_H", "B_H", "C_H")).run()
    cv = CaseView(hb, 0)
    f = {e: estimfeat(cv, e) for e in ("A_H", "B_H", "C_H")}
    return [{"test": "failure_path", "case": "omega_n=0.5 rad/s, Ts=10 s (discrete loop unstable)", "Ts_s": 10.0,
             "cw_status": hb.status["cw"][0], "cw_t_end_s": float(hb.t_end["cw"][0]),
             "nerm_status": hb.status["nerm"][0],
             "estimator_statuses": ";".join(f"{e}:{f[e]['status']}" for e in f),
             "features_are_nan": all(not np.isfinite(f[e]["peak_m"]) for e in f)}]


def main():
    import v30closed
    v30closed.refuseoutsharness(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    jobs = [("fd", (lab, r, w, am, Ts)) for lab, r, w, am in CASES for Ts in (1.0, 10.0)]
    jobs += [("nosat", (lab, r, w, am, Ts)) for lab, r, w, am in CASES[:3] for Ts in (1.0, 10.0)]
    jobs += [("near", ("+R_3km_unsat_base", [3000.0, 0, 0, 0, 0, 0], 6.0, Ts, k)) for Ts in (1.0, 10.0)
             for k in ("transversal", "grazing")]
    jobs += [("ts0", c) for c in CASES]
    jobs += [("fail", None)]
    fn = {"fd": fdladder, "nosat": nosat, "near": nearswitch, "ts0": tstozero, "fail": failpath}
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fn[k], arg) for k, arg in jobs]
        for f in futs:
            rows.extend(f.result())
    p = os.path.join(OUT, "hybridmap.csv")
    if os.path.exists(p):
        m = pd.read_csv(p)
        rows.append({"test": "independent_map", "case": "summary of hybridmap.csv", "n_rows": len(m),
                     "note": "cwvalid3.hybmap (expm + Gauss-Legendre) vs cwvalid3.sampled (fixed-step DOP853); see file"})
    df = pd.DataFrame(rows)
    df["wall_total_s"] = time.perf_counter() - t0
    df.to_csv(os.path.join(OUT, "hybridvalidation.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for t in df.test.unique():
            print("==", t)
            print(df[df.test == t].dropna(axis=1, how="all").drop(columns=["wall_total_s"]).to_string(max_rows=80))


if __name__ == "__main__":
    main()
