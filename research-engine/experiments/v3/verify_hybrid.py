"""P06 verification of the held-command first variation (results/hybrid_validation.csv).

Tests (each row of the CSV is one test/case/setting):
  fd_ladder        central differences (s_lam - s_-lam)/(2 lam) of propagated held homotopy branches
                   (each branch updates its own held command from its own state) against e from A_H and
                   B_H, lam = 1e-1 ... 1e-6, error = max over the 0.5 s grid of |fd_pos - e_pos| / max|e_pos|
  no_saturation    a_max = inf: A_H and B_H must be bitwise identical
  near_switch      a_max set 1e-7 above the sampled command magnitude at one update (transversal) and
                   just below a local maximum of the sampled magnitude (grazing): one-sided differences
                   (s_lam - s_0)/lam for lam > 0 and lam < 0 against A_H, near-switch counter
  ts_to_zero       A_H, A_naive and B_H features at T_s = 10, 5, 2, 1, 0.5, 0.25 s against the continuous
                   production A/B features on the same 0.5 s grid (rtol = atol = 1e-11), plus the held and
                   continuous true discrepancies
  failure_path     a discrete-time-unstable loop (omega_n T_s >> 1) must end with CW numerical_failure
                   and an unavailable (nominal_failed) estimator, never a number
  independent_map  summary of results/hybrid_map_comparison.csv (cwvalid3.hybrid_map vs the batch engine)
Usage: python code/experiments/v3/verify_hybrid.py [--workers 6]
"""

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
OUT = os.path.join(ARCH, "v3_extension", "results")

from cwvalid.constants import MU_EARTH, R_EARTH  # noqa: E402

ALT = 400_000.0
N0 = math.sqrt(MU_EARTH / (R_EARTH + ALT) ** 3)
LAMS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
CASES = [  # label, rel0, wn_ratio, a_max
    ("+R_3km_switching", [3000.0, 0, 0, 0, 0, 0], 6.0, 0.05),
    ("+H_20km_switching", [0, 0, 20000.0, 0, 0, 0], 6.0, 0.05),
    ("mixed_8km_saturated", [3000.0, -8000.0, 2000.0, 1.0, -2.0, 0.5], 5.0, 0.02),
    ("-R_12km_saturated", [-12000.0, 0, 0, 0, 0, 0], 8.0, 0.01),
    ("+V_500m_unsat", [0, 500.0, 0, 0, 0, 0], 6.0, 0.05),
    ("mixed_1.1km_vel", [700.0, -600.0, 550.0, 0.6, 0.2, -0.4], 7.0, 0.006),
]


def _K(wn_ratio):
    from cwvalid3.sampled import pd_matrix
    return pd_matrix(wn_ratio * N0)


def fd_ladder(args):
    label, rel0, wn, am, Ts = args
    from cwvalid3.held_metrics import CaseView
    from cwvalid3.sampled import HeldBatch
    lams = tuple(x for l in LAMS for x in (l, -l))
    h = min(Ts, 2.0)
    hb = HeldBatch([rel0], _K(wn)[None], [am], ALT, Ts, h=h, models=("cw",), estimators=("A_H", "B_H"),
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
                     "n_near_switch_updates": int(hb.near_switch[0].sum()),
                     "hom_status": ";".join(sorted(set(hb.status[f'hom:{x:.12g}'][0] for x in (l, -l))))})
    return rows


def no_saturation(args):
    label, rel0, wn, _, Ts = args
    from cwvalid3.held_metrics import CaseView
    from cwvalid3.sampled import HeldBatch
    hb = HeldBatch([rel0], _K(wn)[None], [None], ALT, Ts, h=min(Ts, 2.0), models=("cw",), estimators=("A_H", "B_H")).run()
    cv = CaseView(hb, 0)
    d = np.abs(cv.grid("A_H") - cv.grid("B_H"))
    return [{"test": "no_saturation", "case": label, "Ts_s": Ts, "max_abs_diff_A_H_B_H": float(d.max()),
             "bitwise_identical": bool(np.array_equal(cv.grid("A_H"), cv.grid("B_H")))}]


def _scan(rel0, K, Ts, amaxes, horizon=1.0):
    from cwvalid3.sampled import HeldBatch
    N = len(amaxes)
    hb = HeldBatch([rel0] * N, np.repeat(K[None], N, 0), list(amaxes), ALT, Ts, h=min(Ts, 2.0), models=("cw",),
                   estimators=(), horizon_orbits=horizon).run()
    sat = hb.sat_at_update
    return sat.sum(1), np.sum(sat[:, 1:] != sat[:, :-1], axis=1)


def _bisect(rel0, K, Ts, a_lo, a_hi, pred, rounds=6, N=24):
    """Shrink [a_lo, a_hi] (pred differs at the ends) to the a_max where pred flips."""
    for _ in range(rounds):
        grid = np.geomspace(a_lo, a_hi, N)
        nsat, nchg = _scan(rel0, K, Ts, grid)
        p = np.array([pred(s, c) for s, c in zip(nsat, nchg)])
        k = int(np.argmax(p != p[0]))
        if k == 0:
            break
        a_lo, a_hi = grid[k - 1], grid[k]
    return a_lo, a_hi


def near_switch(args):
    """Transversal: a_max at which the number of saturated updates changes (one update lies on the
    switching surface). Grazing: a_max at which a re-saturation appears/disappears (the sampled command
    magnitude touches the cap from below at a local maximum). Both located by bisection on a_max; the FD
    ladder is then evaluated just on each side of the boundary."""
    label, rel0, wn, Ts, kind = args
    from cwvalid3.held_metrics import CaseView
    from cwvalid3.sampled import HeldBatch, command
    K = _K(wn)
    scan = np.geomspace(0.002, 0.2, 48)
    nsat, nchg = _scan(rel0, K, Ts, scan)
    if kind == "transversal":
        i = int(np.argmin(np.abs(scan - 0.05)))
        m = nsat[i]
        a_lo, a_hi = _bisect(rel0, K, Ts, scan[i], scan[i] * 1.2, lambda s, c: s >= m)
    else:
        idx = [k for k in range(1, scan.size) if (nchg[k - 1] >= 3) != (nchg[k] >= 3)]
        if not idx:
            return [{"test": "near_switch_grazing", "case": label, "Ts_s": Ts,
                     "note": "no re-saturation boundary found on the a_max scan; grazing case not constructed"}]
        k = idx[0]
        a_lo, a_hi = _bisect(rel0, K, Ts, scan[k - 1], scan[k], lambda s, c: c >= 3)
    rows = []
    lams = tuple(x for l in LAMS for x in (l, -l))
    for side, am in (("lower_a_max", a_lo), ("upper_a_max", a_hi)):
        hb = HeldBatch([rel0], K[None], [am], ALT, Ts, h=min(Ts, 2.0), models=("cw",), estimators=("A_H",),
                       lambdas=lams, near_switch_tol=1e-6).run()
        mags = np.array([command(K[None], np.array([am]), hb.Y[j, :, :6])[2][0] for j in hb.update_step])
        rel_dist = np.abs(mags / am - 1.0)
        j = int(np.argmin(rel_dist))
        cv = CaseView(hb, 0)
        eA = cv.grid("A_H")[:, :3]
        scale = np.max(np.linalg.norm(eA, axis=1))
        s0 = cv.grid("cw")[:, :3]
        for l in LAMS:
            sp = cv.grid(f"hom:{l:.12g}")[:, :3]
            sm = cv.grid(f"hom:{-l:.12g}")[:, :3]
            rows.append({"test": f"near_switch_{kind}", "case": label, "Ts_s": Ts, "side": side, "lambda": l,
                         "a_max": am, "bracket_rel_width": a_hi / a_lo - 1.0, "closest_update_index": j,
                         "closest_update_rel_distance": float(rel_dist[j]), "max_e_A_H_m": scale,
                         "err_forward_vs_A_H_rel": float(np.max(np.linalg.norm((sp - s0) / l - eA, axis=1)) / scale),
                         "err_backward_vs_A_H_rel": float(np.max(np.linalg.norm((s0 - sm) / l - eA, axis=1)) / scale),
                         "err_central_vs_A_H_rel": float(np.max(np.linalg.norm((sp - sm) / (2 * l) - eA, axis=1)) / scale),
                         "n_sat_updates": int(hb.sat_at_update[0].sum()),
                         "n_status_changes": int(np.sum(hb.sat_at_update[0][1:] != hb.sat_at_update[0][:-1])),
                         "n_near_switch_updates_tol1e-6": int(hb.near_switch[0].sum())})
    return rows


def ts_to_zero(args):
    label, rel0, wn, am = args
    from cwvalid.estimate import first_order_discrepancy
    from cwvalid.experiment import RunConfig, evaluate, make_law, run_branches
    from cwvalid.scenario import Scenario
    from cwvalid3.continuous import estimator_matrix
    from cwvalid3.controllers import pd_law
    from cwvalid3.held_metrics import case_summary, estimator_feature, CaseView
    from cwvalid3.sampled import HeldBatch
    s = Scenario("v3hyb", tuple(rel0), wn, am, 1.0, ALT, meta={"variant": f"tsconv:{label}"})
    cfg = RunConfig(models=("cw", "eci", "nerm"), reference="eci", crosscheck="nerm")
    sims, _ = run_branches(s, cfg)
    row, _ = evaluate(s, cfg, sims)
    law = make_law(s)
    A_c = first_order_discrepancy(sims["cw"], law, s.n, s.R, s.horizon, dt_out=0.5, rtol=1e-11, atol=1e-11)["d1_peak_m"]
    B_c = estimator_matrix(sims["cw"], pd_law(s.wn_ratio * s.n, am), s.n, s.R, s.horizon, "B", dt_out=0.5,
                           rtol=1e-11, atol=1e-11)["peak_m"]
    rows = []
    for Ts in (10.0, 5.0, 2.0, 1.0, 0.5, 0.25):
        t0 = time.perf_counter()
        hb = HeldBatch([rel0], _K(wn)[None], [am], ALT, Ts, h=Ts / math.ceil(Ts / 2.0), models=("cw", "nerm", "eci"),
                       estimators=("A_H", "B_H")).run()
        hn = HeldBatch([rel0], _K(wn)[None], [am], ALT, Ts, h=min(Ts, 0.5), models=("cw",), estimators=("A_naive",)).run()
        sm, _ = case_summary(hb, 0, pairs=(("cw", "eci"),))
        fn = estimator_feature(CaseView(hn, 0), "A_naive")
        rows.append({"test": "ts_to_zero", "case": label, "Ts_s": Ts, "omega_n_Ts": wn * s.n * Ts,
                     "A_H_m": sm["A_H_peak_m"], "A_naive_m": fn["peak_m"], "B_H_m": sm["B_H_peak_m"],
                     "A_continuous_m": A_c, "B_continuous_m": B_c,
                     "rel_A_H_vs_cont": abs(sm["A_H_peak_m"] - A_c) / A_c,
                     "rel_A_naive_vs_cont": abs(fn["peak_m"] - A_c) / A_c,
                     "rel_A_H_vs_A_naive": abs(sm["A_H_peak_m"] - fn["peak_m"]) / A_c,
                     "rel_B_H_vs_cont": abs(sm["B_H_peak_m"] - B_c) / B_c,
                     "disc_held_m": sm["cw_vs_eci_peak_m"], "disc_continuous_m": row["cw_vs_eci_max_pos_disc_m"],
                     "rel_A_H_vs_disc_held": abs(sm["A_H_peak_m"] - sm["cw_vs_eci_peak_m"]) / sm["cw_vs_eci_peak_m"],
                     "rel_A_naive_vs_disc_held": abs(fn["peak_m"] - sm["cw_vs_eci_peak_m"]) / sm["cw_vs_eci_peak_m"],
                     "n_sat_updates": sm.get("cw_n_sat_updates"), "n_status_changes": sm.get("cw_n_status_changes"),
                     "cont_n_sat_switch": len(sims["cw"].sat_switch_times), "wall_s": time.perf_counter() - t0})
    return rows


def failure_path(_):
    from cwvalid3.held_metrics import CaseView, estimator_feature
    from cwvalid3.sampled import HeldBatch, pd_matrix
    hb = HeldBatch([[1000.0, 0, 0, 0, 0, 0]], pd_matrix(0.5)[None], [None], ALT, 10.0, h=2.0, models=("cw", "nerm"),
                   estimators=("A_H", "B_H", "C_H")).run()
    cv = CaseView(hb, 0)
    f = {e: estimator_feature(cv, e) for e in ("A_H", "B_H", "C_H")}
    return [{"test": "failure_path", "case": "omega_n=0.5 rad/s, Ts=10 s (discrete loop unstable)", "Ts_s": 10.0,
             "cw_status": hb.status["cw"][0], "cw_t_end_s": float(hb.t_end["cw"][0]),
             "nerm_status": hb.status["nerm"][0],
             "estimator_statuses": ";".join(f"{e}:{f[e]['status']}" for e in f),
             "features_are_nan": all(not np.isfinite(f[e]["peak_m"]) for e in f)}]


def main():
    import v30_closed  # v3.1: preserved v3.0 outputs; runs only inside the rebuild harness
    v30_closed.refuse_outside_harness(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    jobs = [("fd", (lab, r, w, am, Ts)) for lab, r, w, am in CASES for Ts in (1.0, 10.0)]
    jobs += [("nosat", (lab, r, w, am, Ts)) for lab, r, w, am in CASES[:3] for Ts in (1.0, 10.0)]
    jobs += [("near", ("+R_3km_unsat_base", [3000.0, 0, 0, 0, 0, 0], 6.0, Ts, k)) for Ts in (1.0, 10.0)
             for k in ("transversal", "grazing")]
    jobs += [("ts0", c) for c in CASES]
    jobs += [("fail", None)]
    fn = {"fd": fd_ladder, "nosat": no_saturation, "near": near_switch, "ts0": ts_to_zero, "fail": failure_path}
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fn[k], arg) for k, arg in jobs]
        for f in futs:
            rows.extend(f.result())
    p = os.path.join(OUT, "hybrid_map_comparison.csv")
    if os.path.exists(p):
        m = pd.read_csv(p)
        rows.append({"test": "independent_map", "case": "summary of hybrid_map_comparison.csv", "n_rows": len(m),
                     "note": "cwvalid3.hybrid_map (expm + Gauss-Legendre) vs cwvalid3.sampled (fixed-step DOP853); see file"})
    df = pd.DataFrame(rows)
    df["wall_total_s"] = time.perf_counter() - t0
    df.to_csv(os.path.join(OUT, "hybrid_validation.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for t in df.test.unique():
            print("==", t)
            print(df[df.test == t].dropna(axis=1, how="all").drop(columns=["wall_total_s"]).to_string(max_rows=80))


if __name__ == "__main__":
    main()
