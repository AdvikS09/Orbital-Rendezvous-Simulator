"""Compare the independent discrete-map held-command estimators (cwvalid3.hybrid_map) with the
batch engine (cwvalid3.sampled.HeldBatch, public API only) on a deterministic case set.

Cases (altitude 400 km, one orbit, K = pd_matrix(wn), wn = ratio * n):
  a  +R 3000 m,              wn/n = 6, a_max = 0.05
  b  -R 12000 m,             wn/n = 8, a_max = 0.01
  c  +H 20000 m,             wn/n = 6, a_max = 0.05
  d  mixed [3000, -8000, 2000, 1.0, -2.0, 0.5], wn/n = 5, a_max = 0.02
  e  LQR-like K = pd_matrix(6n) + K[0,4] = 0.3 wn, K[1,3] = -0.3 wn, a_max = 0.03,
     rel0 = [-5000, 4000, 1000, 0, 0, 0]
Hold periods Ts = 1, 10, 30 s with batch step h = 1, 2, 10 s. Estimators A_H, B_H, C_H.

Per (case, Ts, estimator): features (max |e_pos| on the 0.5 s monitoring grid) from both, their
absolute / relative difference, max grid difference of e (position, velocity) and of the CW
nominal, the same at the update times (batch step nodes, no dense interpolation), update and
saturated-update counts and flag mismatches, nodes 8 vs 16 for the map, and wall times.
Attribution columns (*_vs_map_hermite_*): the batch grid compared with a cubic Hermite interpolant
built on the batch step nodes (hb.t) from the map's exact node values and exact one-sided
derivatives; when these are ~1e-12 the grid-level differences are entirely the batch's
dense-output interpolation, not its integration.

Usage: python experiments/v3/compare_hybrid_map.py
Output: v3_extension/results/hybrid_map_comparison.csv
"""

import csv
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
OUT = os.path.join(os.path.dirname(REPO), "v3_extension", "results", "hybrid_map_comparison.csv")

ALT = 400e3
TS_H = ((1.0, 1.0), (10.0, 2.0), (30.0, 10.0))
ESTS = ("A_H", "B_H", "C_H")


def case_set(n):
    from cwvalid3.sampled import pd_matrix
    out = []
    for name, rel0, ratio, amax in (
            ("a_+R3000", [3000, 0, 0, 0, 0, 0], 6.0, 0.05),
            ("b_-R12000", [-12000, 0, 0, 0, 0, 0], 8.0, 0.01),
            ("c_+H20000", [0, 0, 20000, 0, 0, 0], 6.0, 0.05),
            ("d_mixed", [3000, -8000, 2000, 1.0, -2.0, 0.5], 5.0, 0.02)):
        out.append(dict(case=name, rel0=np.array(rel0, float), ratio=ratio, a_max=amax,
                        K=np.asarray(pd_matrix(ratio * n), float)))
    wn = 6.0 * n
    K = np.array(pd_matrix(wn), float)
    K[0, 4] = 0.3 * wn
    K[1, 3] = -0.3 * wn
    out.append(dict(case="e_lqr_like", rel0=np.array([-5000, 4000, 1000, 0, 0, 0], float),
                    ratio=6.0, a_max=0.03, K=K))
    return out


def maxnorm(d):
    return float(np.max(np.linalg.norm(d, axis=1))) if d.size else float("nan")


def hermite(tq, tn, Y, D0, D1):
    """Piecewise cubic Hermite on steps [tn_j, tn_j+1] with values Y and one-sided slopes
    D0 (right slope at tn_j) and D1 (left slope at tn_j+1)."""
    j = np.clip(np.searchsorted(tn, tq, side="right") - 1, 0, tn.size - 2)
    hh = (tn[j + 1] - tn[j])[:, None]
    s = ((tq - tn[j]) / (tn[j + 1] - tn[j]))[:, None]
    h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
    h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * Y[j] + h10 * hh * D0[j] + h01 * Y[j + 1] + h11 * hh * D1[j]


def map_hermite(hm, c, Ts, est, tn, tm):
    """Hermite interpolant (on the batch step nodes tn) of the map's exact node values and exact
    one-sided derivatives, evaluated on tm: isolates the batch's dense-output interpolation."""
    R, n, T = hm.orbit(ALT)
    A, B = hm.cw_matrices(n)
    Ae = hm.double_integrator_matrix() if est == "C_H" else A
    mp = hm.held_sensitivity(c["rel0"], c["K"], c["a_max"], ALT, Ts, est, tq=tn)
    k = np.clip(np.searchsorted(mp["t_update"], tn[:-1], side="right") - 1, 0, None)
    u, m = mp["u_update"][k], mp["m_update"][k]

    def ds(s0):
        return s0 @ A.T + u @ B.T

    def de(e, s0):
        return e @ Ae.T + (m + hm.residual_exact(s0[:, :3], n, R)) @ B.T

    s0, e = mp["s0"], mp["e"]
    nom = hermite(tm, tn, s0, ds(s0[:-1]), ds(s0[1:]))
    est_h = hermite(tm, tn, e, de(e[:-1], s0[:-1]), de(e[1:], s0[1:]))
    return nom, est_h


def main():
    import v30_closed  # v3.1: preserved v3.0 outputs; runs only inside the rebuild harness
    v30_closed.refuse_outside_harness(__file__)
    from cwvalid3 import hybrid_map as hm
    from cwvalid3.sampled import HeldBatch
    from cwvalid3.held_metrics import CaseView

    R, n, T = hm.orbit(ALT)
    cases = case_set(n)
    rows = []
    for Ts, h in TS_H:
        t0 = time.perf_counter()
        hb = HeldBatch(np.stack([c["rel0"] for c in cases]), np.stack([c["K"] for c in cases]),
                       [c["a_max"] for c in cases], ALT, Ts, horizon_orbits=1.0, h=h,
                       models=("cw",), estimators=ESTS).run()
        wall_batch = time.perf_counter() - t0
        for i, c in enumerate(cases):
            t1 = time.perf_counter()
            cv = CaseView(hb, i)
            wall_cv = time.perf_counter() - t1
            tm = cv.tm
            nom_b = cv.grid("cw")
            tu = hm.update_times(Ts, T)
            tnode = np.append(tu, T)
            Ynode = hb.dense(i, tnode, cv.Fr, cv.Fl)          # step nodes: no interpolation
            sat_b = np.asarray(hb.sat_at_update[i], bool)
            for est in ESTS:
                t2 = time.perf_counter()
                mp = hm.held_sensitivity(c["rel0"], c["K"], c["a_max"], ALT, Ts, est, tq=tm,
                                         nodes=8)
                w8 = time.perf_counter() - t2
                t3 = time.perf_counter()
                mp16 = hm.held_sensitivity(c["rel0"], c["K"], c["a_max"], ALT, Ts, est, tq=tm,
                                           nodes=16)
                w16 = time.perf_counter() - t3
                eb = cv.grid(est)
                pb = np.linalg.norm(eb[:, :3], axis=1)
                jb = int(np.argmax(pb))
                fb = float(pb[jb])
                fm = mp["feature_m"]
                eb_node = cv.rel_from_rows(est, Ynode)
                nb_node = cv.rel_from_rows("cw", Ynode)
                em_node = mp["e_update"]
                sm_node = mp["s_update"]
                sat_m = mp["sat_at_update"]
                nmin = min(sat_m.size, sat_b.size)
                mism = np.nonzero(sat_m[:nmin] != sat_b[:nmin])[0]
                nom_h, e_h = map_hermite(hm, c, Ts, est, np.asarray(hb.t, float), tm)
                rows.append(dict(
                    Ts_s=Ts, h_batch_s=h, case=c["case"], wn_ratio=c["ratio"], a_max=c["a_max"],
                    estimator=est, batch_status=str(hb.status[est][i]),
                    feature_map_m=fm, feature_batch_m=fb,
                    feature_abs_diff_m=abs(fm - fb), feature_rel_diff=abs(fm - fb) / fm,
                    t_peak_map_s=mp["t_peak_s"], t_peak_batch_s=float(tm[jb]),
                    e_grid_maxdiff_pos_m=maxnorm(mp["e"][:, :3] - eb[:, :3]),
                    e_grid_maxdiff_pos_rel=maxnorm(mp["e"][:, :3] - eb[:, :3]) / fm,
                    e_grid_maxdiff_vel_mps=maxnorm(mp["e"][:, 3:] - eb[:, 3:]),
                    e_node_maxdiff_pos_m=maxnorm(em_node[:, :3] - eb_node[:, :3]),
                    e_node_maxdiff_pos_rel=maxnorm(em_node[:, :3] - eb_node[:, :3]) / fm,
                    e_grid_maxdiff_pos_vs_map_hermite_rel=maxnorm(e_h[:, :3] - eb[:, :3]) / fm,
                    nom_grid_maxdiff_pos_m=maxnorm(mp["s0"][:, :3] - nom_b[:, :3]),
                    nom_grid_maxdiff_pos_vs_map_hermite_m=maxnorm(nom_h[:, :3] - nom_b[:, :3]),
                    nom_grid_maxdiff_vel_mps=maxnorm(mp["s0"][:, 3:] - nom_b[:, 3:]),
                    nom_node_maxdiff_pos_m=maxnorm(sm_node[:, :3] - nb_node[:, :3]),
                    nom_node_maxdiff_pos_rel=maxnorm(sm_node[:, :3] - nb_node[:, :3])
                    / float(np.linalg.norm(c["rel0"][:3])),
                    n_updates_map=int(sat_m.size), n_updates_batch=int(sat_b.size),
                    n_sat_map=int(sat_m.sum()), n_sat_batch=int(sat_b.sum()),
                    n_sat_flag_mismatch=int(mism.size) + abs(sat_m.size - sat_b.size),
                    first_sat_mismatch_idx=int(mism[0]) if mism.size else -1,
                    min_switch_margin_map=float(np.min(mp["switch_margin"])),
                    feature_map_nodes16_m=mp16["feature_m"],
                    feature_rel_diff_nodes8_vs_16=abs(mp16["feature_m"] - fm) / fm,
                    e_grid_maxdiff_nodes8_vs_16_rel=maxnorm(mp16["e"][:, :3] - mp["e"][:, :3]) / fm,
                    wall_map_nodes8_s=w8, wall_map_nodes16_s=w16,
                    wall_batch_run_all5cases_s=wall_batch, wall_batch_caseview_s=wall_cv,
                    T_map_s=T, T_batch_s=float(hb.T)))
                r = rows[-1]
                print(f"Ts={Ts:4.0f} {c['case']:11s} {est}: F_map={fm:.10e} F_batch={fb:.10e} "
                      f"rel={r['feature_rel_diff']:.2e} grid={r['e_grid_maxdiff_pos_rel']:.2e} "
                      f"node={r['e_node_maxdiff_pos_rel']:.2e} nom={r['nom_grid_maxdiff_pos_m']:.2e} m "
                      f"sat {r['n_sat_map']}/{r['n_sat_batch']} mism={r['n_sat_flag_mismatch']}",
                      flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    print("\nmax relative feature difference by Ts and estimator:")
    for Ts, _ in TS_H:
        for est in ESTS:
            sel = [r for r in rows if r["Ts_s"] == Ts and r["estimator"] == est]
            print(f"  Ts={Ts:4.0f} {est}: feature {max(r['feature_rel_diff'] for r in sel):.2e}  "
                  f"grid {max(r['e_grid_maxdiff_pos_rel'] for r in sel):.2e}  "
                  f"nodes {max(r['e_node_maxdiff_pos_rel'] for r in sel):.2e}  "
                  f"grid-vs-map-Hermite {max(r['e_grid_maxdiff_pos_vs_map_hermite_rel'] for r in sel):.2e}  "
                  f"nominal grid {max(r['nom_grid_maxdiff_pos_m'] for r in sel):.1e} m "
                  f"(vs map Hermite {max(r['nom_grid_maxdiff_pos_vs_map_hermite_m'] for r in sel):.1e} m)  "
                  f"8v16 {max(r['feature_rel_diff_nodes8_vs_16'] for r in sel):.1e}  "
                  f"sat mismatches {sum(r['n_sat_flag_mismatch'] for r in sel)}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
