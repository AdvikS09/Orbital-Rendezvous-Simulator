import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
OUT = os.path.join(os.path.dirname(REPO), "v3", "results", "hybridmap.csv")

ALT = 400e3
TSH = ((1.0, 1.0), (10.0, 2.0), (30.0, 10.0))
ESTS = ("A_H", "B_H", "C_H")


def caseset(n):
    from cwvalid3.sampled import pdmatrix
    out = []
    for name, rel0, ratio, amax in (
            ("a_+R3000", [3000, 0, 0, 0, 0, 0], 6.0, 0.05),
            ("b_-R12000", [-12000, 0, 0, 0, 0, 0], 8.0, 0.01),
            ("c_+H20000", [0, 0, 20000, 0, 0, 0], 6.0, 0.05),
            ("d_mixed", [3000, -8000, 2000, 1.0, -2.0, 0.5], 5.0, 0.02)):
        out.append(dict(case=name, rel0=np.array(rel0, float), ratio=ratio, a_max=amax,
                        K=np.asarray(pdmatrix(ratio * n), float)))
    wn = 6.0 * n
    K = np.array(pdmatrix(wn), float)
    K[0, 4] = 0.3 * wn
    K[1, 3] = -0.3 * wn
    out.append(dict(case="e_lqr_like", rel0=np.array([-5000, 4000, 1000, 0, 0, 0], float),
                    ratio=6.0, a_max=0.03, K=K))
    return out


def maxnorm(d):
    return float(np.max(np.linalg.norm(d, axis=1))) if d.size else float("nan")


def hermite(tq, tn, Y, D0, D1):
    j = np.clip(np.searchsorted(tn, tq, side="right") - 1, 0, tn.size - 2)
    hh = (tn[j + 1] - tn[j])[:, None]
    s = ((tq - tn[j]) / (tn[j + 1] - tn[j]))[:, None]
    h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
    h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * Y[j] + h10 * hh * D0[j] + h01 * Y[j + 1] + h11 * hh * D1[j]


def mapherm(hm, c, Ts, est, tn, tm):
    R, n, T = hm.orbit(ALT)
    A, B = hm.cwmats(n)
    Ae = hm.dblintegmat() if est == "C_H" else A
    mp = hm.heldsens(c["rel0"], c["K"], c["a_max"], ALT, Ts, est, tq=tn)
    k = np.clip(np.searchsorted(mp["t_update"], tn[:-1], side="right") - 1, 0, None)
    u, m = mp["u_update"][k], mp["m_update"][k]

    def ds(s0):
        return s0 @ A.T + u @ B.T

    def de(e, s0):
        return e @ Ae.T + (m + hm.residexact(s0[:, :3], n, R)) @ B.T

    s0, e = mp["s0"], mp["e"]
    nom = hermite(tm, tn, s0, ds(s0[:-1]), ds(s0[1:]))
    esth = hermite(tm, tn, e, de(e[:-1], s0[:-1]), de(e[1:], s0[1:]))
    return nom, esth


def main():
    import v30closed
    v30closed.refuseoutsharness(__file__)
    from cwvalid3 import hybmap as hm
    from cwvalid3.sampled import HeldBatch
    from cwvalid3.heldmet import CaseView

    R, n, T = hm.orbit(ALT)
    cas = caseset(n)
    rows = []
    for Ts, h in TSH:
        t0 = time.perf_counter()
        hb = HeldBatch(np.stack([c["rel0"] for c in cas]), np.stack([c["K"] for c in cas]),
                       [c["a_max"] for c in cas], ALT, Ts, horizon_orbits=1.0, h=h,
                       models=("cw",), ests=ESTS).run()
        wallbatch = time.perf_counter() - t0
        for i, c in enumerate(cas):
            t1 = time.perf_counter()
            cv = CaseView(hb, i)
            wallcv = time.perf_counter() - t1
            tm = cv.tm
            nomb = cv.grid("cw")
            tu = hm.updatetimes(Ts, T)
            tnode = np.append(tu, T)
            Ynode = hb.dense(i, tnode, cv.Fr, cv.Fl)
            satb = np.asarray(hb.sat_at_update[i], bool)
            for est in ESTS:
                t2 = time.perf_counter()
                mp = hm.heldsens(c["rel0"], c["K"], c["a_max"], ALT, Ts, est, tq=tm,
                                 nodes=8)
                w8 = time.perf_counter() - t2
                t3 = time.perf_counter()
                mp16 = hm.heldsens(c["rel0"], c["K"], c["a_max"], ALT, Ts, est, tq=tm,
                                   nodes=16)
                w16 = time.perf_counter() - t3
                eb = cv.grid(est)
                pb = np.linalg.norm(eb[:, :3], axis=1)
                jb = int(np.argmax(pb))
                fb = float(pb[jb])
                fm = mp["feature_m"]
                ebnode = cv.relrows(est, Ynode)
                nbnode = cv.relrows("cw", Ynode)
                emnode = mp["e_update"]
                smnode = mp["s_update"]
                satm = mp["sat_at_update"]
                nmin = min(satm.size, satb.size)
                mism = np.nonzero(satm[:nmin] != satb[:nmin])[0]
                nomh, eh = mapherm(hm, c, Ts, est, np.asarray(hb.t, float), tm)
                rows.append(dict(
                    Ts_s=Ts, h_batch_s=h, case=c["case"], wn_ratio=c["ratio"], a_max=c["a_max"],
                    estimator=est, batch_status=str(hb.status[est][i]),
                    feature_map_m=fm, feature_batch_m=fb,
                    feature_abs_diff_m=abs(fm - fb), feature_rel_diff=abs(fm - fb) / fm,
                    t_peak_map_s=mp["t_peak_s"], t_peak_batch_s=float(tm[jb]),
                    e_grid_maxdiff_pos_m=maxnorm(mp["e"][:, :3] - eb[:, :3]),
                    e_grid_maxdiff_pos_rel=maxnorm(mp["e"][:, :3] - eb[:, :3]) / fm,
                    e_grid_maxdiff_vel_mps=maxnorm(mp["e"][:, 3:] - eb[:, 3:]),
                    e_node_maxdiff_pos_m=maxnorm(emnode[:, :3] - ebnode[:, :3]),
                    e_node_maxdiff_pos_rel=maxnorm(emnode[:, :3] - ebnode[:, :3]) / fm,
                    e_grid_maxdiff_pos_vs_map_hermite_rel=maxnorm(eh[:, :3] - eb[:, :3]) / fm,
                    nom_grid_maxdiff_pos_m=maxnorm(mp["s0"][:, :3] - nomb[:, :3]),
                    nom_grid_maxdiff_pos_vs_map_hermite_m=maxnorm(nomh[:, :3] - nomb[:, :3]),
                    nom_grid_maxdiff_vel_mps=maxnorm(mp["s0"][:, 3:] - nomb[:, 3:]),
                    nom_node_maxdiff_pos_m=maxnorm(smnode[:, :3] - nbnode[:, :3]),
                    nom_node_maxdiff_pos_rel=maxnorm(smnode[:, :3] - nbnode[:, :3])
                    / float(np.linalg.norm(c["rel0"][:3])),
                    n_updates_map=int(satm.size), n_updates_batch=int(satb.size),
                    n_sat_map=int(satm.sum()), n_sat_batch=int(satb.sum()),
                    n_sat_flag_mismatch=int(mism.size) + abs(satm.size - satb.size),
                    first_sat_mismatch_idx=int(mism[0]) if mism.size else -1,
                    min_switch_margin_map=float(np.min(mp["switch_margin"])),
                    feature_map_nodes16_m=mp16["feature_m"],
                    feature_rel_diff_nodes8_vs_16=abs(mp16["feature_m"] - fm) / fm,
                    e_grid_maxdiff_nodes8_vs_16_rel=maxnorm(mp16["e"][:, :3] - mp["e"][:, :3]) / fm,
                    wall_map_nodes8_s=w8, wall_map_nodes16_s=w16,
                    wall_batch_run_all5cases_s=wallbatch, wall_batch_caseview_s=wallcv,
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
    for Ts, _ in TSH:
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
