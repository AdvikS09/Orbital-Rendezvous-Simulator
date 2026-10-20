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
sys.path.insert(0, HERE)
ARCH = os.path.dirname(REPO)
V3 = os.path.join(ARCH, "v3")
MAN = os.path.join(V3, "raw", "manifests")
RES = os.path.join(V3, "raw", "results")

from cwvalid3 import ledger as L
from cwvalid3.domains import CELLS

GRIDS = (0.25, 0.125, 0.0625)


def select(cell):
    import anadev as AD
    df = pd.concat([AD.casetable(cell, sp) for sp in ("pilot", "dev")], ignore_index=True).sort_values("case_id")
    held = CELLS[cell]["cmd_mode"] == "sampled"
    fa, fb = ("f_A_H", "f_B_H") if held else ("f_A", "f_B")
    ok = df.truth_status == "completed_valid"
    picks = []

    def add(sub, k, stratum):
        for cid in sub.case_id.head(k):
            if cid not in [p[0] for p in picks]:
                picks.append((cid, stratum))
    add(df[ok & (df.sat_fraction.fillna(0) <= 0)], 3, "ordinary")
    add(df[ok & (df.sat_fraction.fillna(0) >= 0.5)], 3, "high_saturation")
    r = np.abs(np.log(df[fa].clip(lower=1e-300) / df[fb].clip(lower=1e-300)))
    add(df.assign(r=r)[ok].sort_values("r", ascending=False), 3, "estimator_disagreement")
    add(df.assign(r=np.abs(np.log(df[fa].clip(lower=1e-300))))[ok].sort_values("r"), 3, "near_threshold_feature")
    add(df.assign(r=np.abs(np.log(df.disc_m.clip(lower=1e-300))))[ok].sort_values("r"), 3, "near_threshold_truth_1m")
    add(df.assign(r=np.abs(np.log(df.disc_m.clip(lower=1e-300) / 0.1)))[ok].sort_values("r"), 2, "near_threshold_truth_0.1m")
    add(df[df.truth_status == "physical_domain_exit"], 3, "physical_exit")
    add(df[ok].sort_values("disc_m", ascending=False), 2, "large_departure")
    return picks, df.set_index("case_id")


def loadrecs(cell):
    recs, res = {}, {}
    for sp in ("pilot", "dev"):
        for r in L.readjsonlstrict(os.path.join(MAN, f"{cell}{sp.replace('_', '')}.jsonl"), key="run_id"):
            recs[r["case_id"]] = r
        for r in L.readjsonlstrict(os.path.join(RES, f"{cell}{sp.replace('_', '')}.jsonl"), key="run_id"):
            res[r["case_id"]] = r
    return recs, res


def refinecont(args):
    rec, prod, stratum = args
    from cwvalid.control import PDLaw
    from cwvalid.estimate import firstorderdisc
    from cwvalid.experiment import RunCfg
    from cwvalid.scenario import Scenario
    from cwvalid.simulate import SolverCfg
    from cwvalid3 import metrics3 as M3
    from cwvalid3.continuous import estmat, runbranchgen
    from cwvalid3.controllers import MatLaw
    t0 = time.perf_counter()
    s = Scenario.from_dict(rec["scenario"])
    pdrule = rec["controller"].get("rule") == "scalar_pd"
    law = PDLaw.bandw(s.wn_ratio * s.n, s.a_max) if pdrule else MatLaw(rec["K"], s.a_max, rec["controller"])
    solver = SolverCfg(rtol=3e-14, atol_rel_pos=1e-12, atol_rel_vel=1e-15, atol_eci_pos=1e-9, atol_eci_vel=1e-12,
                       atol_dv=1e-15, max_step=5.0)
    cfg = RunCfg(models=("cw", "eci", "nerm"), reference="eci", crosscheck="nerm", solver=solver)
    sims, _ = runbranchgen(s, cfg, law)
    T = s.horizon
    out = {"case_id": rec["case_id"], "cell": rec["cell"], "split": rec["split"], "stratum": stratum,
           "statuses_refined": ";".join(f"{b}:{sims[b].status}" for b in sims)}
    te = min(sims["cw"].tfinal, sims["eci"].tfinal)
    extra = sorted(set(sims["cw"].satswitchtimes) | set(sims["eci"].satswitchtimes))

    def d(t):
        t = np.atleast_1d(t)
        return np.linalg.norm(sims["cw"].states(t)[:, :3] - sims["eci"].states(t)[:, :3], axis=1)
    pk = M3.robustpeak(d, 0.0, te, griddt=0.5, extratimes=extra, monrefs=GRIDS, vec=True)
    out.update(disc_prod_m=prod.get("cw_vs_eci_max_pos_disc_m"), disc_refined_m=pk.peak, peak_uncertainty_m=pk.uncertainty,
               peak_converged=pk.converged, n_peak_candidates=pk.n_candidates, t_peak_refined_s=pk.t_peak,
               t_peak_prod_s=prod.get("cw_vs_eci_t_max_pos_disc_s"), common_end_s=te)
    if pdrule:
        A = firstorderdisc(sims["cw"], law, s.n, s.R, T, dt_out=0.5, rtol=1e-11, atol=1e-11)["d1_peak_m"]
        from cwvalid3.controllers import pdlaw
        B = estmat(sims["cw"], pdlaw(s.wn_ratio * s.n, s.a_max), s.n, s.R, T, "B", dt_out=0.5, rtol=1e-11,
                   atol=1e-11)["peak_m"]
    else:
        A = estmat(sims["cw"], law, s.n, s.R, T, "A", dt_out=0.5, rtol=1e-11, atol=1e-11)["peak_m"]
        B = estmat(sims["cw"], law, s.n, s.R, T, "B", dt_out=0.5, rtol=1e-11, atol=1e-11)["peak_m"]
    out.update(A_prod_m=prod.get("d1_peak_m"), A_refined_m=A, B_prod_m=prod.get("d1u_peak_m"), B_refined_m=B)
    for b, key in (("cw", "cw_terminal_arrived"), ("eci", "eci_terminal_arrived")):
        if sims[b].status == "ok":
            g = M3.termgfun(lambda t, b=b: sims[b].states([t])[0])
            al = M3.arrlabel(g, sims[b].tfinal, T, griddt=0.25, tangency_tol=1e-7)
            out[f"{b}_arrival_refined"] = al.status
            out[f"{b}_dwell_margin_refined_s"] = al.dwell_margin_s
        else:
            out[f"{b}_arrival_refined"] = "incomplete_horizon"
        out[f"{b}_arrival_prod"] = prod.get(key)
        out[f"{b}_dwell_margin_prod_s"] = prod.get(f"{b}_terminal_dwell_margin_s")
    out["wall_s"] = time.perf_counter() - t0
    return out


def refineheld(args):
    cell, items = args
    from cwvalid3 import metrics3 as M3
    from cwvalid3.heldmet import CaseView
    from cwvalid3.sampled import HeldBatch
    Ts, alt = CELLS[cell]["Ts"], CELLS[cell]["altitude"]
    recs = [x[0] for x in items]
    rel0 = [r["scenario"]["rel0"] for r in recs]
    K = np.array([r["K"] for r in recs])
    am = [r["scenario"]["a_max"] for r in recs]
    t0 = time.perf_counter()
    h = min(Ts, 2.0) / 2.0
    hb = HeldBatch(rel0, K, am, alt, Ts, h=h, models=("cw", "nerm", "eci"), ests=("A_H", "B_H")).run()
    hn = HeldBatch(rel0, K, am, alt, Ts, h=0.25, models=("cw",), ests=("A_naive",)).run()
    rows = []
    for i, (rec, prod, stratum) in enumerate(items):
        cv, cvn = CaseView(hb, i), CaseView(hn, i)
        te = min(hb.t_end["cw"][i], hb.t_end["eci"][i])
        upd = hb.updatetimes

        def d(t, cv=cv):
            Y = hb.dense(cv.i, np.atleast_1d(t), cv.Fr, cv.Fl)
            return np.linalg.norm(cv.relrows("cw", Y)[:, :3] - cv.relrows("eci", Y)[:, :3], axis=1)
        pk = M3.robustpeak(d, 0.0, te, griddt=0.5, extratimes=upd[upd <= te], monrefs=GRIDS, vec=True)

        def epk(batch, c, e):
            tq = np.arange(0.0, batch.T, 0.125)
            Y = batch.dense(c.i, tq, c.Fr, c.Fl)
            return float(np.max(np.linalg.norm(c.relrows(e, Y)[:, :3], axis=1)))
        out = {"case_id": rec["case_id"], "cell": cell, "split": rec["split"], "stratum": stratum,
               "statuses_refined": ";".join(f"{b}:{hb.status[b][i]}" for b in ("cw", "nerm", "eci")),
               "disc_prod_m": prod.get("cw_vs_eci_peak_m"), "disc_refined_m": pk.peak, "peak_uncertainty_m": pk.uncertainty,
               "peak_converged": pk.converged, "n_peak_candidates": pk.n_candidates, "t_peak_refined_s": pk.t_peak,
               "t_peak_prod_s": prod.get("cw_vs_eci_t_peak_s"), "common_end_s": te, "h_refined_s": h,
               "A_prod_m": prod.get("A_H_peak_m"), "A_refined_m": epk(hb, cv, "A_H"),
               "B_prod_m": prod.get("B_H_peak_m"), "B_refined_m": epk(hb, cv, "B_H"),
               "A_naive_prod_m": prod.get("A_naive_peak_m"), "A_naive_refined_m": epk(hn, cvn, "A_naive")}
        for b in ("cw", "eci"):
            if hb.status[b][i] == "completed":
                g = M3.termgfun(lambda t, b=b, cv=cv: cv.at(b, t))
                al = M3.arrlabel(g, hb.t_end[b][i], hb.T, griddt=0.25, tangency_tol=1e-7)
                out[f"{b}_arrival_refined"] = al.status
                out[f"{b}_dwell_margin_refined_s"] = al.dwell_margin_s
            else:
                out[f"{b}_arrival_refined"] = "incomplete_horizon"
            out[f"{b}_arrival_prod"] = prod.get(f"{b}_arrived")
            out[f"{b}_dwell_margin_prod_s"] = prod.get(f"{b}_dwell_margin_s")
        out["wall_s"] = (time.perf_counter() - t0) / len(items)
        rows.append(out)
    return rows


def finish(df):
    df["delta_disc_m"] = (df.disc_refined_m - df.disc_prod_m).abs()
    df["num_unc_m"] = df.delta_disc_m + df.peak_uncertainty_m.fillna(0)
    for tol, tag in ((1.0, "1m"), (0.1, "0.1m")):
        m = (df.disc_refined_m - tol).abs()
        df[f"margin_{tag}_m"] = m
        df[f"unc_over_margin_{tag}"] = df.num_unc_m / m
        df[f"resolved_{tag}"] = df.num_unc_m < 0.1 * m
        df[f"label_prod_{tag}"] = np.where(df.disc_prod_m <= tol, "ok", "violation")
        df[f"label_refined_{tag}"] = np.where(df.disc_refined_m <= tol, "ok", "violation")
    for f in ("A", "B", "A_naive"):
        if f"{f}_prod_m" in df:
            df[f"{f}_rel_change"] = (df[f"{f}_refined_m"] - df[f"{f}_prod_m"]).abs() / df[f"{f}_prod_m"].abs()
    df["arrival_cw_agree"] = [(lab(p) == lab(r)) for p, r in zip(df.cw_arrival_prod, df.cw_arrival_refined)]
    df["arrival_eci_agree"] = [(lab(p) == lab(r)) for p, r in zip(df.eci_arrival_prod, df.eci_arrival_refined)]
    return df


def lab(x):
    if isinstance(x, (bool, np.bool_)):
        return "arrived" if x else "not_arrived"
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "not_arrived"
    if str(x) == "incomplete_horizon":
        return "not_arrived"
    return str(x)


def main():
    if os.path.exists(os.path.join(V3, "results", "numerical.csv")):
        sys.exit("results/numerical.csv is a preserved v3.0 output; use experiments/v31/c04audit.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--refinish", action="store_true")
    a = ap.parse_args()
    if a.refinish:
        p = os.path.join(V3, "results", "numerical.csv")
        df = finish(pd.read_csv(p).drop(columns=["population"]))
        df.insert(0, "population", "a-priori stratified development audit (pilot + dev); not iid")
        df.to_csv(p, index=False)
        print(df[["arrival_cw_agree", "arrival_eci_agree", "resolved_1m", "resolved_0.1m"]].all())
        return
    jobsc, jobsh = [], []
    for cell in CELLS:
        picks, _ = select(cell)
        recs, res = loadrecs(cell)
        items = [(recs[c], res[c], s) for c, s in picks]
        if CELLS[cell]["cmd_mode"] == "continuous":
            jobsc += items
        else:
            jobsh.append((cell, items))
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        fh = [ex.submit(refineheld, j) for j in jobsh]
        fc = [ex.submit(refinecont, j) for j in jobsc]
        for f in fc:
            rows.append(f.result())
        for f in fh:
            rows.extend(f.result())
    df = finish(pd.DataFrame(rows))
    df.insert(0, "population", "a-priori stratified development audit (pilot + dev); not iid")
    df.to_csv(os.path.join(V3, "results", "numerical.csv"), index=False)
    print(f"{len(df)} cases in {time.perf_counter() - t0:.0f} s")
    with pd.option_context("display.width", 250):
        print(df[["cell", "stratum", "disc_prod_m", "delta_disc_m", "peak_uncertainty_m", "margin_1m_m",
                  "resolved_1m", "resolved_0.1m", "A_rel_change", "B_rel_change", "arrival_cw_agree",
                  "arrival_eci_agree"]].to_string())


if __name__ == "__main__":
    main()
