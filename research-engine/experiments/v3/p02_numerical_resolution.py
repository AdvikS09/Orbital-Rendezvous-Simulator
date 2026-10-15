"""P02 numerical resolution audit on a rule-based stratified DEVELOPMENT sample (results/numerical_resolution.csv).

[v3.1 correction C04: this audit was described as 'a-priori'. The selection rule below was first committed
on 2026-10-13 22:02 +0400, after the last pilot/development record (21:57:16), and no earlier record of it exists,
so the sample is rule-based and cannot be shown to be a priori. The v3.0 table also excluded t = T from the held estimator grid, compared a frozen feature
with a different (finer-grid) estimand as if the difference were numerical, labelled exits 'violation' and
called labels 'resolved' without requiring convergence. The corrected audit is
experiments/v3_1/c04_numerical_audit.py; this file is kept as the v3.0 producer.]

Strata (selection rules fixed here, applied to pilot + development records of each cell; ties broken
by case_id): ordinary (unsaturated, completed: first 3 by case_id), high_saturation (saturated time
fraction >= 0.5: first 3), estimator_disagreement (largest |log(A/B)|, 3), near_threshold_feature
(A or A_H closest to 1 m in log, 3), near_threshold_truth (truth closest to 1 m, 3, and to 0.1 m, 2),
physical_exit (all, up to 3), large_departure (largest truth discrepancy, 2). Held cells are the
sampled-command stratum.

For each selected case the production value is compared with a refined recomputation:
  continuous: tighter solver (rtol 3e-14, absolute tolerances / 100, max step 5 s), robust peak of the
     same-time discrepancy (cwvalid3.metrics3 robust_peak: on each grid of 0.5/0.25/0.125/0.0625 s, the sampled
     local maxima of at least half the sampled maximum plus the 5 largest are refined [v3.1: not every local maximum];
     saturation-switch times added), A and B on a 0.5 s output grid with rtol = atol = 1e-11;
  held: half the integration step (h/2, A_naive pass h = 0.25 s), robust peak on the Hermite dense
     output with every update time added, estimator peaks on a 0.125 s grid;
  arrival: metrics3.arrival_label on the refined dense output vs the production label.
Reported per case: differences, the metrics3 peak uncertainty, the decision margins |d - 1| and
|d - 0.1|, and whether the label is resolved (numerical difference + uncertainty < 0.1 x margin).
This is assurance for the specified numerical experiment, not flight validation.

Usage: python code/experiments/v3/p02_numerical_resolution.py [--workers 6]
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
sys.path.insert(0, HERE)
ARCH = os.path.dirname(REPO)
V3 = os.path.join(ARCH, "v3_extension")
MAN = os.path.join(V3, "raw", "manifests")
RES = os.path.join(V3, "raw", "results")

from cwvalid3 import ledger as L  # noqa: E402
from cwvalid3.domains import CELLS  # noqa: E402

GRIDS = (0.25, 0.125, 0.0625)


def select(cell):
    import analyze_dev as AD
    df = pd.concat([AD.case_table(cell, sp) for sp in ("pilot", "dev")], ignore_index=True).sort_values("case_id")
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


def load_records(cell):
    recs, res = {}, {}
    for sp in ("pilot", "dev"):
        for r in L.read_jsonl_strict(os.path.join(MAN, f"{cell}_{sp}.jsonl"), key="run_id"):
            recs[r["case_id"]] = r
        for r in L.read_jsonl_strict(os.path.join(RES, f"{cell}_{sp}.jsonl"), key="run_id"):
            res[r["case_id"]] = r
    return recs, res


def refine_continuous(args):
    rec, prod, stratum = args
    from cwvalid.control import PDLaw
    from cwvalid.estimate import first_order_discrepancy
    from cwvalid.experiment import RunConfig
    from cwvalid.scenario import Scenario
    from cwvalid.simulate import SolverConfig
    from cwvalid3 import metrics3 as M3
    from cwvalid3.continuous import estimator_matrix, run_branches_general
    from cwvalid3.controllers import MatrixLaw
    t0 = time.perf_counter()
    s = Scenario.from_dict(rec["scenario"])
    pd_rule = rec["controller"].get("rule") == "scalar_pd"
    law = PDLaw.from_bandwidth(s.wn_ratio * s.n, s.a_max) if pd_rule else MatrixLaw(rec["K"], s.a_max, rec["controller"])
    solver = SolverConfig(rtol=3e-14, atol_rel_pos=1e-12, atol_rel_vel=1e-15, atol_eci_pos=1e-9, atol_eci_vel=1e-12,
                          atol_dv=1e-15, max_step=5.0)
    cfg = RunConfig(models=("cw", "eci", "nerm"), reference="eci", crosscheck="nerm", solver=solver)
    sims, _ = run_branches_general(s, cfg, law)
    T = s.horizon
    out = {"case_id": rec["case_id"], "cell": rec["cell"], "split": rec["split"], "stratum": stratum,
           "statuses_refined": ";".join(f"{b}:{sims[b].status}" for b in sims)}
    te = min(sims["cw"].t_final, sims["eci"].t_final)
    extra = sorted(set(sims["cw"].sat_switch_times) | set(sims["eci"].sat_switch_times))

    def d(t):
        t = np.atleast_1d(t)
        return np.linalg.norm(sims["cw"].states(t)[:, :3] - sims["eci"].states(t)[:, :3], axis=1)
    pk = M3.robust_peak(d, 0.0, te, grid_dt=0.5, extra_times=extra, monitor_refinements=GRIDS, vectorized=True)
    out.update(disc_prod_m=prod.get("cw_vs_eci_max_pos_disc_m"), disc_refined_m=pk.peak, peak_uncertainty_m=pk.uncertainty,
               peak_converged=pk.converged, n_peak_candidates=pk.n_candidates, t_peak_refined_s=pk.t_peak,
               t_peak_prod_s=prod.get("cw_vs_eci_t_max_pos_disc_s"), common_end_s=te)
    if pd_rule:
        A = first_order_discrepancy(sims["cw"], law, s.n, s.R, T, dt_out=0.5, rtol=1e-11, atol=1e-11)["d1_peak_m"]
        from cwvalid3.controllers import pd_law
        B = estimator_matrix(sims["cw"], pd_law(s.wn_ratio * s.n, s.a_max), s.n, s.R, T, "B", dt_out=0.5, rtol=1e-11,
                             atol=1e-11)["peak_m"]
    else:
        A = estimator_matrix(sims["cw"], law, s.n, s.R, T, "A", dt_out=0.5, rtol=1e-11, atol=1e-11)["peak_m"]
        B = estimator_matrix(sims["cw"], law, s.n, s.R, T, "B", dt_out=0.5, rtol=1e-11, atol=1e-11)["peak_m"]
    out.update(A_prod_m=prod.get("d1_peak_m"), A_refined_m=A, B_prod_m=prod.get("d1u_peak_m"), B_refined_m=B)
    for b, key in (("cw", "cw_terminal_arrived"), ("eci", "eci_terminal_arrived")):
        if sims[b].status == "ok":
            g = M3.terminal_g_fun(lambda t, b=b: sims[b].states([t])[0])
            al = M3.arrival_label(g, sims[b].t_final, T, grid_dt=0.25, tangency_tol=1e-7)
            out[f"{b}_arrival_refined"] = al.status
            out[f"{b}_dwell_margin_refined_s"] = al.dwell_margin_s
        else:
            out[f"{b}_arrival_refined"] = "incomplete_horizon"
        out[f"{b}_arrival_prod"] = prod.get(key)
        out[f"{b}_dwell_margin_prod_s"] = prod.get(f"{b}_terminal_dwell_margin_s")
    out["wall_s"] = time.perf_counter() - t0
    return out


def refine_held(args):
    cell, items = args
    from cwvalid3 import metrics3 as M3
    from cwvalid3.held_metrics import CaseView
    from cwvalid3.sampled import HeldBatch
    Ts, alt = CELLS[cell]["Ts"], CELLS[cell]["altitude"]
    recs = [x[0] for x in items]
    rel0 = [r["scenario"]["rel0"] for r in recs]
    K = np.array([r["K"] for r in recs])
    am = [r["scenario"]["a_max"] for r in recs]
    t0 = time.perf_counter()
    h = min(Ts, 2.0) / 2.0
    hb = HeldBatch(rel0, K, am, alt, Ts, h=h, models=("cw", "nerm", "eci"), estimators=("A_H", "B_H")).run()
    hn = HeldBatch(rel0, K, am, alt, Ts, h=0.25, models=("cw",), estimators=("A_naive",)).run()
    rows = []
    for i, (rec, prod, stratum) in enumerate(items):
        cv, cvn = CaseView(hb, i), CaseView(hn, i)
        te = min(hb.t_end["cw"][i], hb.t_end["eci"][i])
        upd = hb.update_times

        def d(t, cv=cv):
            Y = hb.dense(cv.i, np.atleast_1d(t), cv.Fr, cv.Fl)
            return np.linalg.norm(cv.rel_from_rows("cw", Y)[:, :3] - cv.rel_from_rows("eci", Y)[:, :3], axis=1)
        pk = M3.robust_peak(d, 0.0, te, grid_dt=0.5, extra_times=upd[upd <= te], monitor_refinements=GRIDS, vectorized=True)

        def epk(batch, c, e):
            tq = np.arange(0.0, batch.T, 0.125)
            Y = batch.dense(c.i, tq, c.Fr, c.Fl)
            return float(np.max(np.linalg.norm(c.rel_from_rows(e, Y)[:, :3], axis=1)))
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
                g = M3.terminal_g_fun(lambda t, b=b, cv=cv: cv.at(b, t))
                al = M3.arrival_label(g, hb.t_end[b][i], hb.T, grid_dt=0.25, tangency_tol=1e-7)
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
    df["arrival_cw_agree"] = [(_lab(p) == _lab(r)) for p, r in zip(df.cw_arrival_prod, df.cw_arrival_refined)]
    df["arrival_eci_agree"] = [(_lab(p) == _lab(r)) for p, r in zip(df.eci_arrival_prod, df.eci_arrival_refined)]
    return df


def _lab(x):
    if isinstance(x, (bool, np.bool_)):
        return "arrived" if x else "not_arrived"
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "not_arrived"
    if str(x) == "incomplete_horizon":   # branch left the domain: not arrived under either convention
        return "not_arrived"
    return str(x)


def main():
    # v3.1: results/numerical_resolution.csv is a preserved v3.0 output (both the full run and --refinish rewrite it);
    # the corrected audit is experiments/v3_1/c04_numerical_audit.py.
    if os.path.exists(os.path.join(V3, "results", "numerical_resolution.csv")):
        sys.exit("results/numerical_resolution.csv is a preserved v3.0 output; use experiments/v3_1/c04_numerical_audit.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--refinish", action="store_true", help="recompute derived columns of the existing CSV only")
    a = ap.parse_args()
    if a.refinish:
        p = os.path.join(V3, "results", "numerical_resolution.csv")
        df = finish(pd.read_csv(p).drop(columns=["population"]))
        df.insert(0, "population", "a-priori stratified development audit (pilot + dev); not iid")
        df.to_csv(p, index=False)
        print(df[["arrival_cw_agree", "arrival_eci_agree", "resolved_1m", "resolved_0.1m"]].all())
        return
    jobs_c, jobs_h = [], []
    for cell in CELLS:
        picks, _ = select(cell)
        recs, res = load_records(cell)
        items = [(recs[c], res[c], s) for c, s in picks]
        if CELLS[cell]["cmd_mode"] == "continuous":
            jobs_c += items
        else:
            jobs_h.append((cell, items))
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        fh = [ex.submit(refine_held, j) for j in jobs_h]
        fc = [ex.submit(refine_continuous, j) for j in jobs_c]
        for f in fc:
            rows.append(f.result())
        for f in fh:
            rows.extend(f.result())
    df = finish(pd.DataFrame(rows))
    df.insert(0, "population", "a-priori stratified development audit (pilot + dev); not iid")
    df.to_csv(os.path.join(V3, "results", "numerical_resolution.csv"), index=False)
    print(f"{len(df)} cases in {time.perf_counter() - t0:.0f} s")
    with pd.option_context("display.width", 250):
        print(df[["cell", "stratum", "disc_prod_m", "delta_disc_m", "peak_uncertainty_m", "margin_1m_m",
                  "resolved_1m", "resolved_0.1m", "A_rel_change", "B_rel_change", "arrival_cw_agree",
                  "arrival_eci_agree"]].to_string())


if __name__ == "__main__":
    main()
