"""C04: corrected numerical-resolution audit (replaces the derived columns of v3.0 results/numerical_resolution.csv
in a new versioned table; the v3.0 file is kept unchanged).

Sample. The same 108 development (pilot + dev) cases as v3.0, chosen by the stratified rule in
experiments/v3/p02_numerical_resolution.select (re-executed here to show that it reproduces the sample).
The rule was first committed on 2026-10-13 22:02 +0400 (commit e376cec), after the last pilot/development
record was completed (21:57:16); no earlier record of the rule exists, so it cannot be shown to precede the
development labels. The sample is a rule-based stratified development audit, not a demonstrably a-priori
sample, and not iid. Its role is assurance for the development numerics; it estimates no population rate.

What is recomputed
  held (D3, D4; 37 cases): everything, with the C01-corrected engine at half the production step
    (h/2; A_naive pass h = 0.25 s):
      truth  - the production peak algorithm (held_metrics.peak_discrepancy) at h/2 ("frozen algorithm") and
               a robust search for the supremum on the common valid interval [0, t_c] (metrics3.robust_peak),
               for cw-eci and cw-nerm, so the refined cross-check band uses refined values;
      features - the frozen feature by its definition (max |e_pos| on the 0.5 s monitoring grid INCLUDING
               t = T) at h/2, and the supremum over [0, T] by robust_peak (separate estimand);
      arrival  - metrics3.arrival_label on the refined dense output.
  continuous (D0-D2, D5; 71 cases): the frozen features A and B by their definition (2 s output grid, ends
    included) on a CW branch re-propagated with the refined solver (rtol 3e-14, absolute tolerances / 100,
    max step 5 s) and estimator tolerances 1e-11. The truth and arrival columns are taken unchanged from the
    v3.0 table (the continuous engine is unchanged in v3.1); the refined cross-check was not recomputed for
    continuous rows, so their refined band uses the production cross-check discrepancy (band_source column).

Labels and resolution
  labels come from cwvalid3.contracts.criterion_label (exit -> 'exit', failure -> 'failure', band rule), not
  from a bare threshold comparison; 'resolved' requires completed_valid truth in production and refined
  runs, a converged robust search, a finite uncertainty and num_unc < 0.1 x margin.

robust_peak (cwvalid3.metrics3), as used here: samples on grids of 0.5, 0.25, 0.125 and 0.0625 s with the
update / saturation-switch times and the interval ends added; on each grid, refines every sampled local maximum
whose value is at least half the sampled maximum plus the 5 largest local maxima (not every local maximum),
by bounded scalar minimisation and golden-section search between neighbouring samples; 'converged' means
the peak changed by less than max(1e-9, 1e-9 |peak|) between the 0.125 s and 0.0625 s grids; 'uncertainty'
is max(tolerance, that change, the two methods' disagreement). Every reported value is an evaluation of the
function, hence a lower bound on the true supremum; an excursion narrower than 0.0625 s that leaves no trace
in the samples would be missed by every level.

Outputs (v3_1_correction): results/numerical_resolution_v31.csv, results/numerical_resolution_label_changes.csv,
results/numerical_resolution_v31_summary.json, manuscript_assets/FigV31_S1_Numerics.{pdf,png},
manuscript_assets/TabV31_Numerics.tex.
Exit cases: their peaks are partial-horizon values (up to the exit) and are reported only in partial_* columns; the
full-horizon truth columns, truth margins and resolution ratios are NaN for them. Decisions: dec_<slot>_<tol>_* columns
(slot A = A or A_H, B = B or B_H, A_naive) give the family, frozen threshold, margin (production feature - threshold, from
the exact stored value) and status (stable / flip / tie_unresolved / abstain); the long form is
results/numerical_resolution_decisions_v31.csv. Arrival: compared only when production and refined truth are
completed_valid, from the stored Boolean fields (contract semantics); exits are 'not_applicable_exit'.

Usage: python code/experiments/v3_1/c04_numerical_audit.py [--workers 6]
       python code/experiments/v3_1/c04_numerical_audit.py --rederive   (derived columns only, no computation)
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import v31paths as P  # noqa: E402

sys.path.insert(0, os.path.join(P.REPO, "experiments", "v3"))
GRIDS = (0.25, 0.125, 0.0625)
POPULATION = ("rule-based stratified development audit (pilot + dev); selection rule first committed 2026-10-13 "
              "22:02 +0400, after the last development record (21:57:16), with no earlier record of the rule; not shown "
              "to be a priori; not iid")
HELD_STATUS = {"completed": "completed", "domain_exit": "domain_exit", "numerical_failure": "numerical_failure"}
CONT_STATUS = {"ok": "completed", "terminated": "domain_exit", "failed": "numerical_failure"}


def _init(repo):
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, "experiments", "v3"))


def refine_held(args):
    cell, items = args
    from cwvalid3 import metrics3 as M3
    from cwvalid3.domains import CELLS
    from cwvalid3.held_metrics import CaseView, estimator_feature, peak_discrepancy
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
    for i, (rec, stratum) in enumerate(items):
        cv, cvn = CaseView(hb, i), CaseView(hn, i)
        upd = hb.update_times
        out = {"case_id": rec["case_id"], "cell": cell, "split": rec["split"], "stratum": stratum, "kind": "held",
               "h_refined_s": h, "statuses_refined": ";".join(f"{b}:{hb.status[b][i]}" for b in ("cw", "nerm", "eci"))}
        for ref, tag in (("eci", "disc"), ("nerm", "discx")):
            tc = float(min(hb.t_end["cw"][i], hb.t_end[ref][i]))

            def d(t, ref=ref):
                Y = hb.dense(i, np.atleast_1d(t), cv.Fr, cv.Fl)
                return np.linalg.norm(cv.rel_from_rows("cw", Y)[:, :3] - cv.rel_from_rows(ref, Y)[:, :3], axis=1)
            pk = M3.robust_peak(d, 0.0, tc, grid_dt=0.5, extra_times=np.append(upd[upd <= tc], tc),
                                monitor_refinements=GRIDS, vectorized=True)
            pf = peak_discrepancy(cv, "cw", ref)
            out.update({f"{tag}_frozen_alg_refined_m": pf["peak_m"], f"{tag}_sup_refined_m": pk.peak,
                        f"{tag}_sup_uncertainty_m": pk.uncertainty, f"{tag}_sup_converged": pk.converged,
                        f"{tag}_sup_levels": ";".join(f"{dt}:{v!r}" for dt, v in zip(pk.refinement_dts, pk.refinement_values)),
                        f"{tag}_n_candidates": pk.n_candidates, f"{tag}_t_peak_refined_s": pk.t_peak, f"{tag}_common_end_s": tc})
        for e, batch, c in (("A_H", hb, cv), ("B_H", hb, cv), ("A_naive", hn, cvn)):
            key = {"A_H": "A", "B_H": "B", "A_naive": "A_naive"}[e]
            fr = estimator_feature(c, e)
            out[f"{key}_frozen_def_refined_m"] = fr["peak_m"]
            if batch.status[e][i] == "completed" and batch.status["cw"][i] == "completed":
                def f(t, batch=batch, c=c, e=e):
                    Y = batch.dense(c.i, np.atleast_1d(t), c.Fr, c.Fl)
                    return np.linalg.norm(c.rel_from_rows(e, Y)[:, :3], axis=1)
                u = batch.update_times
                pk = M3.robust_peak(f, 0.0, batch.T, grid_dt=0.5, extra_times=np.append(u, batch.T),
                                    monitor_refinements=GRIDS, vectorized=True)
                out.update({f"{key}_sup_refined_m": pk.peak, f"{key}_sup_converged": pk.converged,
                            f"{key}_sup_t_peak_s": pk.t_peak, f"{key}_sup_at_T": bool(abs(pk.t_peak - batch.T) < 1e-9)})
            else:
                out.update({f"{key}_sup_refined_m": math.nan, f"{key}_sup_converged": False})
        for b in ("cw", "eci"):
            if hb.status[b][i] == "completed":
                g = M3.terminal_g_fun(lambda t, b=b, cv=cv: cv.at(b, t))
                al = M3.arrival_label(g, hb.t_end[b][i], hb.T, grid_dt=0.25, tangency_tol=1e-7)
                out[f"{b}_arrival_refined"] = al.status
                out[f"{b}_dwell_margin_refined_s"] = al.dwell_margin_s
            else:
                out[f"{b}_arrival_refined"] = "incomplete_horizon"
                out[f"{b}_dwell_margin_refined_s"] = math.nan
        out["wall_s"] = (time.perf_counter() - t0) / len(items)
        rows.append(out)
    return rows


def refine_cont_features(args):
    rec, stratum = args
    from cwvalid.control import PDLaw
    from cwvalid.estimate import first_order_discrepancy
    from cwvalid.experiment import RunConfig
    from cwvalid.scenario import Scenario
    from cwvalid.simulate import SolverConfig
    from cwvalid3.continuous import estimator_matrix, run_branches_general
    from cwvalid3.controllers import MatrixLaw, pd_law
    t0 = time.perf_counter()
    s = Scenario.from_dict(rec["scenario"])
    pd_rule = rec["controller"].get("rule") == "scalar_pd"
    law = PDLaw.from_bandwidth(s.wn_ratio * s.n, s.a_max) if pd_rule else MatrixLaw(rec["K"], s.a_max, rec["controller"])
    solver = SolverConfig(rtol=3e-14, atol_rel_pos=1e-12, atol_rel_vel=1e-15, atol_eci_pos=1e-9, atol_eci_vel=1e-12,
                          atol_dv=1e-15, max_step=5.0)
    cfg = RunConfig(models=("cw",), reference="cw", crosscheck=None, solver=solver)
    sims, _ = run_branches_general(s, cfg, law)
    T = s.horizon
    out = {"case_id": rec["case_id"], "cw_status_feature_pass": sims["cw"].status}
    kw = dict(dt_out=2.0, rtol=1e-11, atol=1e-11)
    if pd_rule:
        out["A_frozen_def_refined_m"] = first_order_discrepancy(sims["cw"], law, s.n, s.R, T, **kw)["d1_peak_m"]
        out["B_frozen_def_refined_m"] = estimator_matrix(sims["cw"], pd_law(s.wn_ratio * s.n, s.a_max), s.n, s.R, T, "B", **kw)["peak_m"]
    else:
        out["A_frozen_def_refined_m"] = estimator_matrix(sims["cw"], law, s.n, s.R, T, "A", **kw)["peak_m"]
        out["B_frozen_def_refined_m"] = estimator_matrix(sims["cw"], law, s.n, s.R, T, "B", **kw)["peak_m"]
    out["wall_features_s"] = time.perf_counter() - t0
    return out


def _truth_refined(statuses, kind):
    m = HELD_STATUS if kind == "held" else CONT_STATUS
    st = dict(x.split(":", 1) for x in statuses.split(";"))
    cw, ref, cc = (m.get(st.get(b), "unresolved") for b in ("cw", "eci", "nerm"))
    if "numerical_failure" in (cw, ref, cc):
        return "numerical_failure"
    if ref == "domain_exit" and cc == "domain_exit":
        return "physical_domain_exit"
    if ref != cc or cw != "completed" or "unresolved" in (cw, ref, cc):
        return "unresolved"
    return "completed_valid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rederive", action="store_true",
                    help="re-derive labels, margins, decisions, summary, figure and table from the stored raw values (no computation)")
    a = ap.parse_args()
    if a.rederive:
        rederive()
        return
    import p02_numerical_resolution as P02
    from cwvalid3 import contracts as C
    from cwvalid3.domains import CELLS
    old = pd.read_csv(os.path.join(P.V3, "results", "numerical_resolution.csv"))
    freeze = json.load(open(os.path.join(P.V3, "raw", "calibration_freeze.json"), encoding="utf-8"))["selection"]
    replay = {}
    with open(os.path.join(P.V31_RAW, "results", "held_exit_replay_v31.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            replay[r["case_id"]] = r
    # 1. the sample: re-run the frozen selection rule and compare with the v3.0 table
    picks, recs, stored = {}, {}, {}
    for cell in CELLS:
        pk, _ = P02.select(cell)
        picks[cell] = pk
        r, s = P02.load_records(cell)
        recs.update(r)
        stored.update(s)
    sel = [(c, s) for cell in CELLS for c, s in picks[cell]]
    same_sample = sorted(sel) == sorted(zip(old.case_id, old.stratum))
    if not same_sample:
        raise SystemExit("the selection rule no longer reproduces the v3.0 sample")
    jobs_h = [(cell, [(recs[c], s) for c, s in picks[cell]]) for cell in CELLS if CELLS[cell]["cmd_mode"] == "sampled"]
    jobs_c = [(recs[c], s) for cell in CELLS if CELLS[cell]["cmd_mode"] == "continuous" for c, s in picks[cell]]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(P.REPO,)) as ex:
        fh_ = [ex.submit(refine_held, j) for j in jobs_h]
        fc_ = [ex.submit(refine_cont_features, j) for j in jobs_c]
        held_rows = [x for f in fh_ for x in f.result()]
        cont_feat = {x["case_id"]: x for x in (f.result() for f in fc_)}
    wall = time.perf_counter() - t0
    cpu = sum(r["wall_s"] for r in held_rows) + sum(x["wall_features_s"] for x in cont_feat.values())

    raw = assemble(old, stored, replay, held_rows, cont_feat)
    finish(raw, old, stored, replay, freeze, {"same_sample_as_v30": same_sample, "wall_s": wall, "worker_cpu_s_approx": cpu})


RAW_COLS = ["population", "case_id", "cell", "split", "stratum", "kind", "stored_engine_sha", "c01_replayed", "h_refined_s",
            "statuses_refined", "refined_peak_m", "sup_uncertainty_m", "sup_converged", "disc_n_candidates", "disc_t_peak_refined_s",
            "disc_common_end_s", "u_num_refined_m", "band_source", "A_frozen_def_refined_m", "B_frozen_def_refined_m",
            "A_naive_frozen_def_refined_m", "cw_status_feature_pass", "feature_fine_estimand", "A_fine_refined_m", "B_fine_refined_m",
            "A_naive_fine_refined_m", "A_v30_table_refined_m", "cw_arrival_refined", "cw_dwell_margin_refined_s", "eci_arrival_refined",
            "eci_dwell_margin_refined_s", "disc_frozen_alg_refined_m", "disc_sup_refined_m", "disc_sup_uncertainty_m",
            "disc_sup_converged", "disc_sup_levels", "discx_frozen_alg_refined_m", "discx_sup_refined_m", "discx_sup_uncertainty_m",
            "discx_sup_converged", "discx_sup_levels", "discx_n_candidates", "discx_t_peak_refined_s", "discx_common_end_s",
            "A_sup_refined_m", "A_sup_converged", "A_sup_t_peak_s", "A_sup_at_T", "B_sup_refined_m", "B_sup_converged",
            "B_sup_t_peak_s", "B_sup_at_T", "A_naive_sup_refined_m", "A_naive_sup_converged", "A_naive_sup_t_peak_s",
            "A_naive_sup_at_T", "wall_s"]
# (slot, family name, refined-feature column, stored production field) per record kind
FAMILIES = {"continuous": (("A", "A", "d1_peak_m"), ("B", "B", "d1u_peak_m")),
            "held": (("A", "A_H", "A_H_peak_m"), ("B", "B_H", "B_H_peak_m"), ("A_naive", "A_naive", "A_naive_peak_m"))}
ARRIVAL_FIELD = {"continuous": "{b}_terminal_arrived", "held": "{b}_arrived"}
REFINED_ARRIVAL = {"arrived": True, "not_arrived": False}


def assemble(old, stored, replay, held_rows, cont_feat):
    """Raw recomputed values only (no labels, margins or decisions)."""
    from cwvalid3.domains import CELLS
    rows = []
    hmap = {r["case_id"]: r for r in held_rows}
    for _, o in old.iterrows():
        cid, cell = o.case_id, o.cell
        kind = "held" if CELLS[cell]["cmd_mode"] == "sampled" else "continuous"
        st = stored[cid]
        r = {"population": POPULATION, "case_id": cid, "cell": cell, "split": o.split, "stratum": o.stratum, "kind": kind,
             "stored_engine_sha": (st.get("software") or {}).get("engine_sha"), "c01_replayed": cid in replay}
        if kind == "held":
            h = hmap[cid]
            r.update({k: v for k, v in h.items() if k not in ("case_id", "cell", "split", "stratum", "kind")})
            r.update({"refined_peak_m": h["disc_sup_refined_m"], "sup_uncertainty_m": h["disc_sup_uncertainty_m"],
                      "sup_converged": h["disc_sup_converged"], "u_num_refined_m": abs(h["disc_sup_refined_m"] - h["discx_sup_refined_m"]),
                      "band_source": "refined cross-check (cw-nerm robust supremum at h/2)",
                      "feature_fine_estimand": "supremum over [0, T] (robust_peak)", "A_fine_refined_m": h["A_sup_refined_m"],
                      "B_fine_refined_m": h["B_sup_refined_m"], "A_naive_fine_refined_m": h["A_naive_sup_refined_m"],
                      "A_v30_table_refined_m": o.A_refined_m})
        else:
            f = cont_feat[cid]
            r.update({"h_refined_s": math.nan, "statuses_refined": o.statuses_refined, "refined_peak_m": o.disc_refined_m,
                      "sup_uncertainty_m": o.peak_uncertainty_m, "sup_converged": bool(o.peak_converged),
                      "disc_n_candidates": o.n_peak_candidates, "disc_t_peak_refined_s": o.t_peak_refined_s,
                      "disc_common_end_s": o.common_end_s, "u_num_refined_m": math.nan,
                      "band_source": "production cross-check (refined nerm not recomputed for continuous rows)",
                      "A_frozen_def_refined_m": f["A_frozen_def_refined_m"], "B_frozen_def_refined_m": f["B_frozen_def_refined_m"],
                      "cw_status_feature_pass": f["cw_status_feature_pass"],
                      "feature_fine_estimand": "max on a 0.5 s output grid (v3.0 refined column; not nested with the 2 s grid)",
                      "A_fine_refined_m": o.A_refined_m, "B_fine_refined_m": o.B_refined_m, "A_naive_fine_refined_m": math.nan,
                      "A_v30_table_refined_m": o.A_refined_m, "cw_arrival_refined": o.cw_arrival_refined,
                      "cw_dwell_margin_refined_s": o.cw_dwell_margin_refined_s, "eci_arrival_refined": o.eci_arrival_refined,
                      "eci_dwell_margin_refined_s": o.eci_dwell_margin_refined_s})
        rows.append(r)
    return pd.DataFrame(rows)


def _arrival_compare(rec, tr, truth_refined, refined_status, kind, b):
    if tr["truth_status"] != "completed_valid" or truth_refined != "completed_valid":
        return "not_applicable_exit" if "physical_domain_exit" in (tr["truth_status"], truth_refined) else "not_compared"
    prod = rec.get(ARRIVAL_FIELD[kind].format(b=b))
    ref = REFINED_ARRIVAL.get(refined_status)
    if not isinstance(prod, bool) or ref is None:
        return "unresolved"
    return "agree" if prod == ref else "disagree"


def finish(raw, old, stored, replay, freeze, meta):
    """Every derived column, from the raw recomputed values and the exact stored records."""
    from cwvalid3 import contracts as C
    o_by = old.set_index("case_id")
    rows, dec_rows = [], []
    for _, x in raw.iterrows():
        r = x.to_dict()
        cid, kind, cell = r["case_id"], r["kind"], r["cell"]
        st = stored[cid]
        p31 = replay.get(cid, st)
        o = o_by.loc[cid]
        tr30, tr31 = C.truth_record(st, kind), C.truth_record(p31, kind)
        r["truth_prod_v30"], r["truth_prod_v31"] = tr30["truth_status"], tr31["truth_status"]
        r["truth_refined"] = _truth_refined(r["statuses_refined"], kind)
        if kind == "continuous":
            r["u_num_refined_m"] = tr31["u_num_m"]
        completed = r["truth_prod_v31"] == "completed_valid" and r["truth_refined"] == "completed_valid"
        rp = r["refined_peak_m"]
        # full-horizon truth columns only for completed cases; partial-horizon peaks in their own columns
        full = {"disc_prod_v30_m": tr30["disc_m"], "disc_prod_v31_m": tr31["disc_m"], "disc_refined_m": rp}
        part = {"partial_disc_prod_v30_m": tr30["disc_m"], "partial_disc_prod_v31_m": tr31["disc_m"], "partial_disc_refined_m": rp}
        for k, v in full.items():
            r[k] = v if completed else math.nan
        for k, v in part.items():
            r[k] = math.nan if completed else v
        r["u_num_prod_v31_m"] = tr31["u_num_m"] if completed else math.nan
        r["t_peak_prod_v31_s"] = p31.get("cw_vs_eci_t_peak_s" if kind == "held" else "cw_vs_eci_t_max_pos_disc_s")
        r["partial_delta_m"] = abs(rp - tr31["disc_m"]) if (not completed and np.isfinite(rp) and np.isfinite(tr31["disc_m"])) else math.nan
        r["delta_disc_m"] = abs(rp - tr31["disc_m"]) if completed else math.nan
        unc = r["sup_uncertainty_m"]
        r["num_unc_m"] = r["delta_disc_m"] + unc if (completed and np.isfinite(unc)) else math.nan
        for tol, tag in ((1.0, "1m"), (0.1, "0.1m")):
            r[f"label_prod_v30_{tag}"] = C.criterion_label(tr30, tol)
            r[f"label_prod_v31_{tag}"] = C.criterion_label(tr31, tol)
            r[f"label_refined_{tag}"] = C.criterion_label({"truth_status": r["truth_refined"], "disc_m": rp, "u_num_m": r["u_num_refined_m"]}, tol)
            r[f"label_v30_table_prod_{tag}"] = o[f"label_prod_{tag}"]
            r[f"label_v30_table_refined_{tag}"] = o[f"label_refined_{tag}"]
            r[f"resolved_v30_table_{tag}"] = bool(o[f"resolved_{tag}"])
            margin = abs(rp - tol) if completed else math.nan
            r[f"truth_margin_{tag}_m"] = margin
            r[f"unc_over_margin_{tag}"] = r["num_unc_m"] / margin if (completed and np.isfinite(r["num_unc_m"]) and margin > 0) else math.nan
            if not completed:
                res = {"physical_domain_exit": "exit", "numerical_failure": "failure"}.get(
                    r["truth_refined"] if r["truth_refined"] != "completed_valid" else r["truth_prod_v31"], "unresolved_truth")
            elif not bool(r["sup_converged"]):
                res = "not_converged"
            elif not np.isfinite(r["num_unc_m"]):
                res = "nonfinite_uncertainty"
            elif r["num_unc_m"] < 0.1 * margin:
                res = "resolved"
            else:
                res = "within_numerical_band"
            r[f"resolution_{tag}"] = res
            r[f"resolved_{tag}"] = res == "resolved"
            # screening decisions at the frozen thresholds, every audited family (exact stored production values)
            for slot, fam, field in FAMILIES[kind]:
                s = freeze.get(f"{cell}|{fam}|{tag}", {})
                thr = s.get("threshold") if s.get("status") == "selected" else None
                pv = st.get(field)
                pv = float(pv) if pv is not None else math.nan
                rv = float(r[f"{slot}_frozen_def_refined_m"])
                pre = f"dec_{slot}_{tag}"
                r[f"{pre}_family"] = f"{cell}|{fam}|{tag}"
                r[f"{pre}_threshold"] = thr
                if thr is None:
                    status, margin_m, change_m, ratio = "abstain", math.nan, rv - pv, math.nan
                else:
                    margin_m, change_m = pv - thr, rv - pv
                    flip = (np.isfinite(pv) and pv <= thr) != (np.isfinite(rv) and rv <= thr)
                    status = "flip" if flip else ("tie_unresolved" if margin_m == 0.0 else "stable")
                    ratio = abs(change_m) / abs(margin_m) if margin_m != 0.0 else math.nan
                r[f"{pre}_margin_m"], r[f"{pre}_status"] = margin_m, status
                dec_rows.append({"case_id": cid, "cell": cell, "stratum": r["stratum"], "family": f"{cell}|{fam}|{tag}",
                                 "threshold": thr, "feature_prod_m": pv, "feature_frozen_def_refined_m": rv, "margin_m": margin_m,
                                 "change_m": change_m, "change_over_margin": ratio, "status": status})
        for slot, fam, field in FAMILIES[kind]:
            pv, fd, fi = st.get(field), r[f"{slot}_frozen_def_refined_m"], r[f"{slot}_fine_refined_m"]
            ok = pv is not None and np.isfinite(pv) and np.isfinite(fd) and pv > 0
            r[f"{slot}_frozen_def_num_err_rel"] = abs(fd - pv) / pv if ok else math.nan
            r[f"{slot}_definition_gap_rel"] = (fi - fd) / fd if (ok and np.isfinite(fi) and fd > 0) else math.nan
            r[f"{slot}_prod_m"] = pv
        for b in ("cw", "eci"):
            r[f"arrival_{b}_comparison"] = _arrival_compare(st if b == "cw" else p31, tr31, r["truth_refined"], r[f"{b}_arrival_refined"], kind, b)
        rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(P.V31_RES, exist_ok=True)
    df.to_csv(P.guard_v30(os.path.join(P.V31_RES, "numerical_resolution_v31.csv")), index=False)
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(P.guard_v30(os.path.join(P.V31_RES, "numerical_resolution_decisions_v31.csv")), index=False)
    ch = []
    for _, r in df.iterrows():
        for tag in ("1m", "0.1m"):
            ch.append({"case_id": r.case_id, "cell": r.cell, "stratum": r.stratum, "criterion": tag,
                       "v30_table_label_prod": r[f"label_v30_table_prod_{tag}"], "v30_table_label_refined": r[f"label_v30_table_refined_{tag}"],
                       "v31_label_prod_stored": r[f"label_prod_v30_{tag}"], "v31_label_prod_corrected": r[f"label_prod_v31_{tag}"],
                       "v31_label_refined": r[f"label_refined_{tag}"], "v30_table_resolved": r[f"resolved_v30_table_{tag}"],
                       "v31_resolution": r[f"resolution_{tag}"],
                       "label_changed_vs_v30_table": r[f"label_v30_table_prod_{tag}"] != r[f"label_prod_v31_{tag}"],
                       "prod_vs_refined_label_differ": r[f"label_prod_v31_{tag}"] != r[f"label_refined_{tag}"],
                       "resolved_changed": r[f"resolved_v30_table_{tag}"] != r[f"resolved_{tag}"],
                       "decision_statuses": ";".join(f"{r[f'dec_{s}_{tag}_family']}:{r[f'dec_{s}_{tag}_status']}"
                                                     for s, _, _ in FAMILIES[r.kind])})
    chd = pd.DataFrame(ch)
    chd.to_csv(P.guard_v30(os.path.join(P.V31_RES, "numerical_resolution_label_changes.csv")), index=False)
    fin = lambda s: s[np.isfinite(s.astype(float))]
    comp = df[(df.truth_prod_v31 == "completed_valid") & (df.truth_refined == "completed_valid")]
    fam_err, fam_gap = {}, {}
    for kind, fams in FAMILIES.items():
        sub = df[df.kind == kind]
        for slot, fam, _ in fams:
            key = f"{fam} ({kind})"
            fam_err[key] = float(fin(sub[f"{slot}_frozen_def_num_err_rel"]).max())
            g = fin(sub[f"{slot}_definition_gap_rel"])
            fam_gap[key] = {"min": float(g.min()), "max": float(g.max()),
                            "fine_estimand": sub.feature_fine_estimand.iloc[0]}
    arr = {b: df[f"arrival_{b}_comparison"].value_counts().to_dict() for b in ("cw", "eci")}
    summ = {"n_cases": int(len(df)), **meta, "by_kind": {k: int(v) for k, v in df.kind.value_counts().items()},
            "n_completed_both": int(len(comp)),
            "resolution_counts": {tag: df[f"resolution_{tag}"].value_counts().to_dict() for tag in ("1m", "0.1m")},
            "v30_table_resolved_counts": {tag: int(df[f"resolved_v30_table_{tag}"].sum()) for tag in ("1m", "0.1m")},
            "n_label_changed_vs_v30_table": {tag: int(chd[(chd.criterion == tag)].label_changed_vs_v30_table.sum()) for tag in ("1m", "0.1m")},
            "n_prod_vs_refined_label_differ": {tag: int(chd[(chd.criterion == tag)].prod_vs_refined_label_differ.sum()) for tag in ("1m", "0.1m")},
            "max_abs_numerical_difference_completed_m": float(comp.delta_disc_m.max()),
            "max_num_unc_resolved_m": {tag: float(df[df[f"resolved_{tag}"]].num_unc_m.max()) for tag in ("1m", "0.1m")},
            "max_uncertainty_over_margin_resolved": {tag: float(df[df[f"resolved_{tag}"]][f"unc_over_margin_{tag}"].max()) for tag in ("1m", "0.1m")},
            "held_exit_rows": df[(df.kind == "held") & (df.truth_prod_v31 == "physical_domain_exit")][
                ["case_id", "cell", "partial_disc_prod_v30_m", "partial_disc_prod_v31_m", "disc_frozen_alg_refined_m",
                 "partial_disc_refined_m", "partial_delta_m"]].to_dict("records"),
            "feature_frozen_def_num_err_rel_max_by_family": fam_err,
            "feature_definition_gap_rel_by_family": fam_gap,
            "held_feature_sup_at_T": {k: int(df[df.kind == "held"][f"{k}_sup_at_T"].astype(str).eq("True").sum()) for k in ("A", "B", "A_naive")},
            "n_held": int((df.kind == "held").sum()),
            "not_converged": int((~df.sup_converged.astype(bool)).sum()),
            "arrival_comparison": arr,
            "decisions": {"n_decisions": int(dec.threshold.notna().sum()), "status_counts": dec.status.value_counts().to_dict(),
                          "n_flips": int((dec.status == "flip").sum()), "n_ties": int((dec.status == "tie_unresolved").sum()),
                          "ties": dec[dec.status == "tie_unresolved"][["case_id", "family", "feature_prod_m", "threshold",
                                                                      "feature_frozen_def_refined_m"]].to_dict("records"),
                          "max_change_over_margin_nontie": float(dec.change_over_margin.max()),
                          "largest_change_over_margin": dec.sort_values("change_over_margin", ascending=False).head(3)[
                              ["case_id", "family", "feature_prod_m", "feature_frozen_def_refined_m", "threshold", "change_over_margin"]].to_dict("records")}}
    with open(P.guard_v30(os.path.join(P.V31_RES, "numerical_resolution_v31_summary.json")), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, default=str)
    figure(df, old)
    table(df, summ)
    print(json.dumps({k: v for k, v in summ.items() if k != "held_exit_rows"}, indent=1, default=str))


def rederive():
    """Re-derive every derived column, the decisions, the summary, figure and table from the stored raw values (no computation)."""
    import p02_numerical_resolution as P02
    from cwvalid3.domains import CELLS
    cur = pd.read_csv(os.path.join(P.V31_RES, "numerical_resolution_v31.csv"), float_precision="round_trip")
    if "refined_peak_m" not in cur.columns:          # table written before the raw/derived split
        cur["refined_peak_m"] = cur["disc_refined_m"]
    raw = cur[[c for c in RAW_COLS if c in cur.columns]].copy()
    raw["population"] = POPULATION
    raw.loc[raw.kind == "continuous", "feature_fine_estimand"] = "max on a 0.5 s output grid (v3.0 refined column; not nested with the 2 s grid)"
    old = pd.read_csv(os.path.join(P.V3, "results", "numerical_resolution.csv"), float_precision="round_trip")
    freeze = json.load(open(os.path.join(P.V3, "raw", "calibration_freeze.json"), encoding="utf-8"))["selection"]
    replay = {}
    with open(os.path.join(P.V31_RAW, "results", "held_exit_replay_v31.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            replay[q["case_id"]] = q
    stored = {}
    for cell in CELLS:
        stored.update(P02.load_records(cell)[1])
    prev = json.load(open(os.path.join(P.V31_RES, "numerical_resolution_v31_summary.json"), encoding="utf-8"))
    meta = {k: prev[k] for k in ("same_sample_as_v30", "wall_s", "worker_cpu_s_approx") if k in prev}
    meta["rederived_from_stored_table"] = True
    finish(raw, old, stored, replay, freeze, meta)


def figure(df, old):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    comp = df[(df.truth_prod_v31 == "completed_valid") & (df.truth_refined == "completed_valid")]
    for kind, mk, col in (("continuous", "o", "#1f5fa8"), ("held", "s", "#b5562b")):
        s = comp[comp.kind == kind]
        y = np.maximum(s.num_unc_m.astype(float), 1e-16)
        ax[0].scatter(s.truth_margin_1m_m, y, marker=mk, s=22, color=col, label=f"{kind} (n={len(s)})", alpha=0.8, linewidths=0)
    xx = np.logspace(-6, 8, 50)
    ax[0].plot(xx, 0.1 * xx, color="0.3", lw=1)
    ax[0].fill_between(xx, 1e-17, 0.1 * xx, color="0.85", zorder=0)
    ax[0].set(xscale="log", yscale="log", xlabel="margin |d - 1 m| (m)", ylabel="numerical uncertainty (m)",
              title="(a) truth at 1 m: completed cases", ylim=(1e-17, 1e3))
    ax[0].text(1e-5, 1e1, "resolved: below line\n(uncertainty < 0.1 x margin)", fontsize=8)
    ax[0].legend(fontsize=8, loc="lower right")
    ex = df[(df.kind == "held") & (df.truth_prod_v31 == "physical_domain_exit")].copy()
    o = old.set_index("case_id")
    ex["v30_delta"] = [float(o.loc[c, "delta_disc_m"]) for c in ex.case_id]
    xs = np.arange(len(ex))
    ax[1].bar(xs - 0.2, ex.v30_delta, width=0.4, color="#999999", label="v3.0 (h vs h/2, stored end = start)")
    ax[1].bar(xs + 0.2, np.maximum(ex.partial_delta_m.astype(float), 1e-12), width=0.4, color="#b5562b", label="v3.1 (C01-corrected)")
    ax[1].set(yscale="log", xticks=xs, xticklabels=[f"{c}\n{i[:7]}" for c, i in zip(ex.cell, ex.case_id)],
              ylabel="|production - refined| partial-horizon peak (m)", title="(b) held exit cases: peak up to the exit")
    ax[1].tick_params(axis="x", labelsize=7)
    ax[1].legend(fontsize=8)
    for k, mk, col in (("A", "o", "#1f5fa8"), ("B", "^", "#5a9e3a"), ("A_naive", "s", "#b5562b")):  # A = A or A_H, B = B or B_H
        x = df[f"{k}_frozen_def_num_err_rel"].astype(float)
        y = df[f"{k}_definition_gap_rel"].astype(float).abs()
        m = np.isfinite(x) & np.isfinite(y)
        ax[2].scatter(np.maximum(x[m], 1e-17), np.maximum(y[m], 1e-17), marker=mk, s=20, color=col, label=k, alpha=0.8, linewidths=0)
    ax[2].plot([1e-17, 1], [1e-17, 1], color="0.6", lw=0.8, ls="--")
    ax[2].set(xscale="log", yscale="log", xlabel="numerical change of the frozen feature (relative)",
              ylabel="|fine-grid or supremum - frozen feature| (relative)", title="(c) features: numerics vs definition")
    ax[2].legend(fontsize=8)
    ax[2].text(1e-16, 3e-16, "exact zeros plotted at 1e-17", fontsize=7, color="0.35")
    fig.tight_layout()
    os.makedirs(P.V31_ASSETS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(P.guard_v30(os.path.join(P.V31_ASSETS, f"FigV31_S1_Numerics.{ext}")), dpi=200)


def table(df, summ):
    def n(x):
        return int(x)
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\hline",
             r"Kind & $n$ & resolved 1\,m & resolved 0.1\,m & exit & not converged & max $|\Delta d|$ (m) & max unc./margin \\",
             r"\hline"]
    for k in ("continuous", "held"):
        s = df[df.kind == k]
        res1, res01 = s[s.resolved_1m.astype(bool)], s[s["resolved_0.1m"].astype(bool)]
        dmax = res1.delta_disc_m.max()
        rmax = max(res1.unc_over_margin_1m.max(), res01["unc_over_margin_0.1m"].max())
        lines.append(f"{k} & {len(s)} & {n(s.resolved_1m.sum())} & {n(s['resolved_0.1m'].sum())} & "
                     f"{n((s['resolution_1m'] == 'exit').sum())} & {n((~s.sup_converged.astype(bool)).sum())} & "
                     f"{dmax:.1e} & {rmax:.1e} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    with open(P.guard_v30(os.path.join(P.V31_ASSETS, "TabV31_Numerics.tex")), "w", encoding="utf-8") as fh:
        fh.write("% v3.1 corrected numerical audit (rule-based stratified development sample, not iid).\n"
                 "% max |Delta d|: largest |refined - production| truth difference among resolved cases;\n"
                 "% max unc./margin: largest numerical uncertainty divided by the distance to the tolerance (1 m and 0.1 m).\n"
                 "% Exit cases are labelled exit and are not counted as resolved.\n")
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
