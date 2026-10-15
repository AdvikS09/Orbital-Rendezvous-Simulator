"""Run the 24 preselected benchmark cases through the production pipeline and the independent
verifier (indverify/), then compare.

Production side (cwvalid, the code being verified):
  prod      production configuration (DOP853 rtol 1e-13, 0.5 s monitoring), full 0.5 s histories
  prodfine  identical propagation, monitoring grid 0.05 s (isolates the monitoring-resolution effect
            inside the production code)
  The production rows are also compared with the stored campaign rows (reuse check).

Independent side (indverify, implemented from indverify/SPEC.md without importing production code):
  Radau rtol 1e-8 ... 1e-12, DOP853 rtol 1e-11 ... 1e-13, self-written RK4 h = 1, 0.5, 0.25 s.
  Each run reports the production monitoring convention (0.5 s grid + Brent root) and a fine
  convention (0.01 s grid, every crossing root-refined).

Comparisons (per case, per branch ref / cw / so):
  * relative trajectories and applied-command norms on the 0.5 s grid (prod vs independent best)
  * peak CW-to-reference and SO-to-reference discrepancy (absolute and relative difference)
  * terminal range and speed
  * final-entry time and dwell margin under the production convention (implementation check)
    and under the fine convention (monitoring-resolution check)
  * branch status and arrival classification
  * convergence of every quantity along the independent solver ladders
Acceptance targets: |peak difference| <= 0.01 m (relative 1e-3 for peaks above 10 m) and
|entry-time difference| <= 0.1 s, tightened to |difference| <= |dwell margin| / 10 when the margin
is below 1 s.

Usage: python experiments/v2/verifier_run.py production|independent|compare [--workers 7]
Output: results/v2/raw/verifier/{production,independent}/, results/v2/derived/verifier/
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_common as C  # noqa: E402

RAWV = os.path.join(C.RAW, "verifier")
OUTD = os.path.join(C.DERIVED, "verifier")
LADDER = ([{"solver": "radau", "rtol": r, "max_step": 20.0, "label": f"radau_{r:.0e}"} for r in (1e-8, 1e-9, 1e-10, 1e-11, 1e-12)]
          + [{"solver": "dop853", "rtol": r, "max_step": 20.0, "label": f"dop853_{r:.0e}"} for r in (1e-11, 1e-12, 1e-13)]
          + [{"solver": "rk4", "h": h, "label": f"rk4_{h:g}"} for h in (1.0, 0.5, 0.25, 0.125)])
BEST = "radau_1e-12"
BR = {"ref": "eci", "cw": "cw", "so": "so"}


def _prod_one(args):
    case, dt_eval = args
    from cwvalid.experiment import RunConfig, run_scenario
    from cwvalid.scenario import Scenario
    scn = Scenario.from_dict(case)
    cfg = RunConfig(dt_eval=dt_eval)
    t0 = time.perf_counter()
    row, hist = run_scenario(scn, cfg, keep_histories=(dt_eval == 0.5), hist_stride=1)
    row["run_wall_s"] = time.perf_counter() - t0
    row["dt_eval"] = dt_eval
    return C.jsonable(row), hist


def cmd_production(workers):
    cases = json.load(open(os.path.join(RAWV, "cases.json")))
    od = os.path.join(RAWV, "production")
    os.makedirs(os.path.join(od, "histories"), exist_ok=True)
    t0 = time.perf_counter()
    jobs = [(c, 0.5) for c in cases] + [(c, 0.05) for c in cases]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_prod_one, jobs))
    rows = []
    for (c, dt), (row, hist) in zip(jobs, res):
        rows.append(row)
        if hist is not None:
            np.savez_compressed(os.path.join(od, "histories", f"{c['sid']}.npz"), **hist)
    with open(os.path.join(od, "production_rows.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    C.log_run("verifier", "python experiments/v2/verifier_run.py production", "completed", time.perf_counter() - t0)


def cmd_independent(workers):
    od = os.path.join(RAWV, "independent")
    os.makedirs(od, exist_ok=True)
    sp = os.path.join(od, "settings.json")
    json.dump(LADDER, open(sp, "w"), indent=1)
    t0 = time.perf_counter()
    cmd = [sys.executable, "-m", "indverify", "--cases", os.path.join(RAWV, "cases.json"), "--settings", sp,
           "--out", od, "--workers", str(workers), "--branches", "ref,cw,so", "--history-setting", BEST]
    r = subprocess.run(cmd, cwd=C.REPO)
    C.log_run("verifier", " ".join(["python -m indverify"] + cmd[3:]), "completed" if r.returncode == 0 else f"exit {r.returncode}",
              time.perf_counter() - t0)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def cmd_compare():
    cases = json.load(open(os.path.join(RAWV, "cases.json")))
    sel = pd.read_csv(os.path.join(RAWV, "selection.csv")).set_index("sid")
    P = pd.DataFrame(C.read_jsonl(os.path.join(RAWV, "production", "production_rows.jsonl")))
    prod = P[P.dt_eval == 0.5].set_index("sid")
    pfine = P[P.dt_eval == 0.05].set_index("sid")
    I = pd.DataFrame(C.read_jsonl(os.path.join(RAWV, "independent", "indverify_results.jsonl")))
    errs = I[I.get("error").notna()] if "error" in I else I.iloc[0:0]
    I = I[I.get("error").isna()] if "error" in I else I
    os.makedirs(OUTD, exist_ok=True)
    # reuse check: production rerun vs stored campaign rows
    reuse = []
    for c in cases:
        camp = c["meta"]["source_campaign"]
        stored = {r["sid"]: r for r in C.read_jsonl(os.path.join(C.RESULTS, camp, "results.jsonl"))}.get(c["sid"])
        p = prod.loc[c["sid"]]
        diffs = [k for k in ("cw_vs_eci_max_pos_disc_m", "eci_terminal_arrival_time_s", "cw_terminal_arrival_time_s",
                             "so_terminal_arrival_time_s", "eci_status", "so_status")
                 if stored is not None and stored.get(k) != (None if pd.isna(p.get(k)) else p.get(k))]
        reuse.append({"sid": c["sid"], "stored_found": stored is not None, "fields_differing": ";".join(diffs)})
    pd.DataFrame(reuse).to_csv(os.path.join(OUTD, "production_rerun_vs_stored.csv"), index=False)
    rows, conv = [], []
    for c in cases:
        sid = c["sid"]
        p, pf = prod.loc[sid], pfine.loc[sid]
        ind = I[I.sid == sid].set_index("setting_label")
        if BEST not in ind.index:
            rows.append({"sid": sid, "note": "independent best setting missing"})
            continue
        b = ind.loc[BEST]
        r = {"sid": sid, "rule": sel.loc[sid, "rule"], "reason": sel.loc[sid, "reason"], "horizon_orbits": c["horizon_orbits"]}
        for a, pa in (("cw", "cw"), ("so", "so")):
            pk_p = _f(p.get(f"{pa}_vs_eci_max_pos_disc_m"))
            pk_i = _f(b.get(f"{a}_vs_ref_peak_pos_m"))
            r[f"{a}_ref_peak_prod_m"], r[f"{a}_ref_peak_ind_m"] = pk_p, pk_i
            r[f"{a}_ref_peak_absdiff_m"] = abs(pk_p - pk_i)
            r[f"{a}_ref_peak_reldiff"] = abs(pk_p - pk_i) / max(abs(pk_p), 1e-12)
            r[f"{a}_ref_peak_fine_ind_m"] = _f(b.get(f"{a}_vs_ref_peak_pos_fine_m"))
        for ib, pbn in BR.items():
            st_p, st_i = str(p.get(f"{pbn}_status")), str(b.get(f"{ib}_status"))
            r[f"{ib}_status_prod"], r[f"{ib}_status_ind"] = st_p, st_i
            r[f"{ib}_status_agree"] = {"ok": "completed", "terminated": "domain_exit", "failed": "numerical_failure"}.get(st_p) == st_i
            r[f"{ib}_t_end_prod_s"], r[f"{ib}_t_end_ind_s"] = _f(p.get(f"{pbn}_t_final_s")), _f(b.get(f"{ib}_t_end_s"))
            r[f"{ib}_final_range_prod_m"], r[f"{ib}_final_range_ind_m"] = _f(p.get(f"{pbn}_final_pos_m")), _f(b.get(f"{ib}_final_range_m"))
            r[f"{ib}_final_speed_prod_mps"], r[f"{ib}_final_speed_ind_mps"] = _f(p.get(f"{pbn}_final_speed_mps")), _f(b.get(f"{ib}_final_speed_mps"))
            r[f"{ib}_arrived_prod"] = bool(p.get(f"{pbn}_terminal_arrived"))
            r[f"{ib}_arrived_prodfine"] = bool(pf.get(f"{pbn}_terminal_arrived"))
            r[f"{ib}_arrived_ind_prodconv"] = bool(b.get(f"{ib}_prod_arrived"))
            r[f"{ib}_arrived_ind_fine"] = bool(b.get(f"{ib}_fine_arrived"))
            r[f"{ib}_entry_prod_s"] = _f(p.get(f"{pbn}_terminal_arrival_time_s"))
            r[f"{ib}_entry_prodfine_s"] = _f(pf.get(f"{pbn}_terminal_arrival_time_s"))
            r[f"{ib}_entry_ind_prodconv_s"] = _f(b.get(f"{ib}_prod_entry_time_s"))
            r[f"{ib}_entry_ind_fine_s"] = _f(b.get(f"{ib}_fine_entry_time_s"))
            r[f"{ib}_margin_prod_s"] = _f(p.get(f"{pbn}_terminal_dwell_margin_s"))
            r[f"{ib}_margin_ind_prodconv_s"] = _f(b.get(f"{ib}_prod_dwell_margin_s"))
            r[f"{ib}_margin_ind_fine_s"] = _f(b.get(f"{ib}_fine_dwell_margin_s"))
            r[f"{ib}_entry_impl_diff_s"] = abs(r[f"{ib}_entry_prod_s"] - r[f"{ib}_entry_ind_prodconv_s"])
            r[f"{ib}_entry_monitor_diff_s"] = abs(r[f"{ib}_entry_ind_prodconv_s"] - r[f"{ib}_entry_ind_fine_s"])
            r[f"{ib}_n_crossings_prodconv"] = b.get(f"{ib}_prod_n_crossings")
            r[f"{ib}_n_crossings_fine"] = b.get(f"{ib}_fine_n_crossings")
        # classification against the reference (prod convention), production vs independent
        for a in ("cw", "so"):
            r[f"outcome_{a}_prod"] = f"{r[a + '_arrived_prod']}/{r['ref_arrived_prod']}"
            r[f"outcome_{a}_ind"] = f"{r[a + '_arrived_ind_prodconv']}/{r['ref_arrived_ind_prodconv']}"
            r[f"outcome_{a}_ind_fine"] = f"{r[a + '_arrived_ind_fine']}/{r['ref_arrived_ind_fine']}"
        # trajectory and command-norm differences on the 0.5 s grid
        hp = os.path.join(RAWV, "production", "histories", f"{sid}.npz")
        hi = os.path.join(RAWV, "independent", "histories", f"{sid}.npz")
        if os.path.exists(hp) and os.path.exists(hi):
            zp, zi = np.load(hp), np.load(hi)
            for ib, pbn in BR.items():
                tp, ti = zp[f"{pbn}_t"], zi[f"{ib}_t"]
                n = min(tp.size, ti.size)
                same = np.allclose(tp[:n], ti[:n], atol=1e-9)
                if same and n > 1:
                    dpos = np.linalg.norm(zp[f"{pbn}_S"][:n, :3] - zi[f"{ib}_rho"][:n], axis=1)
                    dvel = np.linalg.norm(zp[f"{pbn}_S"][:n, 3:] - zi[f"{ib}_rhodot"][:n], axis=1)
                    dacc = np.abs(np.linalg.norm(zp[f"{pbn}_app"][:n], axis=1) - zi[f"{ib}_accnorm"][:n])
                    rng = np.linalg.norm(zp[f"{pbn}_S"][:n, :3], axis=1)
                    r[f"{ib}_traj_maxdiff_m"] = float(dpos.max())
                    r[f"{ib}_traj_maxreldiff"] = float((dpos / np.maximum(rng, 1.0)).max())
                    r[f"{ib}_vel_maxdiff_mps"] = float(dvel.max())
                    r[f"{ib}_accnorm_maxdiff_mps2"] = float(dacc.max())
                    r[f"{ib}_traj_compared_until_s"] = float(tp[n - 1])
                else:
                    r[f"{ib}_traj_note"] = "time grids differ; not compared"
        # a branch that failed numerically in both implementations: its peak difference is not a physical
        # quantity (the failed polynomial solution diverges); compare failure times instead
        for a in ("cw", "so"):
            if r[f"{a}_status_prod"] == "failed" and r[f"{a}_status_ind"] == "numerical_failure":
                r[f"{a}_ref_peak_absdiff_m"] = float("nan")
                r[f"{a}_ref_peak_reldiff"] = float("nan")
                r[f"{a}_note"] = "numerical failure in both implementations; peak not compared"
                r[f"{a}_failure_time_diff_s"] = abs(r[f"{a}_t_end_prod_s"] - r[f"{a}_t_end_ind_s"])
        # targets
        for a in ("cw", "so"):
            pk = r[f"{a}_ref_peak_prod_m"]
            tol = 0.01 if not (pk > 10.0) else 1e-3 * pk
            d = r[f"{a}_ref_peak_absdiff_m"]
            r[f"{a}_peak_within_target"] = bool(d <= tol) if (np.isfinite(pk) and np.isfinite(d)) else None
        for ib in BR:
            m = r[f"{ib}_margin_prod_s"]
            tol = 0.1 if not (np.isfinite(m) and abs(m) < 1.0) else abs(m) / 10.0
            d = r[f"{ib}_entry_impl_diff_s"]
            r[f"{ib}_entry_tol_s"] = tol
            r[f"{ib}_entry_within_target"] = bool(d <= tol) if np.isfinite(d) else None
        rows.append(r)
        # convergence along the independent ladders (difference to the finest setting of each solver family)
        for fam, finest in (("radau", "radau_1e-12"), ("dop853", "dop853_1e-13"), ("rk4", "rk4_0.125")):
            if finest not in ind.index:
                continue
            fb = ind.loc[finest]
            for lab in ind.index:
                if not lab.startswith(fam):
                    continue
                x = ind.loc[lab]
                cr = {"sid": sid, "family": fam, "setting": lab, "finest": finest, "wall_s": _f(x.get("wall_s"))}
                for a in ("cw", "so"):
                    cr[f"{a}_peak_diff_to_finest_m"] = abs(_f(x.get(f"{a}_vs_ref_peak_pos_m")) - _f(fb.get(f"{a}_vs_ref_peak_pos_m")))
                for ib in BR:
                    cr[f"{ib}_entry_diff_to_finest_s"] = abs(_f(x.get(f"{ib}_prod_entry_time_s")) - _f(fb.get(f"{ib}_prod_entry_time_s")))
                    cr[f"{ib}_final_range_diff_to_finest_m"] = abs(_f(x.get(f"{ib}_final_range_m")) - _f(fb.get(f"{ib}_final_range_m")))
                    cr[f"{ib}_status"] = x.get(f"{ib}_status")
                    cr[f"{ib}_arrived_prodconv"] = x.get(f"{ib}_prod_arrived")
                conv.append(cr)
        # cross-family agreement of the finest settings (Radau 1e-12 vs DOP853 1e-13 vs RK4 0.125)
        fins = [x for x in ("radau_1e-12", "dop853_1e-13", "rk4_0.125") if x in ind.index]
        for x in fins[1:]:
            a0, a1 = ind.loc[fins[0]], ind.loc[x]
            xr = {"sid": sid, "family": "cross", "setting": x, "finest": fins[0], "wall_s": _f(a1.get("wall_s"))}
            for a in ("cw", "so"):
                xr[f"{a}_peak_diff_to_finest_m"] = abs(_f(a1.get(f"{a}_vs_ref_peak_pos_m")) - _f(a0.get(f"{a}_vs_ref_peak_pos_m")))
            for ib in BR:
                xr[f"{ib}_entry_diff_to_finest_s"] = abs(_f(a1.get(f"{ib}_prod_entry_time_s")) - _f(a0.get(f"{ib}_prod_entry_time_s")))
                xr[f"{ib}_final_range_diff_to_finest_m"] = abs(_f(a1.get(f"{ib}_final_range_m")) - _f(a0.get(f"{ib}_final_range_m")))
                xr[f"{ib}_status"] = a1.get(f"{ib}_status")
                xr[f"{ib}_arrived_prodconv"] = a1.get(f"{ib}_prod_arrived")
                xr[f"{ib}_label_agrees_with_finest"] = bool(a1.get(f"{ib}_prod_arrived")) == bool(a0.get(f"{ib}_prod_arrived"))
            conv.append(xr)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUTD, "verifier_comparison.csv"), index=False)
    pd.DataFrame(conv).to_csv(os.path.join(OUTD, "verifier_convergence.csv"), index=False)
    errs.to_csv(os.path.join(OUTD, "independent_errors.csv"), index=False)
    C.log_run("verifier", "python experiments/v2/verifier_run.py compare", "completed", 0.0, f"{len(R)} cases compared, {len(errs)} independent errors")
    pd.set_option("display.width", 250)
    cols = ["sid", "rule", "cw_ref_peak_prod_m", "cw_ref_peak_absdiff_m", "so_ref_peak_absdiff_m", "ref_entry_impl_diff_s",
            "ref_entry_monitor_diff_s", "ref_status_agree", "so_status_agree", "outcome_cw_prod", "outcome_cw_ind",
            "outcome_so_prod", "outcome_so_ind"]
    print(R[[c for c in cols if c in R]].to_string())
    print("independent errors:", len(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["production", "independent", "compare"])
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    {"production": lambda: cmd_production(a.workers), "independent": lambda: cmd_independent(a.workers),
     "compare": cmd_compare}[a.cmd]()


if __name__ == "__main__":
    main()
