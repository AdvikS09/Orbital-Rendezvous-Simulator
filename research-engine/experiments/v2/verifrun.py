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
import v2common as C

RAWV = os.path.join(C.RAW, "verifier")
OUTD = os.path.join(C.DERIVED, "verifier")
LADDER = ([{"solver": "radau", "rtol": r, "max_step": 20.0, "label": f"radau_{r:.0e}"} for r in (1e-8, 1e-9, 1e-10, 1e-11, 1e-12)]
          + [{"solver": "dop853", "rtol": r, "max_step": 20.0, "label": f"dop853_{r:.0e}"} for r in (1e-11, 1e-12, 1e-13)]
          + [{"solver": "rk4", "h": h, "label": f"rk4_{h:g}"} for h in (1.0, 0.5, 0.25, 0.125)])
BEST = "radau_1e-12"
BR = {"ref": "eci", "cw": "cw", "so": "so"}


def prodone(args):
    case, dteval = args
    from cwvalid.experiment import RunCfg, runscen
    from cwvalid.scenario import Scenario
    scn = Scenario.from_dict(case)
    cfg = RunCfg(dt_eval=dteval)
    t0 = time.perf_counter()
    row, hist = runscen(scn, cfg, keep_histories=(dteval == 0.5), hist_stride=1)
    row["run_wall_s"] = time.perf_counter() - t0
    row["dt_eval"] = dteval
    return C.jsonable(row), hist


def cmdprod(workers):
    cas = json.load(open(os.path.join(RAWV, "cases.json")))
    od = os.path.join(RAWV, "production")
    os.makedirs(os.path.join(od, "histories"), exist_ok=True)
    t0 = time.perf_counter()
    jobs = [(c, 0.5) for c in cas] + [(c, 0.05) for c in cas]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(prodone, jobs))
    rows = []
    for (c, dt), (row, hist) in zip(jobs, res):
        rows.append(row)
        if hist is not None:
            np.savez_compressed(os.path.join(od, "histories", f"{c['sid']}.npz"), **hist)
    with open(os.path.join(od, "productionrows.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    C.logrun("verifier", "python experiments/v2/verifrun.py production", "completed", time.perf_counter() - t0)


def cmdindep(workers):
    od = os.path.join(RAWV, "independent")
    os.makedirs(od, exist_ok=True)
    sp = os.path.join(od, "settings.json")
    json.dump(LADDER, open(sp, "w"), indent=1)
    t0 = time.perf_counter()
    cmd = [sys.executable, "-m", "indverify", "--cases", os.path.join(RAWV, "cases.json"), "--settings", sp,
           "--out", od, "--workers", str(workers), "--branches", "ref,cw,so", "--history-setting", BEST]
    r = subprocess.run(cmd, cwd=C.REPO)
    C.logrun("verifier", " ".join(["python -m indverify"] + cmd[3:]), "completed" if r.returncode == 0 else f"exit {r.returncode}",
             time.perf_counter() - t0)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def cmdcmp():
    cas = json.load(open(os.path.join(RAWV, "cases.json")))
    sel = pd.read_csv(os.path.join(RAWV, "selection.csv")).set_index("sid")
    P = pd.DataFrame(C.readjsonl(os.path.join(RAWV, "production", "productionrows.jsonl")))
    prod = P[P.dt_eval == 0.5].set_index("sid")
    pfine = P[P.dt_eval == 0.05].set_index("sid")
    I = pd.DataFrame(C.readjsonl(os.path.join(RAWV, "independent", "indverifyresults.jsonl")))
    errs = I[I.get("error").notna()] if "error" in I else I.iloc[0:0]
    I = I[I.get("error").isna()] if "error" in I else I
    os.makedirs(OUTD, exist_ok=True)
    reuse = []
    for c in cas:
        camp = c["meta"]["source_campaign"]
        stored = {r["sid"]: r for r in C.readjsonl(os.path.join(C.RESULTS, camp, "results.jsonl"))}.get(c["sid"])
        p = prod.loc[c["sid"]]
        diffs = [k for k in ("cw_vs_eci_max_pos_disc_m", "eci_terminal_arrival_time_s", "cw_terminal_arrival_time_s",
                             "so_terminal_arrival_time_s", "eci_status", "so_status")
                 if stored is not None and stored.get(k) != (None if pd.isna(p.get(k)) else p.get(k))]
        reuse.append({"sid": c["sid"], "stored_found": stored is not None, "fields_differing": ";".join(diffs)})
    pd.DataFrame(reuse).to_csv(os.path.join(OUTD, "productionrerun.csv"), index=False)
    rows, conv = [], []
    for c in cas:
        sid = c["sid"]
        p, pf = prod.loc[sid], pfine.loc[sid]
        ind = I[I.sid == sid].set_index("setting_label")
        if BEST not in ind.index:
            rows.append({"sid": sid, "note": "independent best setting missing"})
            continue
        b = ind.loc[BEST]
        r = {"sid": sid, "rule": sel.loc[sid, "rule"], "reason": sel.loc[sid, "reason"], "horizon_orbits": c["horizon_orbits"]}
        for a, pa in (("cw", "cw"), ("so", "so")):
            pkp = f(p.get(f"{pa}_vs_eci_max_pos_disc_m"))
            pki = f(b.get(f"{a}_vs_ref_peak_pos_m"))
            r[f"{a}_ref_peak_prod_m"], r[f"{a}_ref_peak_ind_m"] = pkp, pki
            r[f"{a}_ref_peak_absdiff_m"] = abs(pkp - pki)
            r[f"{a}_ref_peak_reldiff"] = abs(pkp - pki) / max(abs(pkp), 1e-12)
            r[f"{a}_ref_peak_fine_ind_m"] = f(b.get(f"{a}_vs_ref_peak_pos_fine_m"))
        for ib, pbn in BR.items():
            stp, sti = str(p.get(f"{pbn}_status")), str(b.get(f"{ib}_status"))
            r[f"{ib}_status_prod"], r[f"{ib}_status_ind"] = stp, sti
            r[f"{ib}_status_agree"] = {"ok": "completed", "terminated": "domain_exit", "failed": "numerical_failure"}.get(stp) == sti
            r[f"{ib}_t_end_prod_s"], r[f"{ib}_t_end_ind_s"] = f(p.get(f"{pbn}_t_final_s")), f(b.get(f"{ib}_t_end_s"))
            r[f"{ib}_final_range_prod_m"], r[f"{ib}_final_range_ind_m"] = f(p.get(f"{pbn}_final_pos_m")), f(b.get(f"{ib}_final_range_m"))
            r[f"{ib}_final_speed_prod_mps"], r[f"{ib}_final_speed_ind_mps"] = f(p.get(f"{pbn}_final_speed_mps")), f(b.get(f"{ib}_final_speed_mps"))
            r[f"{ib}_arrived_prod"] = bool(p.get(f"{pbn}_terminal_arrived"))
            r[f"{ib}_arrived_prodfine"] = bool(pf.get(f"{pbn}_terminal_arrived"))
            r[f"{ib}_arrived_ind_prodconv"] = bool(b.get(f"{ib}_prod_arrived"))
            r[f"{ib}_arrived_ind_fine"] = bool(b.get(f"{ib}_fine_arrived"))
            r[f"{ib}_entry_prod_s"] = f(p.get(f"{pbn}_terminal_arrival_time_s"))
            r[f"{ib}_entry_prodfine_s"] = f(pf.get(f"{pbn}_terminal_arrival_time_s"))
            r[f"{ib}_entry_ind_prodconv_s"] = f(b.get(f"{ib}_prod_entry_time_s"))
            r[f"{ib}_entry_ind_fine_s"] = f(b.get(f"{ib}_fine_entry_time_s"))
            r[f"{ib}_margin_prod_s"] = f(p.get(f"{pbn}_terminal_dwell_margin_s"))
            r[f"{ib}_margin_ind_prodconv_s"] = f(b.get(f"{ib}_prod_dwell_margin_s"))
            r[f"{ib}_margin_ind_fine_s"] = f(b.get(f"{ib}_fine_dwell_margin_s"))
            r[f"{ib}_entry_impl_diff_s"] = abs(r[f"{ib}_entry_prod_s"] - r[f"{ib}_entry_ind_prodconv_s"])
            r[f"{ib}_entry_monitor_diff_s"] = abs(r[f"{ib}_entry_ind_prodconv_s"] - r[f"{ib}_entry_ind_fine_s"])
            r[f"{ib}_n_crossings_prodconv"] = b.get(f"{ib}_prod_n_crossings")
            r[f"{ib}_n_crossings_fine"] = b.get(f"{ib}_fine_n_crossings")
        for a in ("cw", "so"):
            r[f"outcome_{a}_prod"] = f"{r[a + '_arrived_prod']}/{r['ref_arrived_prod']}"
            r[f"outcome_{a}_ind"] = f"{r[a + '_arrived_ind_prodconv']}/{r['ref_arrived_ind_prodconv']}"
            r[f"outcome_{a}_ind_fine"] = f"{r[a + '_arrived_ind_fine']}/{r['ref_arrived_ind_fine']}"
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
        for a in ("cw", "so"):
            if r[f"{a}_status_prod"] == "failed" and r[f"{a}_status_ind"] == "numerical_failure":
                r[f"{a}_ref_peak_absdiff_m"] = float("nan")
                r[f"{a}_ref_peak_reldiff"] = float("nan")
                r[f"{a}_note"] = "numerical failure in both implementations; peak not compared"
                r[f"{a}_failure_time_diff_s"] = abs(r[f"{a}_t_end_prod_s"] - r[f"{a}_t_end_ind_s"])
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
        for fam, finest in (("radau", "radau_1e-12"), ("dop853", "dop853_1e-13"), ("rk4", "rk4_0.125")):
            if finest not in ind.index:
                continue
            fb = ind.loc[finest]
            for lab in ind.index:
                if not lab.startswith(fam):
                    continue
                x = ind.loc[lab]
                cr = {"sid": sid, "family": fam, "setting": lab, "finest": finest, "wall_s": f(x.get("wall_s"))}
                for a in ("cw", "so"):
                    cr[f"{a}_peak_diff_to_finest_m"] = abs(f(x.get(f"{a}_vs_ref_peak_pos_m")) - f(fb.get(f"{a}_vs_ref_peak_pos_m")))
                for ib in BR:
                    cr[f"{ib}_entry_diff_to_finest_s"] = abs(f(x.get(f"{ib}_prod_entry_time_s")) - f(fb.get(f"{ib}_prod_entry_time_s")))
                    cr[f"{ib}_final_range_diff_to_finest_m"] = abs(f(x.get(f"{ib}_final_range_m")) - f(fb.get(f"{ib}_final_range_m")))
                    cr[f"{ib}_status"] = x.get(f"{ib}_status")
                    cr[f"{ib}_arrived_prodconv"] = x.get(f"{ib}_prod_arrived")
                conv.append(cr)
        fins = [x for x in ("radau_1e-12", "dop853_1e-13", "rk4_0.125") if x in ind.index]
        for x in fins[1:]:
            a0, a1 = ind.loc[fins[0]], ind.loc[x]
            xr = {"sid": sid, "family": "cross", "setting": x, "finest": fins[0], "wall_s": f(a1.get("wall_s"))}
            for a in ("cw", "so"):
                xr[f"{a}_peak_diff_to_finest_m"] = abs(f(a1.get(f"{a}_vs_ref_peak_pos_m")) - f(a0.get(f"{a}_vs_ref_peak_pos_m")))
            for ib in BR:
                xr[f"{ib}_entry_diff_to_finest_s"] = abs(f(a1.get(f"{ib}_prod_entry_time_s")) - f(a0.get(f"{ib}_prod_entry_time_s")))
                xr[f"{ib}_final_range_diff_to_finest_m"] = abs(f(a1.get(f"{ib}_final_range_m")) - f(a0.get(f"{ib}_final_range_m")))
                xr[f"{ib}_status"] = a1.get(f"{ib}_status")
                xr[f"{ib}_arrived_prodconv"] = a1.get(f"{ib}_prod_arrived")
                xr[f"{ib}_label_agrees_with_finest"] = bool(a1.get(f"{ib}_prod_arrived")) == bool(a0.get(f"{ib}_prod_arrived"))
            conv.append(xr)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUTD, "comparison.csv"), index=False)
    pd.DataFrame(conv).to_csv(os.path.join(OUTD, "convergence.csv"), index=False)
    errs.to_csv(os.path.join(OUTD, "independenterrors.csv"), index=False)
    C.logrun("verifier", "python experiments/v2/verifrun.py compare", "completed", 0.0, f"{len(R)} cases compared, {len(errs)} independent errors")
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
    {"production": lambda: cmdprod(a.workers), "independent": lambda: cmdindep(a.workers),
     "compare": cmdcmp}[a.cmd]()


if __name__ == "__main__":
    main()
