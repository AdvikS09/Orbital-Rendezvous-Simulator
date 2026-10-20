import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2common as C

from cwvalid.runner import loadres

OUT = os.path.join(C.DERIVED, "accounting")
RAWF = os.path.join(C.RAW, "sofailures")
MODELS = ("cw", "so", "nerm", "eci")


def census(d, m):
    st = d[f"{m}_status"].astype(str)
    comp = (st == "ok") & d.get(f"{m}_completed_horizon", pd.Series(True, index=d.index)).fillna(False).astype(bool)
    arr = d[f"{m}_terminal_arrived"].fillna(False).astype(bool)
    out = {"completed_arrival": int((comp & arr).sum()), "completed_no_arrival": int((comp & ~arr).sum()),
           "domain_termination": int((st == "terminated").sum()), "numerical_failure": int((st == "failed").sum())}
    out["other"] = len(d) - sum(out.values())
    return out


def statuscensus():
    rows = []
    paths = sorted(glob.glob(os.path.join(C.RESULTS, "**", "results.jsonl"), recursive=True))
    for p in paths:
        camp = os.path.relpath(os.path.dirname(p), C.RESULTS).replace("\\", "/")
        if "/search" in camp:
            continue
        d = loadres(os.path.dirname(p))
        for m in MODELS:
            if f"{m}_status" not in d:
                continue
            c = census(d, m)
            rows.append({"campaign": camp, "model": m, "n_scenarios": len(d), **c,
                         "terminations": json.dumps(d[f"{m}_termination"].value_counts().to_dict())})
    for p in sorted(glob.glob(os.path.join(C.RAW, "families", "search", "*.jsonl"))):
        d = pd.DataFrame(C.readjsonl(p))
        for m in MODELS:
            if f"{m}_status" not in d:
                continue
            dm = d[d[f"{m}_status"].notna()]
            c = census(dm, m)
            rows.append({"campaign": "v2/raw/families/search/" + os.path.basename(p), "model": m,
                         "n_scenarios": len(dm), **c, "terminations": json.dumps(dm[f"{m}_termination"].value_counts().to_dict())})
    t = pd.DataFrame(rows)
    if (t["other"] != 0).any():
        raise SystemExit("status categories do not add up for: " + ", ".join(t[t.other != 0].campaign))
    return t


def agr(d, m, ref="eci"):
    sm, sr = d[f"{m}_status"].astype(str), d[f"{ref}_status"].astype(str)
    comp = (sm != "failed") & (sr != "failed")
    am = d[f"{m}_terminal_arrived"].fillna(False).astype(bool)
    ar = d[f"{ref}_terminal_arrived"].fillna(False).astype(bool)
    c = comp
    return {"N attempted": len(d), "model numerical failures": int((sm == "failed").sum()),
            "reference numerical failures": int((sr == "failed").sum()),
            "reference domain terminations": int((sr == "terminated").sum()),
            "comparable": int(c.sum()), "label agrees": int((c & (am == ar)).sum()),
            "model arrives, reference not": int((c & am & ~ar).sum()),
            "reference arrives, model not": int((c & ~am & ar).sum()),
            "both arrive": int((c & am & ar).sum()), "neither arrives": int((c & ~am & ~ar).sum())}


def fixedoutctable():
    s = {"one orbit, grid + refinement + band points": ["maingrid/grid", "maingrid/refine", "maingrid/refinebands"],
         "two orbits, main grid": ["mechanisms/horizon2"]}
    rows = []
    for name, cs in s.items():
        d = pd.concat([loadres(os.path.join(C.RESULTS, c)) for c in cs], ignore_index=True)
        d = d.drop_duplicates(["meta_axis", "wn_ratio", "a_max", "sep_m", "horizon_orbits"])
        for m, lab in (("cw", "CW"), ("so", "second order"), ("nerm", "NERM (exact, cross-check)")):
            rows.append({"data": name, "model": lab, **agr(d, m)})
    return pd.DataFrame(rows)


def invfail():
    from cwvalid.experiment import makecmd, RunCfg
    from cwvalid.scenario import Scenario
    from cwvalid.simulate import simulate
    h2 = loadres(os.path.join(C.RESULTS, "mechanisms", "horizon2"))
    f = h2[h2.so_status == "failed"].copy()
    scns = {json.loads(l)["sid"]: json.loads(l) for l in open(os.path.join(C.RESULTS, "mechanisms", "horizon2", "scenarios.jsonl")) if l.strip()}
    picks = []
    for sid in (f.sort_values("sep_m").sid.iloc[0], f.sort_values("sep_m").sid.iloc[-1],
                f.sort_values("so_t_final_s").sid.iloc[0], f.sort_values("so_t_final_s").sid.iloc[-1]):
        if sid not in picks:
            picks.append(sid)
    for ax in sorted(f.meta_axis.unique()):
        sid = f[f.meta_axis == ax].sort_values("sep_m").sid.iloc[0]
        if sid not in picks:
            picks.append(sid)
    os.makedirs(RAWF, exist_ok=True)
    rows = []
    for sid in picks:
        scn = Scenario.from_dict(scns[sid])
        sim = simulate("so", scn.rel0, makecmd(scn, RunCfg()), scn.horizon, n=scn.n, R=scn.R, solver=RunCfg().solver)
        ts = np.concatenate([np.asarray(s.sol.ts) for s in sim.segments])
        ts = np.unique(ts[ts <= sim.tfinal])
        S = sim.states(ts)
        rng = np.linalg.norm(S[:, :3], axis=1)
        steps = np.diff(ts)
        tstar = sim.tfinal
        sel = (rng >= scn.R) & (rng <= 1e4 * scn.R) & (tstar - ts > 0)
        p = np.polyfit(np.log(tstar - ts[sel]), np.log(rng[sel]), 1)[0] if sel.sum() > 5 else np.nan
        t01r = float(ts[np.argmax(rng > 0.1 * scn.R)]) if (rng > 0.1 * scn.R).any() else np.nan
        t1r = float(ts[np.argmax(rng > scn.R)]) if (rng > scn.R).any() else np.nan
        r = h2[h2.sid == sid].iloc[0]
        np.savez_compressed(os.path.join(RAWF, f"{sid}.npz"), t=ts, rho=S[:, :3], rhodot=S[:, 3:], range_m=rng)
        rows.append({"sid": sid, "axis": r.meta_axis, "sep_m": r.sep_m, "wn_ratio": r.wn_ratio, "a_max": r.a_max,
                     "stored_so_t_final_s": r.so_t_final_s, "rerun_so_t_final_s": sim.tfinal, "rerun_status": sim.status,
                     "rerun_message": sim.message[:120], "last_step_s": float(steps[-1]) if steps.size else np.nan,
                     "min_step_last_10_s": float(steps[-10:].min()) if steps.size else np.nan,
                     "range_at_failure_m": float(rng[-1]), "range_over_R_at_failure": float(rng[-1] / scn.R),
                     "range_100s_before_m": float(np.interp(tstar - 100.0, ts, rng)),
                     "growth_exponent_p": float(p), "n_fit_points": int(sel.sum()), "n_steps": int(ts.size),
                     "t_range_exceeds_0.1R_s": t01r, "t_range_exceeds_R_s": t1r,
                     "reference_status": r.eci_status, "reference_termination": r.eci_termination,
                     "reference_arrived": bool(r.eci_terminal_arrived), "reference_max_range_over_R": r.eci_max_range_over_R,
                     "cw_arrived": bool(r.cw_terminal_arrived)})
    return pd.DataFrame(rows), f


def main():
    t0 = time.perf_counter()
    os.makedirs(OUT, exist_ok=True)
    cen = statuscensus()
    cen.to_csv(os.path.join(OUT, "census.csv"), index=False)
    cor = fixedoutctable()
    cor.to_csv(os.path.join(OUT, "outcomesstatus.csv"), index=False)
    old = pd.read_csv(os.path.join(C.RESULTS, "tables", "outcomesrefined.csv"))
    old.insert(0, "status", "SUPERSEDED (v1): numerical failures counted as agreeing non-arrivals")
    old.to_csv(os.path.join(OUT, "outcomesv1.csv"), index=False)
    inv, f = invfail()
    inv.to_csv(os.path.join(OUT, "sofailures.csv"), index=False)
    fl = f[["sid", "meta_axis", "sep_m", "wn_ratio", "a_max", "so_t_final_s", "so_message", "eci_status", "eci_termination",
            "eci_terminal_arrived", "eci_max_range_over_R", "cw_terminal_arrived", "nerm_status"]]
    fl.to_csv(os.path.join(OUT, "twoorbit.csv"), index=False)
    C.logrun("failure_accounting", "python experiments/v2/failacct.py", "completed", time.perf_counter() - t0)
    pd.set_option("display.width", 250)
    print(cor.to_string())
    print(inv.to_string())
    print(cen[cen.numerical_failure > 0].to_string())


if __name__ == "__main__":
    main()
