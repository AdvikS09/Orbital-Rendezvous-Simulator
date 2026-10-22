import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2common as C

GROUPS = {
    "dev": ["maingrid/grid", "maingrid/refine", "maingrid/refinebands", "screening/dev"],
    "retro": ["screening/test", "screening/stress", "screening/bandtest"],
    "fresh": ["v2/raw/freshtest"],
    "families": ["v2/raw/families/diagnostic"],
}
OUT = os.path.join(C.RAW, "features")
KEEP = ["sid", "sep_m", "x0", "y0", "z0", "vx0", "vy0", "vz0", "wn_ratio", "a_max", "horizon_s", "n_radps", "R_m",
        "cw_sat_total_s", "cw_sat_initial_s", "cw_sat_n_intervals", "cw_terminal_arrived", "cw_terminal_inside_at_end",
        "cw_terminal_dwell_margin_s", "cw_terminal_arrival_time_s", "cw_final_pos_m", "cw_final_speed_mps",
        "cw_q2_impulse_all_mps", "cw_q2_impulse_sat_mps", "cw_max_range_over_R", "cw_status", "cw_termination",
        "cw_n_sat_switch_events"]


def featsone(scnd):
    from cwvalid.estimate2 import estvar
    from cwvalid.experiment import RunCfg, evaluate, makelaw, runbranch
    from cwvalid.scenario import Scenario
    t0 = time.perf_counter()
    scn = Scenario.from_dict(scnd)
    try:
        cfg = RunCfg(models=("cw",), reference="cw", crosscheck=None)
        sims, _ = runbranch(scn, cfg)
        row, _ = evaluate(scn, cfg, sims)
        out = {k: row.get(k) for k in KEEP}
        out.update(estvar(sims["cw"], makelaw(scn), scn.n, scn.R, scn.horizon))
        out["feature_status"] = "ok"
    except Exception as exc:
        import traceback
        out = {"sid": scn.sid, "feature_status": "exception", "error": traceback.format_exc()[-2000:]}
        for k in ("d1", "d1u", "cres"):
            out[f"{k}_peak_m"] = None
            out[f"{k}_status"] = f"failed: {exc!r}"
    out["v2_wall_s"] = time.perf_counter() - t0
    return C.jsonable(out)


def runcampfeats(campaign, workers):
    src2 = os.path.join(C.RESULTS, campaign)
    scns = [json.loads(l) for l in open(os.path.join(src2, "scenarios.jsonl")) if l.strip()]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, os.path.basename(campaign) + ".jsonl")
    have = {r["sid"] for r in C.readjsonl(path)}
    todo = [s for s in scns if s["sid"] not in have]
    t0 = time.perf_counter()
    print(f"[v2 features] {campaign}: {len(scns)} scenarios, {len(todo)} to compute", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex, open(path, "a", encoding="utf-8") as fh:
            futs = {ex.submit(featsone, s): s["sid"] for s in todo}
            for k, f in enumerate(as_completed(futs)):
                try:
                    r = f.result()
                except Exception as exc:
                    r = {"sid": futs[f], "feature_status": "worker_exception", "error": repr(exc)}
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                if (k + 1) % 250 == 0:
                    print(f"[v2 features] {campaign}: {k + 1}/{len(todo)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    rows = C.readjsonl(path)
    got = {r["sid"] for r in rows}
    missing = [s["sid"] for s in scns if s["sid"] not in got]
    stored = {r["sid"]: r for r in C.readjsonl(os.path.join(src2, "screenfeatures.jsonl"))}
    ncmp = neq = 0
    worst = 0.0
    for r in rows:
        s = stored.get(r["sid"])
        if s is None or r.get("d1_peak_m") is None or s.get("d1_peak_m") is None:
            continue
        ncmp += 1
        a, b = float(r["d1_peak_m"]), float(s["d1_peak_m"])
        if a == b:
            neq += 1
        worst = max(worst, abs(a - b) / max(abs(b), 1e-300))
    summ = {"campaign": campaign, "n_scenarios": len(scns), "n_rows": len(rows), "n_missing": len(missing),
            "n_exceptions": sum(1 for r in rows if r.get("feature_status") != "ok"),
            "n_nonfinite": {k: sum(1 for r in rows if r.get(f"{k}_peak_m") is None) for k in ("d1", "d1u", "cres")},
            "A_vs_stored_v1": {"n_compared": ncmp, "n_identical": neq, "max_rel_diff": worst},
            "wall_s": time.perf_counter() - t0}
    C.writejson(os.path.join(OUT, os.path.basename(campaign) + "summary.json"), summ)
    C.logrun("estimator_features", f"python experiments/v2/estfeat.py {campaign}",
             "completed" if not missing else "incomplete", summ["wall_s"], json.dumps(summ["n_nonfinite"]))
    print(json.dumps(summ), flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    for c in GROUPS.get(a.which, [a.which]):
        runcampfeats(c, a.workers)


if __name__ == "__main__":
    main()
