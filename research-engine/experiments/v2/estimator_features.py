"""v2 estimator features (A: full D1, B: unsaturated-Jacobian D1, C: residual-only diagnostic).

For every scenario of a campaign the CW-only screening workflow is re-run (CW closed
loop with the production solver, CW diagnostics) and the three estimator variants are
computed from that CW run (cwvalid/estimate_v2.py). No nonlinear model is propagated.
The recomputed A is compared with the stored v1 value (screen_features.jsonl) to show
that A is the unchanged v1 estimator.

Campaigns (tags):
  dev:      main_grid/grid, main_grid/refine, main_grid/refine_bands, screening/dev
  retro:    screening/test, screening/stress, screening/band_test   (retrospective sets)
  fresh:    v2/raw/fresh_test                                         (new confirmatory set)
  families: v2/raw/boundary_families/diagnostic                       (new boundary challenge)

Output: results/v2/raw/estimator_features/<campaign with / replaced by __>.jsonl (resumable)
Usage:  python experiments/v2/estimator_features.py dev|retro|fresh|families|<campaign path> [--workers 7]
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_common as C  # noqa: E402

GROUPS = {
    "dev": ["main_grid/grid", "main_grid/refine", "main_grid/refine_bands", "screening/dev"],
    "retro": ["screening/test", "screening/stress", "screening/band_test"],
    "fresh": ["v2/raw/fresh_test"],
    "families": ["v2/raw/boundary_families/diagnostic"],
}
OUT = os.path.join(C.RAW, "estimator_features")
KEEP = ["sid", "sep_m", "x0", "y0", "z0", "vx0", "vy0", "vz0", "wn_ratio", "a_max", "horizon_s", "n_radps", "R_m",
        "cw_sat_total_s", "cw_sat_initial_s", "cw_sat_n_intervals", "cw_terminal_arrived", "cw_terminal_inside_at_end",
        "cw_terminal_dwell_margin_s", "cw_terminal_arrival_time_s", "cw_final_pos_m", "cw_final_speed_mps",
        "cw_q2_impulse_all_mps", "cw_q2_impulse_sat_mps", "cw_max_range_over_R", "cw_status", "cw_termination",
        "cw_n_sat_switch_events"]


def features_one(scn_d):
    from cwvalid.estimate_v2 import estimator_variants
    from cwvalid.experiment import RunConfig, evaluate, make_law, run_branches
    from cwvalid.scenario import Scenario
    t0 = time.perf_counter()
    scn = Scenario.from_dict(scn_d)
    try:
        cfg = RunConfig(models=("cw",), reference="cw", crosscheck=None)
        sims, _ = run_branches(scn, cfg)
        row, _ = evaluate(scn, cfg, sims)
        out = {k: row.get(k) for k in KEEP}
        out.update(estimator_variants(sims["cw"], make_law(scn), scn.n, scn.R, scn.horizon))
        out["feature_status"] = "ok"
    except Exception as exc:
        import traceback
        out = {"sid": scn.sid, "feature_status": "exception", "error": traceback.format_exc()[-2000:]}
        for k in ("d1", "d1u", "cres"):
            out[f"{k}_peak_m"] = None
            out[f"{k}_status"] = f"failed: {exc!r}"
    out["v2_wall_s"] = time.perf_counter() - t0
    return C.jsonable(out)


def run_campaign_features(campaign, workers):
    src = os.path.join(C.RESULTS, campaign)
    scns = [json.loads(l) for l in open(os.path.join(src, "scenarios.jsonl")) if l.strip()]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, campaign.replace("/", "__") + ".jsonl")
    have = {r["sid"] for r in C.read_jsonl(path)}
    todo = [s for s in scns if s["sid"] not in have]
    t0 = time.perf_counter()
    print(f"[v2 features] {campaign}: {len(scns)} scenarios, {len(todo)} to compute", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex, open(path, "a", encoding="utf-8") as fh:
            futs = {ex.submit(features_one, s): s["sid"] for s in todo}
            for k, f in enumerate(as_completed(futs)):
                try:
                    r = f.result()
                except Exception as exc:   # worker crash: record, never drop
                    r = {"sid": futs[f], "feature_status": "worker_exception", "error": repr(exc)}
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                if (k + 1) % 250 == 0:
                    print(f"[v2 features] {campaign}: {k + 1}/{len(todo)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    rows = C.read_jsonl(path)
    got = {r["sid"] for r in rows}
    missing = [s["sid"] for s in scns if s["sid"] not in got]
    # comparison of recomputed A with the stored v1 value
    stored = {r["sid"]: r for r in C.read_jsonl(os.path.join(src, "screen_features.jsonl"))}
    n_cmp = n_eq = 0
    worst = 0.0
    for r in rows:
        s = stored.get(r["sid"])
        if s is None or r.get("d1_peak_m") is None or s.get("d1_peak_m") is None:
            continue
        n_cmp += 1
        a, b = float(r["d1_peak_m"]), float(s["d1_peak_m"])
        if a == b:
            n_eq += 1
        worst = max(worst, abs(a - b) / max(abs(b), 1e-300))
    summ = {"campaign": campaign, "n_scenarios": len(scns), "n_rows": len(rows), "n_missing": len(missing),
            "n_exceptions": sum(1 for r in rows if r.get("feature_status") != "ok"),
            "n_nonfinite": {k: sum(1 for r in rows if r.get(f"{k}_peak_m") is None) for k in ("d1", "d1u", "cres")},
            "A_vs_stored_v1": {"n_compared": n_cmp, "n_identical": n_eq, "max_rel_diff": worst},
            "wall_s": time.perf_counter() - t0}
    C.write_json(os.path.join(OUT, campaign.replace("/", "__") + "_summary.json"), summ)
    C.log_run("estimator_features", f"python experiments/v2/estimator_features.py {campaign}",
              "completed" if not missing else "incomplete", summ["wall_s"], json.dumps(summ["n_nonfinite"]))
    print(json.dumps(summ), flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    for c in GROUPS.get(a.which, [a.which]):
        run_campaign_features(c, a.workers)


if __name__ == "__main__":
    main()
