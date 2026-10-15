"""P07 regression: the matrix-law continuous engine with a PD matrix reproduces the production
scalar-PD pipeline (cwvalid.simulate + evaluate + estimate_v2) on archived fresh-test scenarios.

Usage: python code/experiments/v3/verify_matrix_engine.py [--n 12]
Output: v3_extension/results/matrix_engine_verification.csv
"""
import argparse, json, math, os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, REPO)
ARCH = os.path.dirname(REPO); OUT = os.path.join(ARCH, "v3_extension", "results")


def one(scn_d):
    from cwvalid.scenario import Scenario
    from cwvalid.experiment import RunConfig, run_branches, evaluate, make_law
    from cwvalid.estimate_v2 import estimator_variants
    from cwvalid3.continuous import run_branches_general, evaluate_general, estimator_variants_general
    from cwvalid3.controllers import pd_law
    s = Scenario.from_dict(scn_d)
    cfg = RunConfig(models=("cw", "eci", "nerm"), reference="eci", crosscheck="nerm")
    t0 = time.perf_counter(); sims, _ = run_branches(s, cfg); rp, _ = evaluate(s, cfg, sims)
    fp = estimator_variants(sims["cw"], make_law(s), s.n, s.R, s.horizon); tp = time.perf_counter() - t0
    law = pd_law(s.wn_ratio * s.n, s.a_max)
    t0 = time.perf_counter(); sg, _ = run_branches_general(s, cfg, law); rg, _ = evaluate_general(s, cfg, sg, law)
    fg = estimator_variants_general(sg["cw"], law, s.n, s.R, s.horizon); tg = time.perf_counter() - t0
    out = {"sid": s.sid, "sep_m": s.separation, "wall_production_s": tp, "wall_matrix_s": tg,
           "n_sat_switch_cw": len(sims["cw"].sat_switch_times)}
    worst, worst_abs, n_cmp, n_bitident, diff = 0.0, 0.0, 0, 0, []
    for k, v in rp.items():
        g = rg.get(k)
        if "wall" in k:   # timings are not physics
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None and isinstance(g, (int, float)):
            if math.isfinite(v) and math.isfinite(g):
                n_cmp += 1; n_bitident += (v == g)
                if v != g:
                    diff.append(k); worst = max(worst, abs(v - g) / max(abs(v), 1e-300)); worst_abs = max(worst_abs, abs(v - g))
    out.update(row_fields_compared=n_cmp, row_fields_bit_identical=n_bitident, row_max_rel_diff=worst,
               row_max_abs_diff=worst_abs, row_fields_not_identical=";".join(diff))
    for k in ("d1", "d1u", "cres"):
        out[f"{k}_prod_m"], out[f"{k}_matrix_m"] = fp[f"{k}_peak_m"], fg[f"{k}_peak_m"]
        out[f"{k}_abs_diff_m"] = abs(fp[f"{k}_peak_m"] - fg[f"{k}_peak_m"])
    out["kp_kd_fields_equal"] = (rp["kp"] == rg["kp"]) and (rp["kd"] == rg["kd"])
    for b in ("cw", "eci", "nerm"):
        out[f"{b}_status_equal"] = rp[f"{b}_status"] == rg[f"{b}_status"]
    out["disc_prod_m"], out["disc_matrix_m"] = rp["cw_vs_eci_max_pos_disc_m"], rg["cw_vs_eci_max_pos_disc_m"]
    return out


def main():
    import v30_closed  # v3.1: preserved v3.0 outputs; runs only inside the rebuild harness
    v30_closed.refuse_outside_harness(__file__)
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=12); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    scns = [json.loads(l) for l in open(os.path.join(ARCH, "results/v2/raw/fresh_test/scenarios.jsonl")) if l.strip()]
    # deterministic selection: every 25th archived fresh case (spread over the distribution)
    sel = scns[::max(1, len(scns) // a.n)][:a.n]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, sel))
    df = pd.DataFrame(rows); os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "matrix_engine_verification.csv"), index=False)
    print(df[["sid", "n_sat_switch_cw", "row_fields_compared", "row_fields_bit_identical", "row_max_rel_diff", "row_max_abs_diff",
              "d1_abs_diff_m", "d1u_abs_diff_m", "cres_abs_diff_m", "wall_production_s", "wall_matrix_s"]].to_string())


if __name__ == "__main__":
    main()
