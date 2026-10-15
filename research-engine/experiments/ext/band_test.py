"""Held-out band test: does the FROZEN screen catch CW outcome errors in new configurations?

Added after the first held-out evaluation showed that neither the broad test set nor
the stress set contained a single CW/reference outcome disagreement (they occur only in
bands <= 0.4% wide). The frozen rules are NOT changed by this script.

Families not used in fitting: axes {+R, -R, +V, -V, +H}, omega_n/n in {5, 7},
a_max in {0.02, 0.07} m/s^2 (development used {4, 6, 8} x {0.01, 0.05, 0.10} on the
grid and continuous random values in the 200-case dev set). For each family the CW
and reference capture limits (largest separation with a terminal arrival, zero initial
velocity, one orbit) are located by a 25-point log sweep over 1-60 km and bisection to
0.02%, and evaluation points are placed at 1/4, 1/2 and 3/4 of the band between the two
limits (log scale) and 0.5% outside each end. Locating the bands uses the reference
model, which is legitimate for building an evaluation set; the screen itself still uses
only CW-only inputs.

Outputs: results/screening/band_test/ (campaign + band_limits.csv)
Usage: python experiments/ext/band_test.py [--workers 7]
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from main_grid import CONFIG, write_scenarios  # noqa: E402

from cwvalid.experiment import RunConfig, run_scenario  # noqa: E402
from cwvalid.runner import run_campaign  # noqa: E402
from cwvalid.scenario import axis_scenario  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "results", "screening")
OUT = os.path.join(RES, "band_test")
FROZEN = os.path.join(RES, "frozen", "screening_frozen.json")
FAMILIES = [(ax, w, a) for ax in ("+R", "-R", "+V", "-V", "+H") for w in (5.0, 7.0) for a in (0.02, 0.07)]
TOL = 2e-4


def arrives(model, ax, sep, w, a):
    scn = axis_scenario("bt", ax, sep, w, a)
    row, _ = run_scenario(scn, RunConfig(models=(model,), reference=model, crosscheck=None))
    return bool(row[f"{model}_terminal_arrived"])


def capture_limit(model, ax, w, a):
    seps = np.geomspace(1000.0, 60000.0, 25)
    v = [arrives(model, ax, s, w, a) for s in seps]
    idx = [i for i in range(len(v) - 1) if v[i] and not v[i + 1]]
    if not idx:
        return None
    lo, hi = seps[idx[0]], seps[idx[0] + 1]
    while hi / lo - 1 > TOL:
        mid = math.exp(0.5 * (math.log(lo) + math.log(hi)))
        if arrives(model, ax, mid, w, a):
            lo = mid
        else:
            hi = mid
    return lo, hi, len(idx)


def family(args):
    ax, w, a = args
    c = capture_limit("cw", ax, w, a)
    r = capture_limit("eci", ax, w, a)
    return {"axis": ax, "wn_ratio": w, "a_max": a,
            "cw_lo": c[0] if c else np.nan, "cw_hi": c[1] if c else np.nan, "cw_changes": c[2] if c else 0,
            "ref_lo": r[0] if r else np.nan, "ref_hi": r[1] if r else np.nan, "ref_changes": r[2] if r else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    frozen_hash = hashlib.sha256(open(FROZEN, "rb").read()).hexdigest()
    lim_path = os.path.join(OUT, "band_limits.csv")
    if os.path.exists(lim_path):
        lim = pd.read_csv(lim_path)
    else:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            lim = pd.DataFrame(list(ex.map(family, FAMILIES)))
        lim["offset_pct"] = 100 * (np.sqrt(lim.cw_lo * lim.cw_hi) / np.sqrt(lim.ref_lo * lim.ref_hi) - 1)
        lim.to_csv(lim_path, index=False)
        print(f"limits found in {time.perf_counter() - t0:.0f}s")
    print(lim.to_string())
    scns = []
    for r in lim.itertuples():
        if not (np.isfinite(r.cw_lo) and np.isfinite(r.ref_lo)):
            continue
        c, ref = math.sqrt(r.cw_lo * r.cw_hi), math.sqrt(r.ref_lo * r.ref_hi)
        lo, hi = min(c, ref), max(c, ref)
        kind = "cw_only band" if c > ref else ("nl_only band" if c < ref else "no band")
        pts = [(lo * 0.995, "below band")]
        if hi / lo - 1 > 2 * TOL:
            pts += [(math.exp(math.log(lo) + f * (math.log(hi) - math.log(lo))), f"inside {kind}") for f in (0.25, 0.5, 0.75)]
        pts += [(hi * 1.005, "above band")]
        for s, lab in pts:
            scns.append(axis_scenario("bandtest", r.axis, float(f"{s:.9g}"), r.wn_ratio, r.a_max,
                                      extra_meta={"label": lab, "in_dev_domain": True, "nu": 0.0}))
    write_scenarios(os.path.join(OUT, "scenarios.jsonl"), scns)
    json.dump({"set": "band_test", "n": len(scns), "families": FAMILIES, "tol": TOL, "frozen_sha256_at_run": frozen_hash,
               "written": time.strftime("%Y-%m-%d %H:%M:%S"), "config": CONFIG,
               "note": "added after the first held-out evaluation (no outcome disagreements in test/stress); frozen rules unchanged"},
              open(os.path.join(OUT, "set_definition.json"), "w"), indent=2)
    run_campaign(OUT, scns, CONFIG, workers=a.workers, keep_hist=False)


if __name__ == "__main__":
    main()
