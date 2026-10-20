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
from maingrid import CONFIG, writescen

from cwvalid.experiment import RunCfg, runscen
from cwvalid.runner import runcamp
from cwvalid.scenario import axisscen

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "results", "screening")
OUT = os.path.join(RES, "bandtest")
FROZEN = os.path.join(RES, "frozen", "frozen.json")
FAMILIES = [(ax, w, a) for ax in ("+R", "-R", "+V", "-V", "+H") for w in (5.0, 7.0) for a in (0.02, 0.07)]
TOL = 2e-4


def arrives(model, ax, sep, w, a):
    scn = axisscen("bt", ax, sep, w, a)
    row, _ = runscen(scn, RunCfg(models=(model,), reference=model, crosscheck=None))
    return bool(row[f"{model}_terminal_arrived"])


def captlimit(model, ax, w, a):
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
    c = captlimit("cw", ax, w, a)
    r = captlimit("eci", ax, w, a)
    return {"axis": ax, "wn_ratio": w, "a_max": a,
            "cw_lo": c[0] if c else np.nan, "cw_hi": c[1] if c else np.nan, "cw_changes": c[2] if c else 0,
            "ref_lo": r[0] if r else np.nan, "ref_hi": r[1] if r else np.nan, "ref_changes": r[2] if r else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    frozenhash = hashlib.sha256(open(FROZEN, "rb").read()).hexdigest()
    limpath = os.path.join(OUT, "bandlimits.csv")
    if os.path.exists(limpath):
        lim = pd.read_csv(limpath)
    else:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            lim = pd.DataFrame(list(ex.map(family, FAMILIES)))
        lim["offset_pct"] = 100 * (np.sqrt(lim.cw_lo * lim.cw_hi) / np.sqrt(lim.ref_lo * lim.ref_hi) - 1)
        lim.to_csv(limpath, index=False)
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
            scns.append(axisscen("bandtest", r.axis, float(f"{s:.9g}"), r.wn_ratio, r.a_max,
                                 extra_meta={"label": lab, "in_dev_domain": True, "nu": 0.0}))
    writescen(os.path.join(OUT, "scenarios.jsonl"), scns)
    json.dump({"set": "band_test", "n": len(scns), "families": FAMILIES, "tol": TOL, "frozen_sha256_at_run": frozenhash,
               "written": time.strftime("%Y-%m-%d %H:%M:%S"), "config": CONFIG,
               "note": "added after the first held-out evaluation (no outcome disagreements in test/stress); frozen rules unchanged"},
              open(os.path.join(OUT, "setdefinition.json"), "w"), indent=2)
    runcamp(OUT, scns, CONFIG, workers=a.workers, keephist=False)


if __name__ == "__main__":
    main()
