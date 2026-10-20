import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import reportutil as RU
from maingrid import CONFIG, writescen

from cwvalid.runner import loadres, runcamp
from cwvalid.scenario import axisscen

MG = os.path.join(RU.RESULTS, "maingrid")
OUT = os.path.join(MG, "refinebands")
KEY = ["meta_axis", "wn_ratio", "a_max"]


def loadall():
    parts = [loadres(os.path.join(MG, "grid")), loadres(os.path.join(MG, "refine"))]
    if os.path.exists(os.path.join(OUT, "results.jsonl")):
        parts.append(loadres(OUT))
    d = pd.concat(parts, ignore_index=True)
    return d[d.meta_velocity == "zero"]


def edges(g):
    g = g.sort_values("sep_m").drop_duplicates("sep_m")
    dis = g.outcome_terminal_cw.isin(["cw_only", "nl_only"]).to_numpy()
    s = g.sep_m.to_numpy()
    out = []
    for i in np.where(dis[1:] != dis[:-1])[0]:
        out.append((s[i], s[i + 1], bool(dis[i]), bool(dis[i + 1])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--tol", type=float, default=2e-4)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--all-boundaries", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    from cwvalid.scenario import Scenario
    scpath = os.path.join(OUT, "scenarios.jsonl")
    scns = [Scenario.from_dict(__import__("json").loads(l)) for l in open(scpath) if l.strip()] if os.path.exists(scpath) else []
    for r in range(a.rounds):
        d = loadall()
        fams = d[d.outcome_terminal_cw.isin(["cw_only", "nl_only"])][KEY].drop_duplicates()
        new = []
        if a.all_boundaries:
            for fam, g in d.groupby(KEY):
                g = g.sort_values("sep_m").drop_duplicates("sep_m")
                sp = g.sep_m.to_numpy()
                for col in ("cw_terminal_arrived", f"{CONFIG['reference']}_terminal_arrived"):
                    v = g[col].astype(bool).to_numpy()
                    for i in np.where(v[1:] != v[:-1])[0]:
                        lo, hi = sp[i], sp[i + 1]
                        if hi / lo - 1 > a.tol:
                            mid = math.exp(0.5 * (math.log(lo) + math.log(hi)))
                            new.append(axisscen("band", fam[0], float(f"{mid:.9g}"), float(fam[1]), float(fam[2]),
                                                extra_meta={"band_round": r, "boundary": col}))
        for fam in fams.itertuples(index=False):
            g = d[(d.meta_axis == fam.meta_axis) & (d.wn_ratio == fam.wn_ratio) & (d.a_max == fam.a_max)]
            for lo, hi, _, _ in edges(g):
                if hi / lo - 1 > a.tol:
                    mid = math.exp(0.5 * (math.log(lo) + math.log(hi)))
                    new.append(axisscen("band", fam.meta_axis, float(f"{mid:.9g}"), float(fam.wn_ratio), float(fam.a_max),
                                        extra_meta={"band_round": r}))
        have = {s.sid for s in scns}
        uniq = {}
        for x in new:
            if x.sid not in have:
                uniq[x.sid] = x
        new = list(uniq.values())
        print(f"[bands] round {r}: {len(fams)} families, {len(new)} new points", flush=True)
        if not new:
            break
        scns += new
        writescen(os.path.join(OUT, "scenarios.jsonl"), scns)
        runcamp(OUT, scns, CONFIG, workers=a.workers, keephist=False)
    d = loadall()
    rows = []
    for fam, g in d.groupby(KEY):
        e = edges(g)
        if not e:
            continue
        disrows = g[g.outcome_terminal_cw.isin(["cw_only", "nl_only"])]
        kind = ";".join(sorted(disrows.outcome_terminal_cw.unique()))
        loin = [x for x in e if x[3]]      
        hiout = [x for x in e if x[2]]
        for (a0, a1, _, _), (b0, b1, _, _) in zip(loin, hiout):
            rows.append({"direction": RU.AXISLABEL[fam[0]], "wn/n": fam[1], "a_max": fam[2], "class": kind,
                         "band start bracket [m]": f"{a0:.1f}-{a1:.1f}", "band end bracket [m]": f"{b0:.1f}-{b1:.1f}",
                         "width min [%]": 100 * (b0 / a1 - 1), "width max [%]": 100 * (b1 / a0 - 1),
                         "points in band": int(((disrows.sep_m >= a1) & (disrows.sep_m <= b0)).sum())})
    t = pd.DataFrame(rows)
    RU.writetable(t, "disagreementbands", "Separation bands in which the CW and inertial nonlinear terminal-"
                  "arrival labels differ (main-grid families, zero initial velocity, one orbit), with both band "
                  f"edges bracketed to a relative width of {a.tol:g}. cw_only: CW arrives and the reference does "
                  "not; nl_only: the reverse.", "tab:bands",
                  fmt={"width min [%]": "{:.3f}", "width max [%]": "{:.3f}"})
    print(t.to_string())


if __name__ == "__main__":
    main()
