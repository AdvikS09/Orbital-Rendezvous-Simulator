"""Fine refinement of CW/reference outcome-disagreement bands found by the main refinement.

For every family whose refined brackets contain a terminal-outcome disagreement,
both edges of the disagreement band are bisected in log-separation until the
relative bracket width is <= 2e-4. Bands are reported as
[inner edge bracket, outer edge bracket] with the implied width range.

Output: results/main_grid/refine_bands/ (campaign) and
        results/tables/tab_disagreement_bands.{csv,tex}
Usage:  python experiments/ext/refine_bands.py [--workers 7] [--tol 2e-4]
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import report_utils as RU  # noqa: E402
from main_grid import CONFIG, write_scenarios  # noqa: E402

from cwvalid.runner import load_results, run_campaign  # noqa: E402
from cwvalid.scenario import axis_scenario  # noqa: E402

MG = os.path.join(RU.RESULTS, "main_grid")
OUT = os.path.join(MG, "refine_bands")
KEY = ["meta_axis", "wn_ratio", "a_max"]


def load_all():
    parts = [load_results(os.path.join(MG, "grid")), load_results(os.path.join(MG, "refine"))]
    if os.path.exists(os.path.join(OUT, "results.jsonl")):
        parts.append(load_results(OUT))
    d = pd.concat(parts, ignore_index=True)
    return d[d.meta_velocity == "zero"]


def edges(g):
    """Brackets where the 'disagree' indicator changes, per family (sorted by separation)."""
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
    ap.add_argument("--all-boundaries", action="store_true",
                    help="also bisect every CW and reference arrival boundary (exposes bands narrower than the main refinement)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    from cwvalid.scenario import Scenario
    sc_path = os.path.join(OUT, "scenarios.jsonl")
    scns = [Scenario.from_dict(__import__("json").loads(l)) for l in open(sc_path) if l.strip()] if os.path.exists(sc_path) else []
    for r in range(a.rounds):
        d = load_all()
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
                            new.append(axis_scenario("band", fam[0], float(f"{mid:.9g}"), float(fam[1]), float(fam[2]),
                                                     extra_meta={"band_round": r, "boundary": col}))
        for fam in fams.itertuples(index=False):
            g = d[(d.meta_axis == fam.meta_axis) & (d.wn_ratio == fam.wn_ratio) & (d.a_max == fam.a_max)]
            for lo, hi, _, _ in edges(g):
                if hi / lo - 1 > a.tol:
                    mid = math.exp(0.5 * (math.log(lo) + math.log(hi)))
                    new.append(axis_scenario("band", fam.meta_axis, float(f"{mid:.9g}"), float(fam.wn_ratio), float(fam.a_max),
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
        write_scenarios(os.path.join(OUT, "scenarios.jsonl"), scns)
        run_campaign(OUT, scns, CONFIG, workers=a.workers, keep_hist=False)
    d = load_all()
    rows = []
    for fam, g in d.groupby(KEY):
        e = edges(g)
        if not e:
            continue
        dis_rows = g[g.outcome_terminal_cw.isin(["cw_only", "nl_only"])]
        kind = ";".join(sorted(dis_rows.outcome_terminal_cw.unique()))
        lo_in = [x for x in e if x[3]]      # entering the band
        hi_out = [x for x in e if x[2]]     # leaving the band
        for (a0, a1, _, _), (b0, b1, _, _) in zip(lo_in, hi_out):
            rows.append({"direction": RU.AXIS_LABEL[fam[0]], "wn/n": fam[1], "a_max": fam[2], "class": kind,
                         "band start bracket [m]": f"{a0:.1f}-{a1:.1f}", "band end bracket [m]": f"{b0:.1f}-{b1:.1f}",
                         "width min [%]": 100 * (b0 / a1 - 1), "width max [%]": 100 * (b1 / a0 - 1),
                         "points in band": int(((dis_rows.sep_m >= a1) & (dis_rows.sep_m <= b0)).sum())})
    t = pd.DataFrame(rows)
    RU.write_table(t, "tab_disagreement_bands", "Separation bands in which the CW and inertial nonlinear terminal-"
                   "arrival labels differ (main-grid families, zero initial velocity, one orbit), with both band "
                   f"edges bracketed to a relative width of {a.tol:g}. cw_only: CW arrives and the reference does "
                   "not; nl_only: the reverse.", "tab:bands",
                   fmt={"width min [%]": "{:.3f}", "width max [%]": "{:.3f}"})
    print(t.to_string())


if __name__ == "__main__":
    main()
