import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(os.path.dirname(ROOT), "results", "mechanisms")

from maingrid import AMAX, AXISLIST, CONFIG, SEPS, WN, writescen

from cwvalid.runner import configdict, runcamp
from cwvalid.scenario import Scenario, axisscen


def unrestricted():
    scns = []
    for ax in AXISLIST:
        for wn in WN:
            for k, s in enumerate(SEPS):
                sc = axisscen("unres", ax, float(s), wn, None, extra_meta={"grid_index": k, "variant": "unrestricted"})
                scns.append(sc)
    return scns


def energy():
    scns = []
    for ax in ("+R", "-R"):
        for vel in ("em1", "em_exact"):
            for wn in WN:
                for am in AMAX:
                    for k, s in enumerate(SEPS):
                        scns.append(axisscen("energy", ax, float(s), wn, am, velocity=vel,
                                             extra_meta={"grid_index": k}))
    return scns


def horizon2():
    scns = []
    for ax in AXISLIST:
        for wn in WN:
            for am in AMAX:
                for k, s in enumerate(SEPS):
                    scns.append(axisscen("h2", ax, float(s), wn, am, horizon_orbits=2.0,
                                         extra_meta={"grid_index": k}))
    return scns


DEFCOMMON = [("+R", 1000.0), ("+R", 5500.0), ("-R", 5500.0), ("+V", 7700.0), ("+H", 10000.0),
             ("+R", 12000.0), ("+R", 14251.0)]


def runcommon(workers):
    from concurrent.futures import ProcessPoolExecutor
    path = os.path.join(RES, "commoncases.json")
    if os.path.exists(path):
        cas = [Scenario.from_dict(d) for d in json.load(open(path))]
    else:
        cas = [axisscen("common", ax, s, 6.0, 0.05) for ax, s in DEFCOMMON]
    out = os.path.join(RES, "commoncommand")
    os.makedirs(os.path.join(out, "histories"), exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for row, histname in ex.map(commonone, [(c.to_dict(), out) for c in cas]):
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out, "commandsummary.csv"), index=False)
    print(pd.DataFrame(rows).T)


def commonone(args):
    d, out = args
    from cwvalid.diagnose import common_command
    scn = Scenario.from_dict(d)
    cfg = configdict(CONFIG)
    row, hist = common_command(scn, cfg)
    np.savez_compressed(os.path.join(out, "histories", f"{scn.sid}.npz"), **hist)
    row["history_file"] = f"histories/{scn.sid}.npz"
    return row, row["history_file"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["unrestricted", "energy", "horizon2", "common"])
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    if a.cmd == "common":
        runcommon(a.workers)
        return
    scns = {"unrestricted": unrestricted, "energy": energy, "horizon2": horizon2}[a.cmd]()
    d = os.path.join(RES, a.cmd)
    os.makedirs(d, exist_ok=True)
    writescen(os.path.join(d, "scenarios.jsonl"), scns)
    runcamp(d, scns, CONFIG, workers=a.workers, keephist=False)


if __name__ == "__main__":
    main()
