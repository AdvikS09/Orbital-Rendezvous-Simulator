"""Step 6 - targeted mechanism experiments.

  unrestricted   A. saturated vs unrestricted PD (same gains). Unrestricted is a
                 diagnostic, not a physically available actuator. 6 axes x 3
                 bandwidths x 40 separations (independent of a_max).
  energy         C. +/-radial offsets with first-order (ydot0 = -2 n x0) and exact
                 equal-energy initial along-track rate, all grid bandwidths/caps.
  horizon2       D. the full main grid repeated with a two-orbit horizon.
  common         common-command propagation for representative cases listed in
                 results/mechanisms/common_cases.json (written by the analysis
                 step) or a built-in default list.
(B. positive vs negative offsets uses the main grid directly.)

Usage (repository root):
  python experiments/ext/mechanisms.py {unrestricted|energy|horizon2|common} [--workers 7]
"""

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

from main_grid import AMAX, AXIS_LIST, CONFIG, SEPS, WN, write_scenarios  # noqa: E402

from cwvalid.runner import config_from_dict, run_campaign  # noqa: E402
from cwvalid.scenario import Scenario, axis_scenario  # noqa: E402


def unrestricted():
    scns = []
    for ax in AXIS_LIST:
        for wn in WN:
            for k, s in enumerate(SEPS):
                sc = axis_scenario("unres", ax, float(s), wn, None, extra_meta={"grid_index": k, "variant": "unrestricted"})
                scns.append(sc)
    return scns


def energy():
    scns = []
    for ax in ("+R", "-R"):
        for vel in ("em1", "em_exact"):
            for wn in WN:
                for am in AMAX:
                    for k, s in enumerate(SEPS):
                        scns.append(axis_scenario("energy", ax, float(s), wn, am, velocity=vel,
                                                  extra_meta={"grid_index": k}))
    return scns


def horizon2():
    scns = []
    for ax in AXIS_LIST:
        for wn in WN:
            for am in AMAX:
                for k, s in enumerate(SEPS):
                    scns.append(axis_scenario("h2", ax, float(s), wn, am, horizon_orbits=2.0,
                                              extra_meta={"grid_index": k}))
    return scns


DEFAULT_COMMON = [("+R", 1000.0), ("+R", 5500.0), ("-R", 5500.0), ("+V", 7700.0), ("+H", 10000.0),
                  ("+R", 12000.0), ("+R", 14251.0)]


def run_common(workers):
    from concurrent.futures import ProcessPoolExecutor
    path = os.path.join(RES, "common_cases.json")
    if os.path.exists(path):
        cases = [Scenario.from_dict(d) for d in json.load(open(path))]
    else:
        cases = [axis_scenario("common", ax, s, 6.0, 0.05) for ax, s in DEFAULT_COMMON]
    out = os.path.join(RES, "common_command")
    os.makedirs(os.path.join(out, "histories"), exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for row, hist_name in ex.map(_common_one, [(c.to_dict(), out) for c in cases]):
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out, "common_command_summary.csv"), index=False)
    print(pd.DataFrame(rows).T)


def _common_one(args):
    d, out = args
    from cwvalid.diagnose import common_command
    scn = Scenario.from_dict(d)
    cfg = config_from_dict(CONFIG)
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
        run_common(a.workers)
        return
    scns = {"unrestricted": unrestricted, "energy": energy, "horizon2": horizon2}[a.cmd]()
    d = os.path.join(RES, a.cmd)
    os.makedirs(d, exist_ok=True)
    write_scenarios(os.path.join(d, "scenarios.jsonl"), scns)
    run_campaign(d, scns, CONFIG, workers=a.workers, keep_hist=False)


if __name__ == "__main__":
    main()
