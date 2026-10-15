"""Save (or regenerate) full time histories from saved scenario configurations.

Each output file results/histories/<sid>.npz holds, for the CW, inertial
reference (eci), NERM and second-order (so) branches: t, LVLH state S (N x 6),
raw and applied command (N x 3) and cumulative Delta-v, at a 1 s spacing (the
0.5 s metric grid decimated by 2, plus the branch end), together with the
scenario definition and the run configuration as JSON strings. Metrics in the
campaign result files were computed at full resolution; these files are for
plotting and inspection.

Usage (repository root):
  python experiments/ext/make_histories.py --preset handover      # representative, boundary, disagreement
  python experiments/ext/make_histories.py --sid grid-925b9dfc512a  # any scenario ID from any campaign
"""

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import report_utils as RU  # noqa: E402
from main_grid import CONFIG  # noqa: E402

from cwvalid.experiment import run_scenario  # noqa: E402
from cwvalid.runner import config_from_dict, load_results  # noqa: E402
from cwvalid.scenario import Scenario  # noqa: E402

OUT = os.path.join(RU.RESULTS, "histories")


def all_scenarios():
    """sid -> scenario dict from every campaign's scenarios.jsonl."""
    out = {}
    for p in glob.glob(os.path.join(RU.RESULTS, "**", "scenarios.jsonl"), recursive=True):
        for line in open(p):
            if line.strip():
                d = json.loads(line)
                out[d["sid"]] = d
    return out


def one(args):
    scn_d, stride = args
    scn = Scenario.from_dict(scn_d)
    cfg = config_from_dict(CONFIG)
    row, hist = run_scenario(scn, cfg, keep_histories=True, hist_stride=stride)
    hist["scenario_json"] = np.array(json.dumps(scn.to_dict()))
    hist["config_json"] = np.array(json.dumps(CONFIG))
    np.savez_compressed(os.path.join(OUT, f"{scn.sid}.npz"), **hist)
    return {"sid": scn.sid, "family": scn.family, **{f"meta_{k}": v for k, v in scn.meta.items()},
            "wn_ratio": scn.wn_ratio, "a_max": scn.a_max, "sep_m": scn.separation,
            "outcome_terminal_cw": row["outcome_terminal_cw"], "peak_disc_m": row["cw_vs_eci_max_pos_disc_m"],
            "file": f"histories/{scn.sid}.npz"}


def handover_selection():
    sel = []
    for f in (os.path.join(RU.RESULTS, "mechanisms", "common_cases.json"),):
        if os.path.exists(f):
            sel += [(d, "representative/common-command") for d in json.load(open(f))]
    d = pd.concat([load_results(os.path.join(RU.RESULTS, "main_grid", s)) for s in ("grid", "refine", "refine_bands")
                   if os.path.exists(os.path.join(RU.RESULTS, "main_grid", s, "results.jsonl"))], ignore_index=True)
    sc = all_scenarios()
    dis = d[d.outcome_terminal_cw.isin(["cw_only", "nl_only"])]
    for fam, g in dis.groupby(["meta_axis", "wn_ratio", "a_max"]):
        g = g.sort_values("sep_m")
        for sid in {g.sid.iloc[0], g.sid.iloc[-1]}:
            sel.append((sc[sid], "disagreement band"))
    br = pd.read_csv(os.path.join(RU.RESULTS, "main_grid", "refine", "boundary_brackets.csv"))
    b = br[(br.predicate == "nl_terminal") & (br.wn_ratio == 6.0) & (br.a_max == 0.05)]
    for r in b.itertuples():
        for sid in (r.sid_lo, r.sid_hi):
            sel.append((sc[sid], "arrival boundary (6n, 0.05)"))
    seen, out = set(), []
    for d_, why in sel:
        if d_["sid"] not in seen:
            seen.add(d_["sid"])
            d_ = dict(d_)
            d_.setdefault("meta", {})["history_reason"] = why
            out.append(d_)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["handover"], default=None)
    ap.add_argument("--sid", action="append", default=[])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    todo = []
    if a.preset == "handover":
        todo += handover_selection()
    if a.sid:
        sc = all_scenarios()
        todo += [sc[s] for s in a.sid]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(one, [(d, a.stride) for d in todo]))
    idx = os.path.join(OUT, "index.csv")
    old = pd.read_csv(idx) if os.path.exists(idx) else pd.DataFrame()
    new = pd.DataFrame(rows)
    pd.concat([old, new], ignore_index=True).drop_duplicates("sid", keep="last").to_csv(idx, index=False)
    print(new.to_string())


if __name__ == "__main__":
    main()
