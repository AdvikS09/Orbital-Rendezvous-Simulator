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
import reportutil as RU
from maingrid import CONFIG

from cwvalid.experiment import runscen
from cwvalid.runner import configdict, loadres
from cwvalid.scenario import Scenario

OUT = os.path.join(RU.RESULTS, "histories")


def allscen():
    out = {}
    for p in glob.glob(os.path.join(RU.RESULTS, "**", "scenarios.jsonl"), recursive=True):
        for line in open(p):
            if line.strip():
                d = json.loads(line)
                out[d["sid"]] = d
    return out


def one(args):
    scnd, stride = args
    scn = Scenario.from_dict(scnd)
    cfg = configdict(CONFIG)
    row, hist = runscen(scn, cfg, keep_histories=True, hist_stride=stride)
    hist["scenario_json"] = np.array(json.dumps(scn.to_dict()))
    hist["config_json"] = np.array(json.dumps(CONFIG))
    np.savez_compressed(os.path.join(OUT, f"{scn.sid}.npz"), **hist)
    return {"sid": scn.sid, "family": scn.family, **{f"meta_{k}": v for k, v in scn.meta.items()},
            "wn_ratio": scn.wn_ratio, "a_max": scn.a_max, "sep_m": scn.separation,
            "outcome_terminal_cw": row["outcome_terminal_cw"], "peak_disc_m": row["cw_vs_eci_max_pos_disc_m"],
            "file": f"histories/{scn.sid}.npz"}


def archivesel():
    sel = []
    for f in (os.path.join(RU.RESULTS, "mechanisms", "commoncases.json"),):
        if os.path.exists(f):
            sel += [(d, "representative/common-command") for d in json.load(open(f))]
    d = pd.concat([loadres(os.path.join(RU.RESULTS, "maingrid", s)) for s in ("grid", "refine", "refinebands")
                   if os.path.exists(os.path.join(RU.RESULTS, "maingrid", s, "results.jsonl"))], ignore_index=True)
    sc = allscen()
    dis = d[d.outcome_terminal_cw.isin(["cw_only", "nl_only"])]
    for fam, g in dis.groupby(["meta_axis", "wn_ratio", "a_max"]):
        g = g.sort_values("sep_m")
        for sid in {g.sid.iloc[0], g.sid.iloc[-1]}:
            sel.append((sc[sid], "disagreement band"))
    br = pd.read_csv(os.path.join(RU.RESULTS, "maingrid", "refine", "boundarybrackets.csv"))
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
    ap.add_argument("--preset", choices=["archive"], default=None)
    ap.add_argument("--sid", action="append", default=[])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    todo = []
    if a.preset == "archive":
        todo += archivesel()
    if a.sid:
        sc = allscen()
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
