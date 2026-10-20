import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
ARCH = os.path.dirname(REPO)
V3 = os.path.join(ARCH, "v3")
V31 = os.path.join(V3, "v31")

from cwvalid3 import ledger as L
from cwvalid3.domains import CELLS


def find(case_id):
    for p in glob.glob(os.path.join(V3, "raw", "manifests", "*.jsonl")):
        for r in L.readjsonlstrict(p, key="run_id"):
            if r["case_id"] == case_id:
                res = os.path.join(V3, "raw", "results", os.path.basename(p))
                stored = [x for x in L.readjsonlstrict(res, key="run_id", allow_missing=True) if x["case_id"] == case_id]
                return r, (stored[0] if stored else None)
    raise SystemExit(f"case {case_id} not found in any manifest")


def replaycont(rec):
    from cwvalid.control import PDLaw
    from cwvalid.experiment import RunCfg, runbranch
    from cwvalid.scenario import Scenario
    from cwvalid3.continuous import estmat, evalgen, runbranchgen
    from cwvalid3.controllers import MatLaw, pdlaw
    s = Scenario.from_dict(rec["scenario"])
    pdrule = rec["controller"].get("rule") == "scalar_pd"
    law = PDLaw.bandw(s.wn_ratio * s.n, s.a_max) if pdrule else MatLaw(rec["K"], s.a_max, rec["controller"])
    cfg = RunCfg(models=("cw", "eci", "nerm"), reference="eci", crosscheck="nerm")
    sims = runbranch(s, cfg)[0] if pdrule else runbranchgen(s, cfg, law)[0]
    row, _ = evalgen(s, cfg, sims, law)
    mlaw = pdlaw(s.wn_ratio * s.n, s.a_max) if pdrule else law
    t = np.append(np.arange(0.0, s.horizon, 0.5), s.horizon)
    H = {"t_s": t}
    for b in ("cw", "eci", "nerm"):
        te = sims[b].tfinal
        S = sims[b].states(np.minimum(t, te))
        S[t > te + 1e-9] = np.nan
        for k, nme in enumerate(("x", "y", "z", "vx", "vy", "vz")):
            H[f"{b}_{nme}"] = S[:, k]
    H["disc_cw_eci_m"] = np.linalg.norm(np.c_[H["cw_x"], H["cw_y"], H["cw_z"]] - np.c_[H["eci_x"], H["eci_y"], H["eci_z"]], axis=1)
    for k in ("A", "B", "C"):
        out = estmat(sims["cw"], mlaw, s.n, s.R, s.horizon, k, dt_out=0.5, keep=True)
        if out["status"] == "ok":
            H[f"e_{k}_m"] = np.interp(t, out["t"], np.linalg.norm(out["E"][:, :3], axis=1))
    scal = {"disc_m": row["cw_vs_eci_max_pos_disc_m"], "cw_status": row["cw_status"], "eci_status": row["eci_status"]}
    return pd.DataFrame(H), scal


def replayheld(rec):
    from cwvalid3.heldmet import CaseView, casesumm
    from cwvalid3.sampled import HeldBatch
    c = CELLS[rec["cell"]]
    Ts, alt = c["Ts"], c["altitude"]
    hb = HeldBatch([rec["scenario"]["rel0"]], np.array([rec["K"]]), [rec["scenario"]["a_max"]], alt, Ts, h=min(Ts, 2.0),
                   models=("cw", "nerm", "eci"), ests=("A_H", "B_H", "C_H")).run()
    hn = HeldBatch([rec["scenario"]["rel0"]], np.array([rec["K"]]), [rec["scenario"]["a_max"]], alt, Ts, h=0.5,
                   models=("cw",), ests=("A_naive",)).run()
    sm, cv = casesumm(hb, 0, pairs=(("cw", "eci"), ("cw", "nerm")))
    cvn = CaseView(hn, 0)
    H = {"t_s": cv.tm}
    for b in ("cw", "eci", "nerm"):
        S = cv.grid(b).copy()
        S[cv.tm > hb.t_end[b][0] + 1e-9] = np.nan
        for k, nme in enumerate(("x", "y", "z", "vx", "vy", "vz")):
            H[f"{b}_{nme}"] = S[:, k]
    H["disc_cw_eci_m"] = np.linalg.norm(cv.grid("cw")[:, :3] - cv.grid("eci")[:, :3], axis=1)
    for e in ("A_H", "B_H", "C_H"):
        H[f"e_{e}_m"] = np.linalg.norm(cv.grid(e)[:, :3], axis=1)
    H["e_A_naive_m"] = np.linalg.norm(cvn.grid("A_naive")[:, :3], axis=1)
    sat = hb.sat_at_update[0]
    k = np.clip(np.searchsorted(hb.updatetimes, cv.tm, side="right") - 1, 0, sat.size - 1)
    H["cw_saturated_held"] = sat[k]
    scal = {"disc_m": sm["cw_vs_eci_peak_m"], "t_peak_s": sm["cw_vs_eci_t_peak_s"], "cw_status": sm["cw_status"],
            "eci_status": sm["eci_status"], "eci_t_end_s": sm["eci_t_end_s"], "nerm_t_end_s": sm["nerm_t_end_s"],
            "eci_exit_step": sm.get("eci_exit_step"), "nerm_exit_step": sm.get("nerm_exit_step")}
    return pd.DataFrame(H), scal


def replay(case_id, out_dir):
    sys.path.insert(0, HERE)
    import campaign as _CP
    _CP.v30treeguard(out_dir)
    rec, stored = find(case_id)
    held = CELLS[rec["cell"]]["cmd_mode"] == "sampled"
    H, scal = replayheld(rec) if held else replaycont(rec)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{case_id}.csv")
    H.to_csv(p, index=False)
    ref = None
    if stored is not None:
        ref = stored.get("cw_vs_eci_peak_m") if held else stored.get("cw_vs_eci_max_pos_disc_m")
    meta = {"case_id": case_id, "cell": rec["cell"], "split": rec["split"], "index": rec["index"], "run_id": rec["run_id"],
            "draw": rec["draw"], "controller": rec["controller"], "replayed": scal, "stored_disc_m": ref,
            "abs_diff_disc_m": None if ref is None else abs(float(ref) - float(scal["disc_m"])), "history_csv": os.path.basename(p)}
    sys.path.insert(0, HERE)
    import campaign as CP
    meta["engine_sha"] = CP.enginever()
    meta["stored_engine_sha"] = (stored or {}).get("software", {}).get("engine_sha")
    json.dump(L.jsonable(meta), open(os.path.join(out_dir, f"{case_id}.json"), "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True, nargs="+")
    ap.add_argument("--out", default=os.path.join(V31, "raw", "counterexamples"))
    a = ap.parse_args()
    for c in a.case_id:
        print(json.dumps(L.jsonable(replay(c, a.out))))


if __name__ == "__main__":
    main()
