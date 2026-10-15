"""Replay one planned v3 case from its manifest record and write its full time history.

The case is re-propagated with exactly the frozen configuration of its cell (continuous: production
cwvalid functions for PD, cwvalid3.continuous for the matrix law; held: cwvalid3.sampled.HeldBatch with
the same step choices) and the scalar results are compared with the stored campaign record.
Histories (0.5 s grid; held: 0.5 s grid from the Hermite dense output): time, CW / ECI / NERM relative
states, the CW-ECI position discrepancy and the estimator position errors |e| (continuous: A, B, C on the
same grid via the production estimators re-integrated with dense output; held: A_H, B_H, C_H and A_naive).

Usage: python code/experiments/v3/replay_case.py --case-id C... [--out v3_extension/v3_1_correction/raw/counterexamples]

v3.1: the default output moved to v3_1_correction/raw/counterexamples and writing into the v3.0 tree is refused
(the v3.0 traces are kept unchanged). With the C01-corrected engine every branch history, including
disc_cw_eci_m, is NaN after that branch's end time (HeldBatch.dense masks it).
"""

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
V3 = os.path.join(ARCH, "v3_extension")
V31 = os.path.join(V3, "v3_1_correction")

from cwvalid3 import ledger as L  # noqa: E402
from cwvalid3.domains import CELLS  # noqa: E402


def find(case_id):
    for p in glob.glob(os.path.join(V3, "raw", "manifests", "*.jsonl")):
        for r in L.read_jsonl_strict(p, key="run_id"):
            if r["case_id"] == case_id:
                res = os.path.join(V3, "raw", "results", os.path.basename(p))
                stored = [x for x in L.read_jsonl_strict(res, key="run_id", allow_missing=True) if x["case_id"] == case_id]
                return r, (stored[0] if stored else None)
    raise SystemExit(f"case {case_id} not found in any manifest")


def replay_continuous(rec):
    from cwvalid.control import PDLaw
    from cwvalid.experiment import RunConfig, run_branches
    from cwvalid.scenario import Scenario
    from cwvalid3.continuous import estimator_matrix, evaluate_general, run_branches_general
    from cwvalid3.controllers import MatrixLaw, pd_law
    s = Scenario.from_dict(rec["scenario"])
    pd_rule = rec["controller"].get("rule") == "scalar_pd"
    law = PDLaw.from_bandwidth(s.wn_ratio * s.n, s.a_max) if pd_rule else MatrixLaw(rec["K"], s.a_max, rec["controller"])
    cfg = RunConfig(models=("cw", "eci", "nerm"), reference="eci", crosscheck="nerm")
    sims = run_branches(s, cfg)[0] if pd_rule else run_branches_general(s, cfg, law)[0]
    row, _ = evaluate_general(s, cfg, sims, law)
    mlaw = pd_law(s.wn_ratio * s.n, s.a_max) if pd_rule else law
    t = np.append(np.arange(0.0, s.horizon, 0.5), s.horizon)
    H = {"t_s": t}
    for b in ("cw", "eci", "nerm"):
        te = sims[b].t_final
        S = sims[b].states(np.minimum(t, te))
        S[t > te + 1e-9] = np.nan
        for k, nme in enumerate(("x", "y", "z", "vx", "vy", "vz")):
            H[f"{b}_{nme}"] = S[:, k]
    H["disc_cw_eci_m"] = np.linalg.norm(np.c_[H["cw_x"], H["cw_y"], H["cw_z"]] - np.c_[H["eci_x"], H["eci_y"], H["eci_z"]], axis=1)
    for k in ("A", "B", "C"):
        out = estimator_matrix(sims["cw"], mlaw, s.n, s.R, s.horizon, k, dt_out=0.5, keep=True)
        if out["status"] == "ok":
            H[f"e_{k}_m"] = np.interp(t, out["t"], np.linalg.norm(out["E"][:, :3], axis=1))
    scal = {"disc_m": row["cw_vs_eci_max_pos_disc_m"], "cw_status": row["cw_status"], "eci_status": row["eci_status"]}
    return pd.DataFrame(H), scal


def replay_held(rec):
    from cwvalid3.held_metrics import CaseView, case_summary
    from cwvalid3.sampled import HeldBatch
    c = CELLS[rec["cell"]]
    Ts, alt = c["Ts"], c["altitude"]
    hb = HeldBatch([rec["scenario"]["rel0"]], np.array([rec["K"]]), [rec["scenario"]["a_max"]], alt, Ts, h=min(Ts, 2.0),
                   models=("cw", "nerm", "eci"), estimators=("A_H", "B_H", "C_H")).run()
    hn = HeldBatch([rec["scenario"]["rel0"]], np.array([rec["K"]]), [rec["scenario"]["a_max"]], alt, Ts, h=0.5,
                   models=("cw",), estimators=("A_naive",)).run()
    sm, cv = case_summary(hb, 0, pairs=(("cw", "eci"), ("cw", "nerm")))
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
    upd = np.zeros(cv.tm.size, bool)
    sat = hb.sat_at_update[0]
    k = np.clip(np.searchsorted(hb.update_times, cv.tm, side="right") - 1, 0, sat.size - 1)
    H["cw_saturated_held"] = sat[k]
    scal = {"disc_m": sm["cw_vs_eci_peak_m"], "t_peak_s": sm["cw_vs_eci_t_peak_s"], "cw_status": sm["cw_status"],
            "eci_status": sm["eci_status"], "eci_t_end_s": sm["eci_t_end_s"], "nerm_t_end_s": sm["nerm_t_end_s"],
            "eci_exit_step": sm.get("eci_exit_step"), "nerm_exit_step": sm.get("nerm_exit_step")}
    return pd.DataFrame(H), scal


def replay(case_id, out_dir):
    sys.path.insert(0, HERE)
    import campaign as _CP
    _CP.v30_tree_guard(out_dir)        # case-normalised check (v3.1)
    rec, stored = find(case_id)
    held = CELLS[rec["cell"]]["cmd_mode"] == "sampled"
    H, scal = replay_held(rec) if held else replay_continuous(rec)
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
    meta["engine_sha"] = CP.engine_version()
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
