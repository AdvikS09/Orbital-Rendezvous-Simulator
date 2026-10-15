"""Re-run check for campaigns that ran before the code-review fixes.

The pilot, main grid, boundary refinement, unrestricted and energy campaigns were
run before the review fixes to cwvalid were committed (their run_config.json has
no config_hash). This script re-runs a seeded sample of each with the current
code and the campaign's own stored configuration and compares every stored field.

Sample per campaign: 15 seeded random scenarios, up to 5 whose reference branch
stopped at the altitude floor, and up to 5 with a CW/reference outcome mismatch.

Output: results/corrections/prefix_rerun_check.csv (one row per scenario) and
        results/corrections/prefix_rerun_summary.csv (one row per campaign)
Usage:  python experiments/ext/prefix_rerun_check.py
"""

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

from cwvalid.runner import _read_jsonl, _work  # noqa: E402

CAMPAIGNS = ["main_grid/pilot", "main_grid/grid", "main_grid/refine",
             "mechanisms/unrestricted", "mechanisms/energy"]
SKIP = {"run_wall_s", "config_hash", "history_file"}
TIMING = ("_wall_s", "_time_s")


def pick(rows, rng):
    by = {r["sid"]: r for r in rows}
    ids = list(rng.choice(sorted(by), 15, replace=False))
    floor = [s for s, r in sorted(by.items()) if r.get("eci_termination") == "altitude_floor" and s not in ids]
    mism = [s for s, r in sorted(by.items()) if r.get("outcome_terminal_cw") in ("cw_only", "nl_only") and s not in ids]
    return ids + floor[:5] + mism[:5]


def compare(old, new):
    worst_rel, worst_abs, worst_col, diffs = 0.0, 0.0, "", []
    for k, v in old.items():
        if k in SKIP or k.endswith(TIMING):
            continue
        w = new.get(k, "<missing>")
        if isinstance(v, bool) or isinstance(v, str) or v is None or isinstance(w, (bool, str)) or w is None:
            if v != w:
                diffs.append(f"{k}: {v} -> {w}")
            continue
        if isinstance(v, (int, float)) and isinstance(w, (int, float)):
            if v == w:
                continue
            a = abs(w - v)
            r = a / max(abs(v), 1e-300)
            if r > worst_rel:
                worst_rel, worst_abs, worst_col = r, a, k
        elif v != w:
            diffs.append(f"{k}: changed")
    new_cols = sorted(set(new) - set(old) - SKIP)
    return worst_rel, worst_abs, worst_col, diffs, new_cols


def main():
    rng = np.random.default_rng(20261010)
    jobs, meta = [], []
    for c in CAMPAIGNS:
        d = os.path.join(RU.RESULTS, c)
        cfg = json.load(open(os.path.join(d, "run_config.json")))["config"]
        scns = {s["sid"]: s for s in _read_jsonl(os.path.join(d, "scenarios.jsonl"))}
        rows = {r["sid"]: r for r in _read_jsonl(os.path.join(d, "results.jsonl"))}
        for sid in pick(list(rows.values()), rng):
            s = dict(scns[sid])
            s["meta"] = {**s.get("meta", {}), "keep_history": False}
            jobs.append((s, cfg, None))
            meta.append((c, sid, rows[sid]))
    print(f"re-running {len(jobs)} scenarios", flush=True)
    with ProcessPoolExecutor(7) as ex:
        res = list(ex.map(_work, jobs))
    out = []
    for (c, sid, old), r in zip(meta, res):
        if not r["ok"]:
            out.append({"campaign": c, "sid": sid, "rerun_ok": False, "note": r["error"][-300:]})
            continue
        wr, wa, wc, diffs, newc = compare(old, r["row"])
        out.append({"campaign": c, "sid": sid, "rerun_ok": True,
                    "eci_termination": old.get("eci_termination"), "outcome_terminal_cw": old.get("outcome_terminal_cw"),
                    "max_rel_change": wr, "abs_change_at_max": wa, "column_at_max": wc,
                    "n_nonnumeric_changes": len(diffs), "nonnumeric_changes": "; ".join(diffs[:10]),
                    "fields_added_by_current_code": ", ".join(newc)})
    df = pd.DataFrame(out)
    od = os.path.join(RU.RESULTS, "corrections")
    os.makedirs(od, exist_ok=True)
    df.to_csv(os.path.join(od, "prefix_rerun_check.csv"), index=False)
    summ = df.groupby("campaign").agg(n=("sid", "size"), all_ok=("rerun_ok", "all"),
                                      max_rel_change=("max_rel_change", "max"),
                                      nonnumeric_changes=("n_nonnumeric_changes", "sum")).reset_index()
    summ.to_csv(os.path.join(od, "prefix_rerun_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.to_string())
    print(df[["campaign", "sid", "max_rel_change", "column_at_max", "n_nonnumeric_changes"]].sort_values("max_rel_change").tail(12).to_string())


if __name__ == "__main__":
    main()
