import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ext"))
import v2common as C

from cwvalid.runner import readjsonl2, _work

CAMP = {"maingrid/grid": 8, "maingrid/refine": 6, "maingrid/refinebands": 6, "mechanisms/horizon2": 4,
        "screening/dev": 4, "screening/test": 4, "screening/stress": 3, "screening/bandtest": 3}
SKIP = {"run_wall_s", "config_hash", "history_file", "screen_wall_s"}


def pick(campaign, rows, k, rng):
    by = {r["sid"]: r for r in rows}
    ids = sorted(by)
    chosen = list(rng.choice(ids, k, replace=False))
    extra = []
    if campaign == "mechanisms/horizon2":
        extra += [s for s in ids if by[s].get("so_status") == "failed"][:2]
        extra += [s for s in ids if by[s].get("outcome_terminal_cw") == "cw_only"][:2]
    if campaign in ("maingrid/refine", "maingrid/refinebands"):
        extra += [s for s in ids if by[s].get("outcome_terminal_cw") in ("cw_only", "nl_only")][:2]
        extra += [s for s in ids if by[s].get("outcome_terminal_so") in ("cw_only", "nl_only")][:1]
    return chosen + [s for s in extra if s not in chosen]


def compare(old, new):
    worstrel, worstcol, diffs = 0.0, "", []
    for k, v in old.items():
        if k in SKIP or "wall" in k or k.startswith("meta_keep_history"):
            continue
        w = new.get(k, "<missing>")
        if isinstance(v, (bool, str)) or v is None or isinstance(w, (bool, str)) or w is None:
            if v != w:
                diffs.append(f"{k}: {v} -> {w}")
            continue
        if isinstance(v, (int, float)) and isinstance(w, (int, float)):
            if v != w:
                r = abs(w - v) / max(abs(v), 1e-300)
                if r > worstrel:
                    worstrel, worstcol = r, k
        elif v != w:
            diffs.append(f"{k}: changed")
    return worstrel, worstcol, diffs


def feat(s):
    from screening import cwonlyfeats
    return cwonlyfeats(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    t0 = time.perf_counter()
    rng = np.random.default_rng(20261011)
    jobs, meta, fjobs = [], [], []
    for c, k in CAMP.items():
        d = os.path.join(C.RESULTS, c)
        cfg = json.load(open(os.path.join(d, "config.json")))["config"]
        scns = {s["sid"]: s for s in readjsonl2(os.path.join(d, "scenarios.jsonl"))}
        rows = {r["sid"]: r for r in readjsonl2(os.path.join(d, "results.jsonl"))}
        sf = {}
        if os.path.exists(os.path.join(d, "screenfeatures.jsonl")):
            sf = {r["sid"]: r for r in readjsonl2(os.path.join(d, "screenfeatures.jsonl"))}
        for sid in pick(c, list(rows.values()), k, rng):
            s = dict(scns[sid])
            s["meta"] = dict(s.get("meta", {}))
            s["meta"]["keep_history"] = False
            jobs.append((s, cfg, None))
            meta.append((c, sid, rows[sid]))
            if sid in sf:
                fjobs.append((c, sid, scns[sid], sf[sid]))
    print(f"[repro {a.tag}] re-running {len(jobs)} paired scenarios and {len(fjobs)} CW-only feature sets", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_work, jobs))
        fres = list(ex.map(feat, [f[2] for f in fjobs]))
    out = []
    for (c, sid, old), r in zip(meta, res):
        if not r["ok"]:
            out.append({"campaign": c, "sid": sid, "rerun_ok": False, "note": r["error"][-300:]})
            continue
        wr, wc, diffs = compare(old, r["row"])
        out.append({"campaign": c, "sid": sid, "rerun_ok": True, "max_rel_change": wr, "column_at_max": wc,
                    "n_nonnumeric_changes": len(diffs), "nonnumeric_changes": "; ".join(diffs[:8]),
                    "so_status": old.get("so_status"), "outcome_terminal_cw": old.get("outcome_terminal_cw")})
    fo = []
    for (c, sid, _, old), new in zip(fjobs, fres):
        wr, wc, diffs = compare(old, new)
        fo.append({"campaign": c, "sid": sid, "d1_stored_m": old.get("d1_peak_m"), "d1_rerun_m": new.get("d1_peak_m"),
                   "max_rel_change": wr, "column_at_max": wc, "n_nonnumeric_changes": len(diffs)})
    od = os.path.join(C.RAW, "reprocheck", a.tag)
    os.makedirs(od, exist_ok=True)
    df, fd = pd.DataFrame(out), pd.DataFrame(fo)
    df.to_csv(os.path.join(od, "rows.csv"), index=False)
    fd.to_csv(os.path.join(od, "features.csv"), index=False)
    summ = {"tag": a.tag, "code": C.coderev(), "environment": C.environment(),
            "n_paired": len(df), "n_rerun_failed": int((~df.rerun_ok).sum()),
            "max_rel_change_paired": float(df.max_rel_change.max()), "nonnumeric_changes_paired": int(df.n_nonnumeric_changes.sum()),
            "n_features": len(fd), "max_rel_change_features": float(fd.max_rel_change.max()) if len(fd) else None,
            "wall_s": time.perf_counter() - t0}
    C.writejson(os.path.join(od, "summary.json"), summ)
    C.logrun("repro_check", f"python experiments/v2/reprocheck.py --tag {a.tag}", "completed", summ["wall_s"])
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
