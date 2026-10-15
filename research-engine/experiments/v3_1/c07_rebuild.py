"""C07: rebuild the outputs that depend on held exit records, from stored records plus the C01 replay overlay.

Kind of work: REBUILD FROM STORED RECORDS. No propagation is run here. The v3.0 output scripts are executed
unchanged (p08_failure_atlas --no-replay, p08_cohort, evaluate_test, analyze_dev --splits pilot dev,
make_assets) inside a harness that
  * overlays the 341 C01-replayed held exit records (v3_1_correction/raw/results/held_exit_replay_v31.jsonl,
    keyed by the original run_id) on the stored D3/D4 result records when they are read (physics fields from
    the replay; timing and provenance fields of the stored campaign record are kept);
  * redirects every write aimed at the v3.0 tree to v3_1_correction/rebuilt/<same relative path>, and serves
    reads of an already rebuilt file from there (so a script that re-reads its own output sees the rebuilt one).
Labelling is the frozen v3.0 one (cwvalid3.calibration); the v3.1 contract differs from it only for the 43 D1
worker_crash records (C02), which these outputs already report as operational interruptions.
Afterwards every rebuilt file is compared with its v3.0 original:
  v3_1_correction/results/CHANGED_OUTPUTS_MANIFEST.csv  path, hashes, identical?, rows/columns changed, max change
The v3.0 tree is verified unchanged at the end (c00_baseline.py --tag final).
Usage: python code/experiments/v3_1/c07_rebuild.py
"""

import builtins
import csv
import hashlib
import io
import json
import os
import runpy
import sys
import time

import v31paths as P

sys.path.insert(0, os.path.join(P.REPO, "experiments", "v3"))
REBUILT = os.path.join(P.V31, "rebuilt")
SCRIPTS = (("p08_failure_atlas.py", ["--no-replay"]), ("p08_cohort.py", []), ("evaluate_test.py", []),
           ("analyze_dev.py", ["--splits", "pilot", "dev", "--tag", "pilot_dev"]), ("make_assets.py", []))
WRITES = []
_open = builtins.open


def _v30(p):
    return P.in_closed_v30(p)


def _target(p):
    rel = os.path.relpath(os.path.realpath(p), os.path.realpath(P.V3))   # ntpath.relpath matches case-insensitively, keeps the file's case
    return os.path.join(REBUILT, rel)


def r_open(file, mode="r", *a, **k):
    if isinstance(file, (str, os.PathLike)):
        p = os.path.abspath(os.fspath(file))
        if _v30(p):
            q = _target(p)
            if any(c in mode for c in "wax+"):
                os.makedirs(os.path.dirname(q), exist_ok=True)
                WRITES.append(p)
                return _open(q, mode, *a, **k)
            if os.path.exists(q):
                return _open(q, mode, *a, **k)
    return _open(file, mode, *a, **k)


def sha(p):
    h = hashlib.sha256()
    with _open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def compare_csv(a, b):
    """Cell-by-cell comparison of the written text (no dtype inference); numeric size of the changes."""
    import numpy as np
    import pandas as pd
    x = pd.read_csv(a, dtype=str, keep_default_na=False)
    y = pd.read_csv(b, dtype=str, keep_default_na=False)
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        return {"n_rows": len(y), "n_rows_changed": None, "changed_columns": "shape or columns differ", "max_abs_change": None}
    changed, rows, mx = [], np.zeros(len(x), bool), 0.0
    for c in x.columns:
        d = (x[c] != y[c]).to_numpy()
        if d.any():
            changed.append(f"{c}({int(d.sum())})")
            rows |= d
            u = pd.to_numeric(x[c][d], errors="coerce").to_numpy(float)
            v = pd.to_numeric(y[c][d], errors="coerce").to_numpy(float)
            ok = np.isfinite(u) & np.isfinite(v)
            if ok.any():
                mx = max(mx, float(np.max(np.abs(u[ok] - v[ok]))))
    return {"n_rows": len(y), "n_rows_changed": int(rows.sum()), "changed_columns": ";".join(changed), "max_abs_change": mx}


def main():
    from cwvalid3 import ledger as L
    from cwvalid3 import serialization as SER
    replay = {}
    with _open(os.path.join(P.V31_RAW, "results", "held_exit_replay_v31.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            replay[r["replay"]["of_run_id"]] = r
    used = set()
    orig_read = L.read_jsonl_strict

    def read_overlay(path, key=None, allow_missing=False):
        rows = orig_read(path, key=key, allow_missing=allow_missing)
        ap = os.path.abspath(path)
        if os.path.dirname(ap) == os.path.abspath(P.RES) and os.path.basename(ap)[:3] in ("D3_", "D4_"):
            out = []
            for r in rows:
                if r.get("run_id") in replay:
                    used.add(r["run_id"])
                    # physics from the replay; timing/provenance of the stored campaign record are kept
                    out.append({**replay[r["run_id"]], **{k: v for k, v in r.items() if SER.is_volatile(k)}})
                else:
                    out.append(r)
            return out
        return rows
    L.read_jsonl_strict = read_overlay
    builtins.open = r_open
    io.open = r_open
    os.environ["CWVALID3_REBUILD_HARNESS"] = "1"     # lets the v3.0 writers run (experiments/v3/v30_closed.py)
    log = []
    try:
        for script, args in SCRIPTS:
            t0 = time.perf_counter()
            n0 = len(WRITES)
            sys.argv = [script] + args
            try:
                runpy.run_path(os.path.join(P.REPO, "experiments", "v3", script), run_name="__main__")
                status = "ok"
            except SystemExit as e:
                status = f"exit {e.code}"
            log.append({"script": script, "args": " ".join(args), "status": status, "wall_s": time.perf_counter() - t0,
                        "writes": sorted({os.path.relpath(w, P.V3) for w in WRITES[n0:]})})
    finally:
        builtins.open = _open
        io.open = _open
        L.read_jsonl_strict = orig_read
        os.environ.pop("CWVALID3_REBUILD_HARNESS", None)
    rows = []
    for rel in sorted({os.path.relpath(w, P.V3) for w in WRITES}):
        o, n = os.path.join(P.V3, rel), os.path.join(REBUILT, rel)
        r = {"path": rel.replace("\\", "/"), "original_sha256": sha(o) if os.path.exists(o) else None, "rebuilt_sha256": sha(n)}
        r["identical"] = r["original_sha256"] == r["rebuilt_sha256"]
        if not r["identical"] and rel.endswith(".csv") and r["original_sha256"]:
            r.update(compare_csv(o, n))
        rows.append(r)
    os.makedirs(P.V31_RES, exist_ok=True)
    cols = ["path", "identical", "original_sha256", "rebuilt_sha256", "n_rows", "n_rows_changed", "changed_columns", "max_abs_change"]
    with _open(P.guard_v30(os.path.join(P.V31_RES, "CHANGED_OUTPUTS_MANIFEST.csv")), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    summ = {"kind": "rebuild from stored records (C01 replay overlay); no propagation", "replay_records": len(replay),
            "replay_records_used": len(used), "scripts": log, "n_files_written": len(rows),
            "n_identical": sum(r["identical"] for r in rows), "changed": [r["path"] for r in rows if not r["identical"]]}
    with _open(P.guard_v30(os.path.join(P.V31_RES, "rebuild_log.json")), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
