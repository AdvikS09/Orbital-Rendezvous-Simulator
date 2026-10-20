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
SCRIPTS = (("p08atlas.py", ["--no-replay"]), ("p08cohort.py", []), ("evaltest.py", []),
           ("anadev.py", ["--splits", "pilot", "dev", "--tag", "pilotdev"]), ("assets.py", []))
WRITES = []
_open = builtins.open


def _v30(p):
    return P.inclosed(p)


def tgt(p):
    rel = os.path.relpath(os.path.realpath(p), os.path.realpath(P.V3))
    return os.path.join(REBUILT, rel)


def ropen(file, mode="r", *a, **k):
    if isinstance(file, (str, os.PathLike)):
        p = os.path.abspath(os.fspath(file))
        if _v30(p):
            q = tgt(p)
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


def cmpcsv(a, b):
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
    with _open(os.path.join(P.V31RAW, "results", "heldexit31.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            replay[r["replay"]["of_run_id"]] = r
    used = set()
    origread = L.readjsonlstrict

    def readoverlay(path, key=None, allow_missing=False):
        rows = origread(path, key=key, allow_missing=allow_missing)
        ap = os.path.abspath(path)
        if os.path.dirname(ap) == os.path.abspath(P.RES) and os.path.basename(ap)[:2] in ("D3", "D4"):
            out = []
            for r in rows:
                if r.get("run_id") in replay:
                    used.add(r["run_id"])
                    out.append({**replay[r["run_id"]], **{k: v for k, v in r.items() if SER.volat(k)}})
                else:
                    out.append(r)
            return out
        return rows
    L.readjsonlstrict = readoverlay
    builtins.open = ropen
    io.open = ropen
    os.environ["CWVALID3_REBUILD_HARNESS"] = "1"
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
        L.readjsonlstrict = origread
        os.environ.pop("CWVALID3_REBUILD_HARNESS", None)
    rows = []
    for rel in sorted({os.path.relpath(w, P.V3) for w in WRITES}):
        o, n = os.path.join(P.V3, rel), os.path.join(REBUILT, rel)
        r = {"path": rel.replace("\\", "/"), "original_sha256": sha(o) if os.path.exists(o) else None, "rebuilt_sha256": sha(n)}
        r["identical"] = r["original_sha256"] == r["rebuilt_sha256"]
        if not r["identical"] and rel.endswith(".csv") and r["original_sha256"]:
            r.update(cmpcsv(o, n))
        rows.append(r)
    os.makedirs(P.V31RES, exist_ok=True)
    cols = ["path", "identical", "original_sha256", "rebuilt_sha256", "n_rows", "n_rows_changed", "changed_columns", "max_abs_change"]
    with _open(P.guardv30(os.path.join(P.V31RES, "changedoutputs.csv")), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    summ = {"kind": "rebuild from stored records (C01 replay overlay); no propagation", "replay_records": len(replay),
            "replay_records_used": len(used), "scripts": log, "n_files_written": len(rows),
            "n_identical": sum(r["identical"] for r in rows), "changed": [r["path"] for r in rows if not r["identical"]]}
    with _open(P.guardv30(os.path.join(P.V31RES, "rebuildlog.json")), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
