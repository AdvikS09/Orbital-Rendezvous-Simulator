import argparse
import json
import math
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from . import __version__
from .experiment import RunCfg, runscen
from .metrics import OutcCrit
from .scenario import Scenario
from .simulate import EciCfg, SolverCfg
from .dynamics import InertForce


def softwvers():
    import pandas
    import scipy
    import matplotlib
    return {"python": sys.version, "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pandas.__version__, "matplotlib": matplotlib.__version__,
            "cwvalid": __version__}


def jsonable(v):
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None if math.isnan(v) else ("inf" if v > 0 else "-inf")
    return v


def configdict(d):
    d = dict(d or {})
    solver = SolverCfg(**d.pop("solver", {}))
    crit = OutcCrit(**d.pop("crit", {}))
    ecid = d.pop("eci", None)
    eci = EciCfg()
    if ecid:
        eci = EciCfg(inclination=ecid.get("inclination", eci.inclination),
                     force_target=InertForce(**ecid.get("force_target", {})),
                     force_chaser=InertForce(**ecid.get("force_chaser", {})),
                     exact_frame_rate=ecid.get("exact_frame_rate", True))
    if "models" in d:
        d["models"] = tuple(d["models"])
    return RunCfg(solver=solver, crit=crit, eci=eci, **d)


def cfghash(cfgdict):
    import hashlib
    return hashlib.sha256(json.dumps(cfgdict, sort_keys=True, default=str).encode()).hexdigest()[:16]


def readjsonl2(path, log=print):
    rows, bad = [], 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        log(f"[runner] WARNING: {bad} unparsable line(s) skipped in {path}")
    return rows


def _work(args):
    scnd, cfgd, histdir = args
    scn = Scenario.from_dict(scnd)
    cfg = configdict(cfgd)
    keep = bool(scn.meta.get("keep_history", False)) and histdir is not None
    stride = int(scn.meta.get("hist_stride", 4))
    t0 = time.perf_counter()
    try:
        row, hist = runscen(scn, cfg, keep_histories=keep, hist_stride=stride)
        row["run_wall_s"] = time.perf_counter() - t0
        row["config_hash"] = cfghash(cfgd)
        if keep and hist is not None:
            np.savez_compressed(os.path.join(histdir, f"{scn.sid}.npz"), **hist)
            row["history_file"] = f"{scn.sid}.npz"
        return {"ok": True, "row": {k: jsonable(v) for k, v in row.items()}}
    except Exception:
        return {"ok": False, "sid": scn.sid, "family": scn.family,
                "error": traceback.format_exc(), "wall_s": time.perf_counter() - t0}


def runcamp(out_dir, scenarios, cfgdict, workers=7, limit=None, keephist=True, log=print):
    os.makedirs(out_dir, exist_ok=True)
    histdir = os.path.join(out_dir, "histories") if keephist else None
    if histdir:
        os.makedirs(histdir, exist_ok=True)
    respath = os.path.join(out_dir, "results.jsonl")
    failpath = os.path.join(out_dir, "failures.jsonl")
    done = set()
    if os.path.exists(respath):
        done = {r["sid"] for r in readjsonl2(respath, log) if "sid" in r}
        with open(respath, "rb+") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() > 0:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    fh.write(b"\n")
    h = cfghash(cfgdict)
    rcpath = os.path.join(out_dir, "config.json")
    if os.path.exists(rcpath) and done:
        prev = json.load(open(rcpath))
        prevh = prev.get("config_hash") or cfghash(prev.get("config", {}))
        if prevh != h:
            raise RuntimeError(f"{out_dir}: configuration differs from the one used for existing results "
                               f"({prevh} vs {h}); use a new output directory")
    with open(rcpath, "w") as fh:
        json.dump({"config": cfgdict, "config_hash": h, "software": softwvers()}, fh, indent=2, default=str)
    allids = [s["sid"] if isinstance(s, dict) else s.sid for s in scenarios]
    if len(set(allids)) != len(allids):
        raise ValueError("duplicate scenario IDs in campaign")
    todo = [s if isinstance(s, dict) else s.to_dict() for s in scenarios]
    todo = [s for s in todo if s["sid"] not in done]
    if limit is not None:
        todo = todo[:limit]
    log(f"[runner] {out_dir}: {len(allids)} scenarios, {len(done)} already done, {len(todo)} to run")
    nok = nfail = 0
    tstart = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex, \
                open(respath, "a") as fres, open(failpath, "a") as ffail:
            futs = [ex.submit(_work, (s, cfgdict, histdir)) for s in todo]
            for k, fut in enumerate(as_completed(futs)):
                r = fut.result()
                if r["ok"]:
                    fres.write(json.dumps(r["row"]) + "\n")
                    fres.flush()
                    nok += 1
                else:
                    ffail.write(json.dumps(r) + "\n")
                    ffail.flush()
                    nfail += 1
                if (k + 1) % 100 == 0:
                    el = time.perf_counter() - tstart
                    log(f"[runner] {k + 1}/{len(todo)} done ({nfail} failed), {el:.0f}s elapsed")
    writeman(out_dir, allids)
    log(f"[runner] finished: {nok} ok, {nfail} failed, {time.perf_counter() - tstart:.1f}s")
    return nok, nfail


def writeman(out_dir, allids):
    respath = os.path.join(out_dir, "results.jsonl")
    failpath = os.path.join(out_dir, "failures.jsonl")
    ok, failed = set(), set()
    numrows = set()
    if os.path.exists(respath):
        for r in readjsonl2(respath):
            ok.add(r["sid"])
            if r.get("ambiguous") and "numerical_failure" in (r.get("ambiguity_flags") or ""):
                numrows.add(r["sid"])
    if os.path.exists(failpath):
        for r in readjsonl2(failpath):
            failed.add(r["sid"])
    failed -= ok
    unexec = [s for s in allids if s not in ok and s not in failed]
    man = {"total": len(allids), "completed": len(ok), "failed_exception": len(failed),
           "completed_with_branch_numerical_failure": len(numrows),
           "unexecuted": len(unexec), "failed_ids": sorted(failed), "unexecuted_ids": unexec,
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=2)
    return man


def loadres(out_dir):
    import pandas as pd
    rows = readjsonl2(os.path.join(out_dir, "results.jsonl"))
    df = pd.DataFrame(rows)
    return df.replace({"inf": np.inf, "-inf": -np.inf})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    camp = json.load(open(a.campaign))
    base = os.path.dirname(os.path.abspath(a.campaign))
    outdir = os.path.join(base, camp["out_dir"]) if not os.path.isabs(camp["out_dir"]) else camp["out_dir"]
    scenfile = os.path.join(base, camp["scenarios"]) if not os.path.isabs(camp["scenarios"]) else camp["scenarios"]
    scen = [json.loads(l) for l in open(scenfile) if l.strip()]
    runcamp(outdir, scen, camp.get("config", {}), workers=a.workers, limit=a.limit,
            keephist=camp.get("keep_histories", True))


if __name__ == "__main__":
    main()
