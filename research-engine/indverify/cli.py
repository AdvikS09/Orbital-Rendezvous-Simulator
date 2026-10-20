import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .vinfo import __version__
from .constants import FINEDT
from .runner import SolverSet, normbranch, runcase

RESNAME = "indverifyresults.jsonl"
BADNAME = "indverify_results.bad"
HISTDIR = "histories"
THREADVARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
UNSAFE = re.compile(r"[^A-Za-z0-9._+=-]")


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    return obj


def casesid(case, index):
    if isinstance(case, dict) and case.get("sid") is not None:
        return str(case["sid"])
    return "#%d" % index


def setlabel(setting, index):
    try:
        return SolverSet.from_dict(setting).label
    except Exception:
        if isinstance(setting, dict) and setting.get("label"):
            return str(setting["label"])
        return "#setting%d" % index


def sidhash(sid):
    return hashlib.sha1(str(sid).encode("utf-8")).hexdigest()[:10]


def histfnames(sids):
    sids = [str(s) for s in sids]
    safe = {s: UNSAFE.sub("_", s) for s in sids}
    counts = Counter(v.lower() for v in safe.values())
    out = {}
    for s in sids:
        name = safe[s]
        if name != s or counts[name.lower()] > 1 or not name or name.startswith("."):
            name = "%s_%s" % (name, sidhash(s))
        out[s] = name + ".npz"
    lowered = [v.lower() for v in out.values()]
    if len(set(lowered)) != len(lowered):
        raise SystemExit("history file names collide; rename the scenarios")
    return out


def histfname(sid):
    return histfnames([sid])[str(sid)]


def loadjsonlist(path, what):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise SystemExit("%s must contain a JSON list" % what)
    return data


def recordkey(rec):
    return str(rec.get("sid")), str(rec.get("setting_label"))


def compactres(path, badpath, retry_errors=True):
    stats = {"lines": 0, "unparseable": 0, "superseded": 0, "errors_to_retry": 0,
             "errors_kept": 0, "rewritten": False}
    if not os.path.exists(path):
        return set(), stats
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        rawlines = [ln for ln in fh.read().split("\n") if ln.strip()]
    stats["lines"] = len(rawlines)
    parsed, moved = [], []
    for raw in rawlines:
        try:
            rec = json.loads(raw)
            if not isinstance(rec, dict):
                raise ValueError("not a JSON object")
        except ValueError:
            moved.append(raw)
            stats["unparseable"] += 1
            continue
        parsed.append((recordkey(rec), "error" in rec, raw))
    best = {}
    for i, (key, iserr, _) in enumerate(parsed):
        j = best.get(key)
        if j is None or not iserr or parsed[j][1]:
            best[key] = i
    keep = set()
    for key, i in best.items():
        if parsed[i][1] and retry_errors:
            stats["errors_to_retry"] += 1
        else:
            keep.add(i)
            stats["errors_kept"] += int(parsed[i][1])
    for i, (key, iserr, raw) in enumerate(parsed):
        if i not in keep:
            moved.append(raw)
            if best[key] != i:
                stats["superseded"] += 1
    if moved:
        with open(badpath, "a", encoding="utf-8") as fh:
            for raw in moved:
                fh.write(raw.rstrip("\r") + "\n")
        tmp = path + ".tmp%d" % os.getpid()
        with open(tmp, "w", encoding="utf-8") as fh:
            for i in sorted(keep):
                fh.write(parsed[i][2].rstrip("\r") + "\n")
        os.replace(tmp, path)
        stats["rewritten"] = True
    done = {parsed[i][0] for i in keep}
    return done, stats


def ensuretrailnl(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb") as fh:
        fh.seek(-1, os.SEEK_END)
        last = fh.read(1)
    if last != b"\n":
        with open(path, "ab") as fh:
            fh.write(b"\n")


def errrec(sid, label, setting, tb):
    return {"sid": sid, "setting_label": label,
            "solver": setting.get("solver") if isinstance(setting, dict) else None,
            "indverify_version": __version__, "error": tb}


def worker(case, setting, branches, fine_dt, histfile, sid=None, label=None):
    if sid is None:
        sid = casesid(case, 0)
    if label is None:
        label = setlabel(setting, 0)
    try:
        keep = histfile is not None
        summary, hist = runcase(case, setting, branches=branches, fine_dt=fine_dt,
                                keep_history=keep)
        if keep:
            tmp = "%s.tmp%d.npz" % (histfile[:-4], os.getpid())
            np.savez_compressed(tmp, **hist)
            os.replace(tmp, histfile)
            summary["history_file"] = histfile
        return summary
    except Exception:
        return errrec(sid, label, setting, traceback.format_exc())


def buildparser():
    p = argparse.ArgumentParser(prog="python -m indverify")
    p.add_argument("--cases", required=True)
    p.add_argument("--settings", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--branches", default="ref,cw,so")
    p.add_argument("--history-setting", default=None)
    p.add_argument("--fine-dt", type=float, default=FINEDT)
    p.add_argument("--no-retry-errors", dest="retry_errors", action="store_false")
    p.add_argument("--retry-errors", dest="retry_errors", action="store_true")
    p.set_defaults(retry_errors=True)
    return p
def main(argv=None):
    args = buildparser().parse_args(argv)
    cas = loadjsonlist(args.cases, "--cases")
    settings = loadjsonlist(args.settings, "--settings")
    branches = normbranch(args.branches)
    labels = [setlabel(s, j) for j, s in enumerate(settings)]
    if len(set(labels)) != len(labels):
        raise SystemExit("setting labels must be unique: %s" % labels)
    sids = [casesid(c, i) for i, c in enumerate(cas)]
    if len(set(sids)) != len(sids):
        dup = sorted(k for k, v in Counter(sids).items() if v > 1)
        raise SystemExit("scenario sids must be unique (duplicates: %s)" % dup)
    nbadcases = sum(1 for i, c in enumerate(cas) if sids[i] == "#%d" % i)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    histdir = os.path.join(out, HISTDIR)
    histnames = {}
    if args.history_setting is not None:
        os.makedirs(histdir, exist_ok=True)
        histnames = histfnames(sids)
    respath = os.path.join(out, RESNAME)
    badpath = os.path.join(out, BADNAME)
    done, cstats = compactres(respath, badpath, args.retry_errors)
    if cstats["rewritten"] or cstats["errors_kept"]:
        print("resume: %d line(s) read; %d unparseable and %d superseded line(s) moved to %s; "
              "%d error record(s) %s"
              % (cstats["lines"], cstats["unparseable"], cstats["superseded"], badpath,
                 cstats["errors_to_retry"] + cstats["errors_kept"],
                 "kept (--no-retry-errors)" if not args.retry_errors
                 else "moved there and re-run"), flush=True)

    tasks = []
    for case, sid in zip(cas, sids):
        for st, label in zip(settings, labels):
            if (sid, label) in done:
                continue
            hist = None
            if args.history_setting is not None and label == args.history_setting:
                hist = os.path.join(histdir, histnames[sid])
            tasks.append((case, st, sid, label, hist))
    nskip = len(cas) * len(settings) - len(tasks)
    print("indverify %s: %d task(s) to run, %d already present in %s%s"
          % (__version__, len(tasks), nskip, respath,
             (" (%d malformed case(s) without sid)" % nbadcases) if nbadcases else ""),
          flush=True)
    if not tasks:
        return 0

    for var in THREADVARS:
        os.environ.setdefault(var, "1")
    ensuretrailnl(respath)
    t0 = time.perf_counter()
    workers = max(1, int(args.workers))
    nerr = 0
    with open(respath, "a", encoding="utf-8") as fh, \
            ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for case, st, sid, label, hist in tasks:
            fut = ex.submit(worker, case, st, branches, args.fine_dt, hist, sid, label)
            futs[fut] = (sid, label, st)
        for i, fut in enumerate(as_completed(futs), 1):
            sid, label, st = futs[fut]
            try:
                rec = fut.result()
            except BaseException:  
                rec = errrec(sid, label, st, traceback.format_exc())
            fh.write(json.dumps(jsonable(rec), allow_nan=True) + "\n")
            fh.flush()
            if "error" in rec:
                nerr += 1
                msg = "ERROR: " + rec["error"].strip().splitlines()[-1]
            else:
                stats = ["%s=%s" % (b, rec.get(b + "_status")) for b in branches]
                msg = "%s wall=%.1fs" % (" ".join(stats), rec.get("wall_s", math.nan))
            print("[%d/%d] %s %s %s (elapsed %.0fs)"
                  % (i, len(tasks), sid, label, msg, time.perf_counter() - t0), flush=True)
    print("done: %d task(s), %d error(s)%s"
          % (len(tasks), nerr,
             "; re-run the same command to retry them" if nerr and args.retry_errors else ""),
          flush=True)
    return 0


if __name__ == "__main__":  
    sys.exit(main())
