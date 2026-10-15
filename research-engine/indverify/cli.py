"""Command-line entry point (SPEC section 9).

python -m indverify --cases CASES.json --settings SETTINGS.json --out OUTDIR
                    [--workers 7] [--branches ref,cw,so] [--history-setting LABEL]
                    [--fine-dt 0.01] [--no-retry-errors]

Writes OUTDIR/indverify_results.jsonl (one line per (case, setting), appended
as each finishes) and, for the setting whose label equals --history-setting,
OUTDIR/histories/<sid>.npz.

Resume.  On start the results file is compacted to one record per
(sid, setting_label): the last successful record if there is one, otherwise
the last error record.  Pairs with a successful record are skipped.  Pairs
whose only record is an error are re-run by default (a crashed worker turns
every pending task into an error line, see BrokenProcessPool); their error
record is moved out of the results file first, so a pair never has two
records.  --no-retry-errors keeps error records and skips those pairs.
Unparseable lines (e.g. a line truncated by a crash mid-write) and superseded
records are moved to OUTDIR/indverify_results.bad, never silently deleted.

Malformed input.  A case without a usable 'sid' gets the synthetic sid
'#<index>' and a setting that cannot be parsed gets its 'label' or
'#setting<index>'; both are run and produce error records, so no
(case, setting) pair silently disappears.  Only duplicate sids or labels
abort the run.

History files.  <sid>.npz when the sid is a safe file name; otherwise (or if
two sids would collide, also case-insensitively as on Windows) the sanitised
sid plus a short hash of the raw sid, so no history overwrites another.
"""

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

from ._version import __version__
from .constants import FINE_DT
from .runner import SolverSetting, normalise_branches, run_case

RESULTS_NAME = "indverify_results.jsonl"
BAD_NAME = "indverify_results.bad"
HISTORY_DIR = "histories"
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_UNSAFE = re.compile(r"[^A-Za-z0-9._+=-]")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# Identifiers and history file names
# ---------------------------------------------------------------------------
def case_sid(case, index):
    """The case's sid, or '#<index>' if it has none (malformed case)."""
    if isinstance(case, dict) and case.get("sid") is not None:
        return str(case["sid"])
    return "#%d" % index


def setting_label(setting, index):
    """The setting's label; for an unparseable setting its 'label' or '#setting<index>'."""
    try:
        return SolverSetting.from_dict(setting).label
    except Exception:  # noqa: BLE001  (reported per case by the worker)
        if isinstance(setting, dict) and setting.get("label"):
            return str(setting["label"])
        return "#setting%d" % index


def _sid_hash(sid):
    return hashlib.sha1(str(sid).encode("utf-8")).hexdigest()[:10]


def history_filenames(sids):
    """Map every sid to a distinct history file name (distinct also when
    compared case-insensitively).  A sid that is already a safe file name and
    does not collide keeps the name '<sid>.npz'; otherwise the sanitised sid
    gets '_<hash of the raw sid>' appended."""
    sids = [str(s) for s in sids]
    safe = {s: _UNSAFE.sub("_", s) for s in sids}
    counts = Counter(v.lower() for v in safe.values())
    out = {}
    for s in sids:
        name = safe[s]
        if name != s or counts[name.lower()] > 1 or not name or name.startswith("."):
            name = "%s_%s" % (name, _sid_hash(s))
        out[s] = name + ".npz"
    lowered = [v.lower() for v in out.values()]
    if len(set(lowered)) != len(lowered):  # pragma: no cover - needs a hash collision
        raise SystemExit("history file names collide; rename the scenarios")
    return out


def history_filename(sid):
    """History file name of a single sid (see history_filenames)."""
    return history_filenames([sid])[str(sid)]


# ---------------------------------------------------------------------------
# Results file: loading, compaction and resume
# ---------------------------------------------------------------------------
def _load_json_list(path, what):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise SystemExit("%s must contain a JSON list" % what)
    return data


def _record_key(rec):
    return str(rec.get("sid")), str(rec.get("setting_label"))


def compact_results(path, bad_path, retry_errors=True):
    """Reduce the results file to one record per (sid, setting_label).

    Kept per pair: the last successful record, else the last error record
    (dropped instead when ``retry_errors``, because the pair is about to be
    re-run).  Unparseable lines, superseded records and dropped error records
    are appended to ``bad_path``; the results file is rewritten atomically
    only if something was moved.  Returns (done_pairs, stats).
    """
    stats = {"lines": 0, "unparseable": 0, "superseded": 0, "errors_to_retry": 0,
             "errors_kept": 0, "rewritten": False}
    if not os.path.exists(path):
        return set(), stats
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        raw_lines = [ln for ln in fh.read().split("\n") if ln.strip()]
    stats["lines"] = len(raw_lines)
    parsed, moved = [], []
    for raw in raw_lines:
        try:
            rec = json.loads(raw)
            if not isinstance(rec, dict):
                raise ValueError("not a JSON object")
        except ValueError:
            moved.append(raw)
            stats["unparseable"] += 1
            continue
        parsed.append((_record_key(rec), "error" in rec, raw))
    best = {}
    for i, (key, is_err, _) in enumerate(parsed):
        j = best.get(key)
        if j is None or not is_err or parsed[j][1]:
            best[key] = i          # a success beats anything earlier; an error only an error
    keep = set()
    for key, i in best.items():
        if parsed[i][1] and retry_errors:
            stats["errors_to_retry"] += 1
        else:
            keep.add(i)
            stats["errors_kept"] += int(parsed[i][1])
    for i, (key, is_err, raw) in enumerate(parsed):
        if i not in keep:
            moved.append(raw)
            if best[key] != i:
                stats["superseded"] += 1
    if moved:
        with open(bad_path, "a", encoding="utf-8") as fh:
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


def _ensure_trailing_newline(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb") as fh:
        fh.seek(-1, os.SEEK_END)
        last = fh.read(1)
    if last != b"\n":
        with open(path, "ab") as fh:
            fh.write(b"\n")


# ---------------------------------------------------------------------------
# Worker and main
# ---------------------------------------------------------------------------
def _error_record(sid, label, setting, tb):
    return {"sid": sid, "setting_label": label,
            "solver": setting.get("solver") if isinstance(setting, dict) else None,
            "indverify_version": __version__, "error": tb}


def _worker(case, setting, branches, fine_dt, hist_file, sid=None, label=None):
    """Run one (case, setting); never raises (errors become an 'error' record)."""
    if sid is None:
        sid = case_sid(case, 0)
    if label is None:
        label = setting_label(setting, 0)
    try:
        keep = hist_file is not None
        summary, hist = run_case(case, setting, branches=branches, fine_dt=fine_dt,
                                 keep_history=keep)
        if keep:
            tmp = "%s.tmp%d.npz" % (hist_file[:-4], os.getpid())
            np.savez_compressed(tmp, **hist)
            os.replace(tmp, hist_file)
            summary["history_file"] = hist_file
        return summary
    except Exception:  # noqa: BLE001
        return _error_record(sid, label, setting, traceback.format_exc())


def build_parser():
    p = argparse.ArgumentParser(prog="python -m indverify",
                                description="Independent closed-loop verifier (indverify/SPEC.md)")
    p.add_argument("--cases", required=True, help="JSON list of scenario dicts")
    p.add_argument("--settings", required=True, help="JSON list of solver-setting dicts")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--branches", default="ref,cw,so")
    p.add_argument("--history-setting", default=None,
                   help="label of the setting whose histories are written")
    p.add_argument("--fine-dt", type=float, default=FINE_DT)
    p.add_argument("--no-retry-errors", dest="retry_errors", action="store_false",
                   help="keep error records on resume and do not re-run those pairs")
    p.add_argument("--retry-errors", dest="retry_errors", action="store_true",
                   help="re-run pairs whose only record is an error (default)")
    p.set_defaults(retry_errors=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cases = _load_json_list(args.cases, "--cases")
    settings = _load_json_list(args.settings, "--settings")
    branches = normalise_branches(args.branches)
    labels = [setting_label(s, j) for j, s in enumerate(settings)]
    if len(set(labels)) != len(labels):
        raise SystemExit("setting labels must be unique: %s" % labels)
    sids = [case_sid(c, i) for i, c in enumerate(cases)]
    if len(set(sids)) != len(sids):
        dup = sorted(k for k, v in Counter(sids).items() if v > 1)
        raise SystemExit("scenario sids must be unique (duplicates: %s)" % dup)
    n_bad_cases = sum(1 for i, c in enumerate(cases) if sids[i] == "#%d" % i)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    hist_dir = os.path.join(out, HISTORY_DIR)
    hist_names = {}
    if args.history_setting is not None:
        os.makedirs(hist_dir, exist_ok=True)
        hist_names = history_filenames(sids)
    res_path = os.path.join(out, RESULTS_NAME)
    bad_path = os.path.join(out, BAD_NAME)
    done, cstats = compact_results(res_path, bad_path, args.retry_errors)
    if cstats["rewritten"] or cstats["errors_kept"]:
        print("resume: %d line(s) read; %d unparseable and %d superseded line(s) moved to %s; "
              "%d error record(s) %s"
              % (cstats["lines"], cstats["unparseable"], cstats["superseded"], bad_path,
                 cstats["errors_to_retry"] + cstats["errors_kept"],
                 "kept (--no-retry-errors)" if not args.retry_errors
                 else "moved there and re-run"), flush=True)

    tasks = []
    for case, sid in zip(cases, sids):
        for st, label in zip(settings, labels):
            if (sid, label) in done:
                continue
            hist = None
            if args.history_setting is not None and label == args.history_setting:
                hist = os.path.join(hist_dir, hist_names[sid])
            tasks.append((case, st, sid, label, hist))
    n_skip = len(cases) * len(settings) - len(tasks)
    print("indverify %s: %d task(s) to run, %d already present in %s%s"
          % (__version__, len(tasks), n_skip, res_path,
             (" (%d malformed case(s) without sid)" % n_bad_cases) if n_bad_cases else ""),
          flush=True)
    if not tasks:
        return 0

    for var in _THREAD_VARS:
        os.environ.setdefault(var, "1")
    _ensure_trailing_newline(res_path)
    t0 = time.perf_counter()
    workers = max(1, int(args.workers))
    n_err = 0
    with open(res_path, "a", encoding="utf-8") as fh, \
            ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for case, st, sid, label, hist in tasks:
            fut = ex.submit(_worker, case, st, branches, args.fine_dt, hist, sid, label)
            futs[fut] = (sid, label, st)
        for i, fut in enumerate(as_completed(futs), 1):
            sid, label, st = futs[fut]
            try:
                rec = fut.result()
            except BaseException:  # noqa: BLE001  (e.g. a crashed worker process)
                rec = _error_record(sid, label, st, traceback.format_exc())
            fh.write(json.dumps(_jsonable(rec), allow_nan=True) + "\n")
            fh.flush()
            if "error" in rec:
                n_err += 1
                msg = "ERROR: " + rec["error"].strip().splitlines()[-1]
            else:
                stats = ["%s=%s" % (b, rec.get(b + "_status")) for b in branches]
                msg = "%s wall=%.1fs" % (" ".join(stats), rec.get("wall_s", math.nan))
            print("[%d/%d] %s %s %s (elapsed %.0fs)"
                  % (i, len(tasks), sid, label, msg, time.perf_counter() - t0), flush=True)
    print("done: %d task(s), %d error(s)%s"
          % (len(tasks), n_err,
             "; re-run the same command to retry them" if n_err and args.retry_errors else ""),
          flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
