"""Configuration-driven batch runner with checkpoints, resumability and a manifest.

Usage (from the repository root)::

    python -m cwvalid.runner <campaign.json> [--workers 7] [--limit N]

A campaign JSON names an output directory, a scenario list file (JSONL of
:class:`cwvalid.scenario.Scenario` dicts) and a :class:`RunConfig` override.
Results are appended one JSON line per scenario to ``results.jsonl`` by the
parent process only (single writer); completed scenario IDs are skipped on
restart. Exceptions are captured per scenario and recorded as failed runs in
``failures.jsonl``; nothing is silently discarded. ``manifest.json`` records
counts of completed / failed / unexecuted scenarios plus software versions.
"""

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
from .experiment import RunConfig, run_scenario
from .metrics import OutcomeCriteria
from .scenario import Scenario
from .simulate import EciConfig, SolverConfig
from .dynamics import InertialForce


def software_versions():
    import pandas
    import scipy
    import matplotlib
    return {"python": sys.version, "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pandas.__version__, "matplotlib": matplotlib.__version__,
            "cwvalid": __version__}


def _jsonable(v):
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None if math.isnan(v) else ("inf" if v > 0 else "-inf")
    return v


def config_from_dict(d):
    d = dict(d or {})
    solver = SolverConfig(**d.pop("solver", {}))
    crit = OutcomeCriteria(**d.pop("crit", {}))
    eci_d = d.pop("eci", None)
    eci = EciConfig()
    if eci_d:
        eci = EciConfig(inclination=eci_d.get("inclination", eci.inclination),
                        force_target=InertialForce(**eci_d.get("force_target", {})),
                        force_chaser=InertialForce(**eci_d.get("force_chaser", {})),
                        exact_frame_rate=eci_d.get("exact_frame_rate", True))
    if "models" in d:
        d["models"] = tuple(d["models"])
    return RunConfig(solver=solver, crit=crit, eci=eci, **d)


def config_hash(cfg_dict):
    import hashlib
    return hashlib.sha256(json.dumps(cfg_dict, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _read_jsonl(path, log=print):
    """Read a JSONL file, skipping (and reporting) unparsable lines such as a
    partial last line left by an interrupted write."""
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
    scn_d, cfg_d, hist_dir = args
    scn = Scenario.from_dict(scn_d)
    cfg = config_from_dict(cfg_d)
    keep = bool(scn.meta.get("keep_history", False)) and hist_dir is not None
    stride = int(scn.meta.get("hist_stride", 4))
    t0 = time.perf_counter()
    try:
        row, hist = run_scenario(scn, cfg, keep_histories=keep, hist_stride=stride)
        row["run_wall_s"] = time.perf_counter() - t0
        row["config_hash"] = config_hash(cfg_d)
        if keep and hist is not None:
            np.savez_compressed(os.path.join(hist_dir, f"{scn.sid}.npz"), **hist)
            row["history_file"] = f"{scn.sid}.npz"
        return {"ok": True, "row": {k: _jsonable(v) for k, v in row.items()}}
    except Exception:
        return {"ok": False, "sid": scn.sid, "family": scn.family,
                "error": traceback.format_exc(), "wall_s": time.perf_counter() - t0}


def run_campaign(out_dir, scenarios, cfg_dict, workers=7, limit=None, keep_hist=True, log=print):
    os.makedirs(out_dir, exist_ok=True)
    hist_dir = os.path.join(out_dir, "histories") if keep_hist else None
    if hist_dir:
        os.makedirs(hist_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "results.jsonl")
    fail_path = os.path.join(out_dir, "failures.jsonl")
    done = set()
    if os.path.exists(res_path):
        done = {r["sid"] for r in _read_jsonl(res_path, log) if "sid" in r}
        # make sure the file ends with a newline before appending (interrupted writes)
        with open(res_path, "rb+") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() > 0:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    fh.write(b"\n")
    # configuration guard: refuse to mix settings within one results file
    h = config_hash(cfg_dict)
    rc_path = os.path.join(out_dir, "run_config.json")
    if os.path.exists(rc_path) and done:
        prev = json.load(open(rc_path))
        prev_h = prev.get("config_hash") or config_hash(prev.get("config", {}))
        if prev_h != h:
            raise RuntimeError(f"{out_dir}: configuration differs from the one used for existing results "
                               f"({prev_h} vs {h}); use a new output directory")
    with open(rc_path, "w") as fh:
        json.dump({"config": cfg_dict, "config_hash": h, "software": software_versions()}, fh, indent=2, default=str)
    all_ids = [s["sid"] if isinstance(s, dict) else s.sid for s in scenarios]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("duplicate scenario IDs in campaign")
    todo = [s if isinstance(s, dict) else s.to_dict() for s in scenarios]
    todo = [s for s in todo if s["sid"] not in done]
    if limit is not None:
        todo = todo[:limit]
    log(f"[runner] {out_dir}: {len(all_ids)} scenarios, {len(done)} already done, {len(todo)} to run")
    n_ok = n_fail = 0
    t_start = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex, \
                open(res_path, "a") as fres, open(fail_path, "a") as ffail:
            futs = [ex.submit(_work, (s, cfg_dict, hist_dir)) for s in todo]
            for k, fut in enumerate(as_completed(futs)):
                r = fut.result()
                if r["ok"]:
                    fres.write(json.dumps(r["row"]) + "\n")
                    fres.flush()
                    n_ok += 1
                else:
                    ffail.write(json.dumps(r) + "\n")
                    ffail.flush()
                    n_fail += 1
                if (k + 1) % 100 == 0:
                    el = time.perf_counter() - t_start
                    log(f"[runner] {k + 1}/{len(todo)} done ({n_fail} failed), {el:.0f}s elapsed")
    write_manifest(out_dir, all_ids)
    log(f"[runner] finished: {n_ok} ok, {n_fail} failed, {time.perf_counter() - t_start:.1f}s")
    return n_ok, n_fail


def write_manifest(out_dir, all_ids):
    res_path = os.path.join(out_dir, "results.jsonl")
    fail_path = os.path.join(out_dir, "failures.jsonl")
    ok, failed = set(), set()
    num_fail_rows = set()
    if os.path.exists(res_path):
        for r in _read_jsonl(res_path):
            ok.add(r["sid"])
            if r.get("ambiguous") and "numerical_failure" in (r.get("ambiguity_flags") or ""):
                num_fail_rows.add(r["sid"])
    if os.path.exists(fail_path):
        for r in _read_jsonl(fail_path):
            failed.add(r["sid"])
    failed -= ok
    unexec = [s for s in all_ids if s not in ok and s not in failed]
    man = {"total": len(all_ids), "completed": len(ok), "failed_exception": len(failed),
           "completed_with_branch_numerical_failure": len(num_fail_rows),
           "unexecuted": len(unexec), "failed_ids": sorted(failed), "unexecuted_ids": unexec,
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=2)
    return man


def load_results(out_dir):
    import pandas as pd
    rows = _read_jsonl(os.path.join(out_dir, "results.jsonl"))
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
    out_dir = os.path.join(base, camp["out_dir"]) if not os.path.isabs(camp["out_dir"]) else camp["out_dir"]
    scen_file = os.path.join(base, camp["scenarios"]) if not os.path.isabs(camp["scenarios"]) else camp["scenarios"]
    scenarios = [json.loads(l) for l in open(scen_file) if l.strip()]
    run_campaign(out_dir, scenarios, camp.get("config", {}), workers=a.workers, limit=a.limit,
                 keep_hist=camp.get("keep_histories", True))


if __name__ == "__main__":
    main()
