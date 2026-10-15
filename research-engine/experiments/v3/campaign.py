"""v3 campaign driver: manifests, propagation, per-case scalar records and the run ledger.

  plan   --cell D0 --split pilot --n 40     write the case manifest (no propagation, no labels)
  run    --cell D0 --split pilot            propagate every planned case not yet completed (resumable)
  plan-cohort --cell D0 --split dev --targets D3 D4 D5   transfer-diagnostic cohort (paired, exploratory)
  status                                    reconcile manifests, results and ledger for every cell/split

Continuous cells (D0-D2 scalar PD, D5 matrix LQR): branches cw, eci (reference), nerm (cross-check)
with the production solver configuration; PD cells call the production cwvalid functions with a
PDLaw, the matrix cell calls cwvalid3.continuous (verified bit-identical to production for a PD
matrix). Features: A = d1, B = d1u, C = cres (production definitions, 2 s output grid).
Held cells (D3 Ts = 1 s, D4 Ts = 10 s): cwvalid3.sampled.HeldBatch, branches cw, nerm, eci with
held LVLH commands (own-state updates at t_k = k Ts), step h = min(Ts, 2 s); estimators A_H, B_H,
C_H in the same pass; A_naive in a separate CW-only pass with h = 0.5 s. Features are max |e_pos|
on the 0.5 s monitoring grid.

Outputs (archive root/v3_extension):
  raw/manifests/<cell>_<split>.jsonl      planned cases (case_id, physical_id, run_id, scenario, K)
  raw/results/<cell>_<split>.jsonl        one scalar record per completed run (strictly parsed)
  RUN_LEDGER.jsonl                        planned and completed events with hashes
Usage: python code/experiments/v3/campaign.py plan|run|status ...

v3.1 corrections (v3_extension/v3_1_correction/reports/CORRECTION_REPORT.md):
  C02  workers are staged; 'failed_stage' names the stage of an exception. Continuous cells: the nominal CW branch
       and the screening features are computed, stored in the record and written to a per-case checkpoint before any
       reference/cross-check propagation. Held cells: the nominal and reference branches are propagated together in
       one combined batch, so held features are NOT checkpointed before reference propagation. After that batch the
       primary features (CW branch, A_H, B_H, C_H, saturation) are computed and checkpointed before the optional
       A_naive pass and before any reference metric (v3.1.1 R01/R02). Each feature is isolated: a failure of one
       estimator, of the saturation summary or of the optional A_naive comparator (its pass, case view, extraction or
       diagnostic) removes only that feature or diagnostic and is listed in feature_errors. An exception after the
       feature stage gives worker_status 'reference_failed' (features kept, truth unresolved). If a worker process is
       lost, the pool breaks and every unfinished task fails with it, including tasks that never started; cmd_run
       re-executes each such task alone in a fresh process (task_retry), so only a task that is lost again on its own
       becomes worker_crash. Its cases get a checkpoint written in the same run session by the same engine
       (feature_recovery 'checkpoint') or, without one, a nominal-only recovery without reference branches in a fresh
       process (feature_recovery 'driver_nominal_recovery'). That is post-failure recovery, not pre-reference
       checkpointing, and truth stays 'interrupted' (adverse if accepted). A failed checkpoint write is recorded
       (feature_checkpoint_error) and never fails the case.
       Held cells: a batch that raises is retried case by case; if the combined batch (or its case view) still fails
       for one case, the features come from a CW + estimator batch for that case alone and the case is
       reference_failed. Held records store batch_members_sha256 (batch composition can move results by an ulp).
       worker_crash stays an operational status.
  C03  records and ledger events are written in the canonical v3.1 form (cwvalid3.serialization) and the
       ledger stores the SHA-256 of the exact stored line (record_sha256) plus physics_sha256. The engine
       identity is path-based and covers every repository module the workers import (ENGINE_FILES_V31).
  The v3.0 campaign tree (raw/manifests, raw/results, RUN_LEDGER.jsonl) is closed: plan and run write
  under --out (default v3_1_correction/raw/campaign) and refuse to write into the v3.0 tree.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
ARCH = os.path.dirname(REPO)
V3 = os.path.join(ARCH, "v3_extension")
MAN = os.path.join(V3, "raw", "manifests")
RES = os.path.join(V3, "raw", "results")
LEDGER = os.path.join(V3, "RUN_LEDGER.jsonl")
OUT_DEFAULT = os.path.join(V3, "v3_1_correction", "raw", "campaign")

import numpy as np  # noqa: E402

from cwvalid3 import ledger as L  # noqa: E402
from cwvalid3 import serialization as SER  # noqa: E402
from cwvalid3.domains import CELLS, make_cases  # noqa: E402

# v3.0 engine identity (basename-based; kept to compare with the stored engine_sha of v3.0 records)
ENGINE_FILES = [os.path.join(REPO, p) for p in (
    "cwvalid/simulate.py", "cwvalid/dynamics.py", "cwvalid/metrics.py", "cwvalid/experiment.py", "cwvalid/control.py",
    "cwvalid/estimate.py", "cwvalid/estimate_v2.py", "cwvalid/scenario.py", "cwvalid/constants.py",
    "cwvalid3/continuous.py", "cwvalid3/controllers.py", "cwvalid3/sampled.py", "cwvalid3/held_metrics.py",
    "cwvalid3/domains.py", "cwvalid3/ledger.py", "experiments/v3/campaign.py")]
# v3.1 engine identity: every repository module imported by the workers and the record writer
ENGINE_FILES_V31 = sorted(set(ENGINE_FILES) | {os.path.join(REPO, p) for p in (
    "cwvalid/__init__.py", "cwvalid3/__init__.py", "cwvalid3/serialization.py", "src/__init__.py", "src/collision.py",
    "src/controller.py", "src/cw.py", "src/external_validation.py", "src/lvlh.py", "src/orbit.py",
    "src/perturbations.py", "src/plotting.py", "src/scenarios.py", "src/validation.py")})


def _inside(path, root):
    """True if path is root or below it (case-normalised real paths; safe on case-insensitive filesystems)."""
    p, r = (os.path.normcase(os.path.realpath(os.path.abspath(x))) for x in (path, root))
    try:
        return os.path.commonpath([p, r]) == r
    except ValueError:      # different drives
        return False


def v30_tree_guard(path):
    """The v3.0 campaign tree is closed: refuse to create or append anything inside v3_extension
    except under v3_1_correction."""
    ap = os.path.abspath(path)
    if ap.startswith("\\\\"):         # \\?\ and UNC spellings can alias the protected trees (v3.1.1)
        raise SystemExit(f"refusing a UNC or extended-length path: {ap}")
    if _inside(ap, V3) and not _inside(ap, os.path.join(V3, "v3_1_correction")):
        raise SystemExit(f"refusing to write into the closed v3.0 tree: {ap}")
    return ap


def out_paths(out):
    out = v30_tree_guard(out)
    return {"man": os.path.join(out, "manifests"), "res": os.path.join(out, "results"),
            "ledger": os.path.join(out, "RUN_LEDGER.jsonl")}
BRANCH_KEYS = ("status", "termination", "message", "t_final_s", "nfev", "nsteps", "wall_time_s", "n_sat_switch_events",
               "sat_events_disabled_at_s", "completed_horizon", "numerical_failure", "terminal_inside_at_end",
               "terminal_arrival_time_s", "terminal_dwell_margin_s", "terminal_arrived", "final_pos_m",
               "final_speed_mps", "max_range_m", "max_range_over_R", "sat_total_s", "sat_fraction", "sat_n_intervals",
               "sat_initial_s", "peak_raw_acc_mps2", "dv_aug_mps")
PAIR_KEYS = ("max_pos_disc_m", "max_pos_disc_grid_m", "t_max_pos_disc_s", "final_pos_disc_m", "common_horizon_s",
             "max_vel_disc_mps")


def engine_identity():
    return SER.engine_identity(ENGINE_FILES_V31, REPO)


def engine_version():
    """v3.1 engine identity (16 hex of the path-based SHA-256); binds new run_ids."""
    return engine_identity()["sha16"]


def engine_version_v30():
    """The v3.0 formula applied to the current files (differs from v3.0 records once any file changed)."""
    return L.code_version(ENGINE_FILES)


def software():
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "engine_sha": engine_version(), "engine_id_scheme": SER.ENGINE_ID_SCHEME,
            "engine_sha_v30_formula": engine_version_v30()}


def run_config(cell):
    from cwvalid.metrics import OutcomeCriteria
    from cwvalid.simulate import SolverConfig
    c = CELLS[cell]
    base = {"cell": cell, "chief": "circular two-body", "reference": "eci", "crosscheck": "nerm",
            "models": ["cw", "eci", "nerm"], "crit": OutcomeCriteria().as_dict(), "monitor_dt_s": 0.5,
            "altitude_floor_m": 100000.0, "engine_sha": engine_version()}
    if c["cmd_mode"] == "continuous":
        base.update({"cmd_mode": "continuous", "Ts": None, "phase_s": None, "solver": SolverConfig().as_dict(),
                     "engine": "cwvalid (PD) / cwvalid3.continuous (matrix)",
                     "features": {"A": "d1: e'=(A+B J_U(s0))e+B r, 2 s grid", "B": "d1u: J_U=-K", "C": "cres",
                                  "sep": "|rho0|", "sat": "CW saturated time / T"}})
    else:
        Ts = c["Ts"]
        base.update({"cmd_mode": "sampled", "Ts": Ts, "phase_s": 0.0, "first_update_s": 0.0,
                     "engine": "cwvalid3.sampled.HeldBatch (DOP853 tableau, fixed step)",
                     "h_main_s": min(Ts, 2.0), "h_naive_s": 0.5,
                     "features": {"A_H": "held memory J_U(s0(t_k)) e(t_k)", "B_H": "-K e(t_k)",
                                  "A_naive": "continuous formula on held path", "C_H": "residual only",
                                  "sep": "|rho0|", "sat": "fraction of held time with saturated command"}})
    return base


def physics_keys_archive():
    """Physics keys of every archived scenario (same rounding as experiments/v2/fresh_test.py)."""
    import glob
    keys = set()
    for p in glob.glob(os.path.join(ARCH, "results", "**", "scenarios.jsonl"), recursive=True):
        for line in open(p):
            if line.strip():
                s = json.loads(line)
                keys.add((tuple(round(float(v), 9) for v in s["rel0"]), round(float(s["wn_ratio"]), 9),
                          None if s["a_max"] is None else round(float(s["a_max"]), 9),
                          float(s.get("horizon_orbits", 1.0)), float(s.get("altitude", 400000.0))))
    return keys


def cmd_plan(cell, split, N, out=OUT_DEFAULT):
    P = out_paths(out)
    os.makedirs(P["man"], exist_ok=True)
    path = os.path.join(P["man"], f"{cell}_{split}.jsonl")
    if os.path.exists(path):
        sys.exit(f"manifest {path} exists; refusing to replace a planned set")
    if os.path.exists(os.path.join(MAN, f"{cell}_{split}.jsonl")):
        sys.exit(f"{cell}/{split} names a v3.0 campaign split; plan new work under a new split name")
    recs, rng_info = make_cases(cell, split, N)
    cfg = run_config(cell)
    keys = physics_keys_archive()
    overlap = []
    for r in recs:
        s = r["scenario"]
        k = (tuple(round(float(v), 9) for v in s["rel0"]), round(float(s["wn_ratio"]), 9), round(float(s["a_max"]), 9),
             float(s["horizon_orbits"]), float(s["altitude"]))
        if k in keys:
            overlap.append(r["case_id"])
        r["run_id"] = L.run_id(r["physical_id"], cfg)
    if overlap:
        sys.exit(f"{len(overlap)} planned cases coincide with archived scenarios: {overlap[:5]}")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in recs:
            SER.write_record(fh, r, schema=None)
    summ = {"cell": cell, "split": split, "n": len(recs), "rng": rng_info, "run_config": cfg,
            "manifest_sha256": L.file_sha256(path), "archived_overlap": 0, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_archived_physics_keys_checked": len(keys), "engine_identity": engine_identity()}
    json.dump(summ, open(os.path.join(P["man"], f"{cell}_{split}_summary.json"), "w"), indent=2)
    with open(P["ledger"], "a", encoding="utf-8", newline="\n") as fh:
        for r in recs:
            SER.write_record(fh, {"ledger_id": f"planned:{r['run_id']}", "event": "planned", "run_id": r["run_id"],
                                  "case_id": r["case_id"], "physical_id": r["physical_id"], "cell": cell,
                                  "split": split, "index": r["index"], "config_hash": L.stable_hash(cfg),
                                  "manifest_sha256": summ["manifest_sha256"], "time": summ["written"]}, schema=None)
    print(json.dumps({k: summ[k] for k in ("cell", "split", "n", "manifest_sha256")}))


def cmd_plan_cohort(source_cell, source_split, targets, out=OUT_DEFAULT):
    """Transfer-diagnostic cohort: the source cell's draws re-run under each target cell's command
    timing / controller at the same altitude (paired engineering cases; exploratory, never pooled with
    the iid splits)."""
    from cwvalid3.domains import make_law
    P = out_paths(out)
    os.makedirs(P["man"], exist_ok=True)
    sp = os.path.join(P["man"], f"{source_cell}_{source_split}.jsonl")
    src = L.read_jsonl_strict(sp if os.path.exists(sp) else os.path.join(MAN, f"{source_cell}_{source_split}.jsonl"),
                              key="run_id")
    for tgt in targets:
        c = CELLS[tgt]
        if c["altitude"] != CELLS[source_cell]["altitude"]:
            sys.exit("cohort targets must share the source altitude (same initial states and n)")
        path = os.path.join(P["man"], f"{tgt}_transfer_cohort.jsonl")
        if os.path.exists(path) or os.path.exists(os.path.join(MAN, f"{tgt}_transfer_cohort.jsonl")):
            sys.exit(f"{tgt} transfer cohort already planned (v3.0 or {path})")
        cfg = run_config(tgt)
        recs = []
        for r in src:
            law = make_law(tgt, r["n_radps"], r["draw"]["wn_ratio"], r["draw"]["a_max"])
            q = dict(r)
            q.update(cell=tgt, split="transfer_cohort", K=law.K.tolist(), controller=law.design, cmd_mode=c["cmd_mode"],
                     Ts=c["Ts"], source_case_id=r["case_id"], source_physical_id=r["physical_id"])
            q["physical_id"] = L.physical_id(r["draw"]["rel0"], c["altitude"], 1.0, law.K, r["draw"]["a_max"])
            q["case_id"] = "C" + L.stable_hash({"cohort": tgt, "source": r["case_id"]})
            q["run_id"] = L.run_id(q["physical_id"], cfg)
            recs.append(q)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for q in recs:
                SER.write_record(fh, q, schema=None)
        summ = {"cell": tgt, "split": "transfer_cohort", "n": len(recs), "source": f"{source_cell}_{source_split}",
                "run_config": cfg, "manifest_sha256": L.file_sha256(path), "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                "engine_identity": engine_identity()}
        json.dump(summ, open(os.path.join(P["man"], f"{tgt}_transfer_cohort_summary.json"), "w"), indent=2)
        with open(P["ledger"], "a", encoding="utf-8", newline="\n") as fh:
            for q in recs:
                SER.write_record(fh, {"ledger_id": f"planned:{q['run_id']}", "event": "planned", "run_id": q["run_id"],
                                      "case_id": q["case_id"], "physical_id": q["physical_id"], "cell": tgt,
                                      "split": "transfer_cohort", "index": q["index"],
                                      "config_hash": L.stable_hash(cfg), "manifest_sha256": summ["manifest_sha256"],
                                      "time": summ["written"]}, schema=None)
        print(json.dumps({k: summ[k] for k in ("cell", "split", "n", "manifest_sha256")}))


# --------------------------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------------------------


def _pick(row, prefix, keys):
    return {f"{prefix}_{k}": row.get(f"{prefix}_{k}") for k in keys}


def _propagate(scn, cfg, law, models):
    """Propagate the given branches with the production per-branch loop (branches are independent)."""
    from dataclasses import replace
    from cwvalid.control import PDLaw
    from cwvalid3.continuous import run_branches_general
    c = replace(cfg, models=tuple(models))
    if isinstance(law, PDLaw):
        from cwvalid.experiment import run_branches
        sims, _ = run_branches(scn, c)
    else:
        sims, _ = run_branches_general(scn, c, law)
    return sims


def _branch_fields(scn, cfg, sim, m):
    """The per-branch part of cwvalid.experiment.evaluate for one branch (depends on that branch only)."""
    from cwvalid import metrics as M
    T = scn.horizon
    tr = M.make_trace(sim, M.common_grid(T, cfg.dt_eval))
    row = {}
    for k, v in sim.summary().items():
        if k != "model":
            row[f"{m}_{k}"] = v
    for k, v in M.legacy_arrival(sim, T, cfg.crit).items():
        row[f"{m}_{k}"] = v
    for k, v in M.branch_diagnostics(sim, tr, T, cfg.crit).items():
        row[f"{m}_{k}"] = v
    return row


def _tb(n=3000):
    import traceback
    return traceback.format_exc()[-n:]


HELD_PAIRS = (("cw", "eci"), ("cw", "nerm"), ("nerm", "eci"))
CHECKPOINT_ENV = "CWVALID3_FEATURE_CHECKPOINT_DIR"
SESSION_ENV = "CWVALID3_RUN_SESSION"
FAULT_ENV = "CWVALID3_FAULT_INJECT"
_ENGINE_SHA = []


def _engine_sha_cached():
    if not _ENGINE_SHA:
        _ENGINE_SHA.append(engine_version())
    return _ENGINE_SHA[0]


def _fault(point, run_ids=()):
    """Test-only fault injection (v3.1.1), inactive unless CWVALID3_FAULT_INJECT lists the point, either bare (every
    task) or as point@run_id (only a task containing that case). A 'loss_*' point ends the process with os._exit (a
    real process loss, no cleanup) and acts only inside a worker process, never in the driver; any other point raises
    RuntimeError."""
    spec = os.environ.get(FAULT_ENV, "")
    items = spec.split(",") if spec else []
    if point not in items and not any(f"{point}@{rid}" in items for rid in run_ids):
        return
    if point.startswith("loss_"):
        import multiprocessing
        if multiprocessing.parent_process() is not None:
            os._exit(97)
        return
    raise RuntimeError(f"injected fault at {point}")


def _feature_checkpoint(out):
    """Persist the feature-stage fields of one case (C02) so that cmd_run can merge them into a worker_crash record if
    the worker process is lost later. The file carries the run session and the engine identity; cmd_run merges only a
    checkpoint written by the same run and the same code. Written atomically; a later call for the same case (held:
    after the optional A_naive pass) replaces it atomically. A failed write (e.g. a file held open by another program)
    is operational: it is recorded in feature_checkpoint_error and never fails the case (v3.1.1)."""
    d = os.environ.get(CHECKPOINT_ENV)
    if not d:
        return
    p = os.path.join(d, f"{out['run_id']}.json")
    try:
        os.makedirs(d, exist_ok=True)
        ck = {**out, "checkpoint_session": os.environ.get(SESSION_ENV), "checkpoint_engine_sha": _engine_sha_cached()}
        with open(p + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
            SER.write_record(fh, ck, schema=None)
        os.replace(p + ".tmp", p)
    except Exception as exc:
        out["feature_checkpoint_error"] = repr(exc)[:500]


# ------------------------------------------------------------------------------------------------ continuous

def _continuous_setup(rec):
    from cwvalid.control import PDLaw
    from cwvalid.experiment import RunConfig
    from cwvalid.scenario import Scenario
    from cwvalid3.controllers import MatrixLaw
    scn = Scenario.from_dict(rec["scenario"])
    if rec["controller"].get("rule") == "scalar_pd":
        law = PDLaw.from_bandwidth(scn.wn_ratio * scn.n, scn.a_max)
    else:
        law = MatrixLaw(rec["K"], scn.a_max, rec["controller"])
    cfg = RunConfig(models=("cw", "eci", "nerm"), reference="eci", crosscheck="nerm")
    fields = {"sep_m": scn.separation, "wn_ratio": scn.wn_ratio, "a_max": scn.a_max, "n_radps": scn.n, "R_m": scn.R,
              "horizon_s": scn.horizon, "nu": rec["draw"]["nu"], "axis": rec["draw"]["axis"], "velocity": rec["draw"]["velocity"]}
    return scn, law, cfg, fields


def _continuous_features(scn, cfg, law, cw_sim):
    """Feature-stage fields from the nominal CW branch: its branch diagnostics (which certify CW completion for every
    feature), the estimators A, B, C (each isolated inside estimator_variants) and the saturated-time fraction."""
    from cwvalid3.continuous import estimator_variants_general
    out = {}
    cwrow = _branch_fields(scn, cfg, cw_sim, "cw")
    out.update(_pick(cwrow, "cw", BRANCH_KEYS))
    out["cw_sat_switch_times"] = [float(t) for t in cw_sim.sat_switch_times][:200]
    feats = estimator_variants_general(cw_sim, law, scn.n, scn.R, scn.horizon)
    for k in ("d1", "d1u", "cres"):
        for f in ("peak_m", "t_peak_s", "status", "nfev", "failure_reason"):
            out[f"{k}_{f}"] = feats.get(f"{k}_{f}")
    out["feature_sat_fraction"] = (cwrow.get("cw_sat_total_s") / scn.horizon) if cwrow.get("cw_sat_total_s") is not None else None
    out["feature_stage"] = "complete"
    return out, cwrow


def work_continuous(rec):
    """Staged (v3.1 C02): setup -> nominal CW -> features (stored and checkpointed) -> reference/cross-check -> metrics."""
    from cwvalid3.continuous import evaluate_general
    c0, w0 = time.process_time(), time.perf_counter()
    out = {k: rec[k] for k in ("case_id", "physical_id", "run_id", "cell", "split", "index")}
    stage = "setup"
    try:
        scn, law, cfg, fields = _continuous_setup(rec)
        out.update(fields)
        stage = "nominal"
        t0 = time.perf_counter()
        sims = _propagate(scn, cfg, law, ("cw",))
        t1 = time.perf_counter()
        _fault("loss_continuous_before_features", [rec["run_id"]])
        stage = "features"
        feats, cwrow = _continuous_features(scn, cfg, law, sims["cw"])
        out.update(feats)
        _feature_checkpoint(out)
        _fault("loss_continuous_after_features", [rec["run_id"]])
        t2 = time.perf_counter()
        stage = "reference"
        sims.update(_propagate(scn, cfg, law, ("eci", "nerm")))
        sims = {m: sims[m] for m in cfg.models}
        t3 = time.perf_counter()
        stage = "metrics"
        row, _ = evaluate_general(scn, cfg, sims, law)
        t4 = time.perf_counter()
        if _pick(row, "cw", BRANCH_KEYS) != _pick(cwrow, "cw", BRANCH_KEYS) and \
                SER.normalize(_pick(row, "cw", BRANCH_KEYS)) != SER.normalize(_pick(cwrow, "cw", BRANCH_KEYS)):
            raise RuntimeError("CW branch fields differ between the feature stage and the full evaluation")
        out.update({"kp": row["kp"], "kd": row["kd"]})
        for b in ("cw", "eci", "nerm"):
            out.update(_pick(row, b, BRANCH_KEYS))
        for p in ("cw_vs_eci", "cw_vs_nerm", "nerm_vs_eci"):
            out.update(_pick(row, p, PAIR_KEYS))
        out["ambiguity_flags"] = row.get("ambiguity_flags", "")
        out.update({"wall_propagate_s": (t1 - t0) + (t3 - t2), "wall_propagate_nominal_s": t1 - t0,
                    "wall_evaluate_s": t4 - t3, "wall_features_s": t2 - t1, "worker_status": "ok"})
    except Exception:  # recorded, never dropped
        tb = _tb()
        if stage in ("reference", "metrics"):
            out.update({"worker_status": "reference_failed", "reference_error": tb, "failed_stage": stage})
        else:
            out.update({"worker_status": "exception", "error": tb, "failed_stage": stage})
    out["cpu_s"] = time.process_time() - c0
    out["wall_s"] = time.perf_counter() - w0
    return [out]


# ------------------------------------------------------------------------------------------------ held

def _held_primary_features(cv):
    """Primary feature-stage fields of one held case (v3.1.1 R01): the CW branch status and end time (straight from
    the propagation; they certify CW completion for every feature), the CW arrival metrics, each held estimator and
    the saturation summary. Each part is isolated: a failure removes only that part (estimator status
    'feature_extraction_failed'; no saturation fields, so the sat feature is unavailable; no CW arrival fields, so
    arrival is unresolved) and is recorded in the returned errors."""
    from cwvalid3.held_metrics import estimator_feature, saturation_summary, terminal_arrival
    b, i = cv.b, cv.i
    out, errors = {"cw_status": b.status["cw"][i], "cw_t_end_s": float(b.t_end["cw"][i])}, {}
    try:
        for k, v in terminal_arrival(cv, "cw").items():
            out[f"cw_{k}"] = v
    except Exception:
        errors["cw_arrival"] = _tb(2000)
    for e in b.estimators:
        try:
            fe = estimator_feature(cv, e)
        except Exception:
            errors[e] = _tb(2000)
            fe = {"peak_m": float("nan"), "t_peak_s": float("nan"), "status": "feature_extraction_failed"}
        for k, v in fe.items():
            out[f"{e}_{k}"] = v
    try:
        for k, v in saturation_summary(cv).items():
            out[f"cw_{k}"] = v
    except Exception:
        errors["saturation"] = _tb(2000)
    out["feature_sat_fraction"] = out.get("cw_sat_fraction")
    return out, errors


def _naive_pass_diagnostic(cvn, cvf):
    """Largest CW position difference between the A_naive pass (h = 0.5 s) and the main pass (diagnostic only)."""
    return float(np.max(np.linalg.norm(cvn.grid("cw")[:, :3] - cvf.grid("cw")[:, :3], axis=1)))


def _held_naive_fields(hn, i, cvf, err_hn):
    """Optional comparator A_naive and its diagnostic, isolated from every other feature (v3.1.1 R01): a failure of the
    pass, of its case view or of the feature extraction makes only A_naive unavailable; a failure of the naive-pass
    diagnostic only blanks the diagnostic."""
    from cwvalid3.held_metrics import CaseView, estimator_feature
    nan = float("nan")
    out, errors = {}, {}
    if hn is None:
        out.update({"A_naive_peak_m": nan, "A_naive_t_peak_s": nan, "A_naive_status": "naive_pass_failed",
                    "naive_pass_error": err_hn})
        errors["A_naive"] = "the A_naive pass raised (traceback in naive_pass_error)"
        return out, errors
    try:
        cvn = CaseView(hn, i)
        fn = estimator_feature(cvn, "A_naive")
    except Exception:
        errors["A_naive"] = _tb(2000)
        out.update({"A_naive_peak_m": nan, "A_naive_t_peak_s": nan, "A_naive_status": "naive_feature_failed"})
        return out, errors
    out["A_naive_peak_m"], out["A_naive_t_peak_s"], out["A_naive_status"] = fn["peak_m"], fn["t_peak_s"], fn["status"]
    try:
        out["cw_naive_pass_max_state_diff_m"] = _naive_pass_diagnostic(cvn, cvf)
    except Exception:
        errors["cw_naive_pass_diagnostic"] = _tb(2000)
        out["cw_naive_pass_max_state_diff_m"] = nan
    return out, errors


def _held_reference_fields(cv):
    """Reference and cross-check fields of one held case: the non-feature part of held_metrics.case_summary (same
    values), so that a feature failure cannot fail the reference metrics."""
    from cwvalid3.held_metrics import peak_discrepancy, terminal_arrival
    b, i = cv.b, cv.i
    out = {}
    for br in ("nerm", "eci"):
        for k, v in terminal_arrival(cv, br).items():
            out[f"{br}_{k}"] = v
        out[f"{br}_exit_step"] = int(b.exit_step[br][i])
        out[f"{br}_exit_fallback"] = bool(b.exit_fallback[br][i])
    for a, r in HELD_PAIRS:
        for k, v in peak_discrepancy(cv, a, r).items():
            out[f"{a}_vs_{r}_{k}"] = v
    return out


def _held_scenario_fields(r, batch, Ts):
    d = r["draw"]
    return {"sep_m": d["sep_m"], "wn_ratio": d["wn_ratio"], "a_max": d["a_max"], "n_radps": batch.n, "R_m": batch.R,
            "horizon_s": batch.T, "nu": d["nu"], "axis": d["axis"], "velocity": d["velocity"], "Ts": Ts, "h_main_s": batch.h,
            "h_naive_s": 0.5}


def _with_errors(out, errors):
    if errors:
        out["feature_errors"] = {**out.get("feature_errors", {}), **errors}
    return out


def work_held(recs):
    """Staged held worker (v3.1 C02; v3.1.1 R01/R02).
      1. The combined batch (cw, nerm, eci and A_H, B_H, C_H in one state vector) is propagated. A batch that raises
         is retried case by case; if it still fails for one case, or its case view fails, the primary features come
         from a CW + estimator batch for that case alone (no reference branches; feature_source
         'cw_estimator_only_fallback'; values may differ from a combined batch by round-off) and the case is
         'reference_failed'.
      2. Per case, the primary features (CW branch, A_H, B_H, C_H, saturation; each isolated) are computed and
         checkpointed before the optional A_naive pass and before any reference metric.
      3. The separate CW-only A_naive pass runs; per case its feature and diagnostic are isolated and the checkpoint
         is replaced atomically. A failure there removes only A_naive (or only the diagnostic).
      4. Per case, the reference/cross-check metrics are computed from the combined batch.
    The nominal and reference branches share step 1, so a process loss during step 1 leaves no checkpoint; cmd_run
    re-executes the task alone and, if it is lost again, runs a nominal-only recovery in a fresh process
    (recover_features: post-failure recovery, not pre-reference checkpointing)."""
    from cwvalid3.held_metrics import CaseView
    from cwvalid3.sampled import HeldBatch
    c0, w0 = time.process_time(), time.perf_counter()
    cell = recs[0]["cell"]
    Ts = CELLS[cell]["Ts"]
    alt = CELLS[cell]["altitude"]
    outs = [{k: r[k] for k in ("case_id", "physical_id", "run_id", "cell", "split", "index")} for r in recs]
    members = SER.batch_members_sha256([r["run_id"] for r in recs])
    try:
        rel0 = [r["scenario"]["rel0"] for r in recs]
        K = np.array([r["K"] for r in recs])
        am = [r["scenario"]["a_max"] for r in recs]
    except Exception:
        tb = _tb()
        for out in outs:
            out.update({"worker_status": "exception", "error": tb, "failed_stage": "setup"})
        return outs
    kw = dict(horizon_orbits=1.0, h=min(Ts, 2.0))
    hb = hn = None
    err_hb = err_hn = ""
    t0 = time.perf_counter()
    try:
        hb = HeldBatch(rel0, K, am, alt, Ts, models=("cw", "nerm", "eci"), estimators=("A_H", "B_H", "C_H"), **kw).run()
    except Exception:
        err_hb = _tb()
    t1 = time.perf_counter()
    _fault("loss_held_after_combined", [r["run_id"] for r in recs])
    if hb is None and len(recs) > 1:
        outs = []
        for r in recs:
            o = work_held([r])[0]
            o["batch_retry"] = f"case by case after an exception in a batch of {len(recs)}"
            outs.append(o)
        return outs
    # 2. primary features, checkpointed before the optional A_naive pass and before any reference metric (the reference
    #    branches were already propagated in step 1, together with the nominal branch)
    views = []
    for i, (r, out) in enumerate(zip(recs, outs)):
        ref_problem = "" if hb is not None else "main batch raised: " + err_hb
        cv = None
        try:
            if hb is not None:
                try:
                    cv = CaseView(hb, i)
                except Exception:
                    ref_problem = "case view of the main batch raised: " + _tb(2500)
                    cv = None
            if cv is None:
                hc = HeldBatch([rel0[i]], K[i:i + 1], [am[i]], alt, Ts, models=("cw",), estimators=("A_H", "B_H", "C_H"), **kw).run()
                cvf = CaseView(hc, 0)
                out["feature_source"] = "cw_estimator_only_fallback"
            else:
                cvf = cv
            out.update(_held_scenario_fields(r, cvf.b, Ts))
            prim, errors = _held_primary_features(cvf)
            out.update(prim)
            _with_errors(out, errors)
            out["feature_stage"] = "complete"
            _feature_checkpoint(out)
            views.append((cv, cvf, ref_problem))
        except Exception:
            out.update({"worker_status": "exception", "error": _tb(), "failed_stage": "features"})
            views.append(None)
    _fault("loss_held_after_features", [r["run_id"] for r in recs])
    # 3. optional comparator A_naive (separate CW-only pass)
    t2a = time.perf_counter()
    try:
        hn = HeldBatch(rel0, K, am, alt, Ts, horizon_orbits=1.0, h=0.5, models=("cw",), estimators=("A_naive",)).run()
    except Exception:
        err_hn = _tb()
    t2 = time.perf_counter()
    for i, (out, v) in enumerate(zip(outs, views)):
        if v is None:
            continue
        nf, errors = _held_naive_fields(hn, i, v[1], err_hn)
        out.update(nf)
        _with_errors(out, errors)
        _feature_checkpoint(out)
    # 4. reference/cross-check metrics
    for i, (out, v) in enumerate(zip(outs, views)):
        if v is None:
            continue
        cv, _, ref_problem = v
        try:
            ts = time.perf_counter()
            if ref_problem:
                raise RuntimeError("reference branches unavailable: " + ref_problem)
            out.update(_held_reference_fields(cv))
            out["wall_metrics_s"] = time.perf_counter() - ts
            out["worker_status"] = "ok"
        except Exception:
            out.update({"worker_status": "reference_failed", "reference_error": _tb(), "failed_stage": "reference"})
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    for out in outs:
        out["wall_propagate_main_batch_s"] = t1 - t0
        out["wall_propagate_naive_batch_s"] = t2 - t2a
        out["batch_size"] = len(recs)
        out["batch_members_sha256"] = members
        out["cpu_s"] = cpu / len(recs)
        out["wall_s"] = wall / len(recs)
        out["batch_cpu_s"], out["batch_wall_s"] = cpu, wall
    return outs


# ------------------------------------------------------------------------------------------------ recovery

def recover_features(recs):
    """Nominal-only feature recovery (v3.1.1 R02) for planned cases of one lost task that have no valid checkpoint.
    _retry_alone calls it in a fresh process when a re-executed task is lost again; no reference branch is propagated.
    Held: a CW + estimator batch for the same cases (the computation of the v3.1 CW-only fallback) and the optional
    A_naive pass; values can differ from the combined batch by round-off (different state-vector shape; quantified in
    v3_1_1_followup/results/recovery_path_differences.csv). Continuous: the CW branch and its estimators exactly as
    in work_continuous. Returns {run_id: feature-stage fields} or {run_id: {'feature_recovery_error': ...}}."""
    from cwvalid3.held_metrics import CaseView
    from cwvalid3.sampled import HeldBatch
    _fault("fail_recovery", [r["run_id"] for r in recs])
    cell = recs[0]["cell"]
    res = {}
    if CELLS[cell]["cmd_mode"] != "sampled":
        for r in recs:
            try:
                scn, law, cfg, fields = _continuous_setup(r)
                sims = _propagate(scn, cfg, law, ("cw",))
                feats, _ = _continuous_features(scn, cfg, law, sims["cw"])
                res[r["run_id"]] = {**fields, **feats}
            except Exception:
                res[r["run_id"]] = {"feature_recovery_error": _tb()}
        return res
    Ts, alt = CELLS[cell]["Ts"], CELLS[cell]["altitude"]
    try:
        rel0 = [r["scenario"]["rel0"] for r in recs]
        K = np.array([r["K"] for r in recs])
        am = [r["scenario"]["a_max"] for r in recs]
        hc = HeldBatch(rel0, K, am, alt, Ts, horizon_orbits=1.0, h=min(Ts, 2.0), models=("cw",),
                       estimators=("A_H", "B_H", "C_H")).run()
    except Exception:
        if len(recs) > 1:
            for r in recs:
                res.update(recover_features([r]))
            return res
        return {recs[0]["run_id"]: {"feature_recovery_error": _tb()}}
    hn, err_hn = None, ""
    try:
        hn = HeldBatch(rel0, K, am, alt, Ts, horizon_orbits=1.0, h=0.5, models=("cw",), estimators=("A_naive",)).run()
    except Exception:
        err_hn = _tb()
    for i, r in enumerate(recs):
        try:
            cv = CaseView(hc, i)
            o = _held_scenario_fields(r, hc, Ts)
            prim, errors = _held_primary_features(cv)
            o.update(prim)
            nf, nerr = _held_naive_fields(hn, i, cv, err_hn)
            o.update(nf)
            _with_errors(o, {**errors, **nerr})
            o["feature_stage"] = "complete"
            res[r["run_id"]] = o
        except Exception:
            res[r["run_id"]] = {"feature_recovery_error": _tb()}
    return res


def _recover_in_fresh_process(recs):
    """recover_features in a new single-worker process (the driver itself never propagates)."""
    try:
        with ProcessPoolExecutor(max_workers=1) as ex:
            return ex.submit(recover_features, recs).result()
    except Exception as exc:
        return {r["run_id"]: {"feature_recovery_error": f"recovery process failed: {exc!r}"} for r in recs}


def _merge_features(o, feats, how):
    """Merge feature-stage fields into a worker_crash record without touching its identity, status or error fields."""
    for k, v in feats.items():
        if k not in o and not SER.is_volatile(k) and not k.startswith("checkpoint_"):
            o[k] = v
    o["feature_recovery"] = how
    if how == "checkpoint":
        o["feature_checkpoint_recovered"] = True
    return o


def _crash_record(r, exc, ckdir, session=None):
    """worker_crash record for one case (operational). The case's feature checkpoint is merged in only if it names this
    run_id, is complete, was written by this engine and (when cmd_run passes its session) in this run session;
    otherwise it is ignored and the reason is stored in stale_checkpoint_ignored."""
    o = {**{k: r[k] for k in ("case_id", "physical_id", "run_id", "cell", "split", "index")},
         "worker_status": "worker_crash", "error": repr(exc), "cpu_s": None, "wall_s": None}
    p = os.path.join(ckdir, f"{r['run_id']}.json") if ckdir else None
    if p and os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as fh:
                ck = json.loads(fh.read())
        except Exception as e:
            o["stale_checkpoint_ignored"] = f"unreadable: {e!r}"
            return o
        why = [w for w, bad in (("run_id", ck.get("run_id") != r["run_id"]),
                                ("incomplete", ck.get("feature_stage") != "complete"),
                                ("engine", ck.get("checkpoint_engine_sha") != _engine_sha_cached()),
                                ("session", session is not None and ck.get("checkpoint_session") != session)) if bad]
        if why:
            o["stale_checkpoint_ignored"] = ",".join(why)
        else:
            _merge_features(o, ck, "checkpoint")
    return o


def _run_alone(fn, arg):
    """One task in its own single-worker process: a process loss there affects only this task."""
    with ProcessPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, arg).result()


def _retry_alone(fn, task, held, exc, ckdir, session):
    """Re-execute, alone in a fresh process, a task whose future failed when the worker pool broke (v3.1.1). A pool
    break fails every unfinished task, also those that never started, so the first failure does not show which task
    lost its process. If the task completes, its records are the normal ones (plus task_retry, volatile). If it is
    lost again, its cases become worker_crash (truth 'interrupted') with the features of this run's checkpoint or,
    without one, of the nominal-only recovery in a fresh process. Returns the records to write, in task order."""
    try:
        outs = _run_alone(fn, task if held else task[0])
        for o in outs:
            o["task_retry"] = f"re-executed alone in a fresh process after the worker pool broke ({type(exc).__name__})"
        return outs
    except Exception as exc2:
        outs = []
        for r in task:
            o = _crash_record(r, exc2, ckdir, session)
            o["task_retry"] = "lost again when re-executed alone in a fresh process"
            outs.append(o)
        lost = [o for o in outs if o.get("feature_recovery") != "checkpoint"]
        if lost:
            want = {o["run_id"] for o in lost}
            got = _recover_in_fresh_process([r for r in task if r["run_id"] in want])
            for o in lost:
                fr = got.get(o["run_id"]) or {"feature_recovery_error": "no result from the recovery process"}
                if "feature_recovery_error" in fr:
                    o.update({"feature_recovery": "failed", "feature_recovery_error": fr["feature_recovery_error"]})
                else:
                    _merge_features(o, fr, "driver_nominal_recovery")
        return outs


def cmd_run(cell, split, workers, batch, limit=None, out=OUT_DEFAULT):
    """Run planned cases into <out>/results and <out>/RUN_LEDGER.jsonl (canonical v3.1 records). The manifest is read
    from <out>/manifests if planned there, else from the v3.0 manifests (read-only). If a worker process is lost, the
    pool breaks and every unfinished task fails with it; after the main pool each such task is re-executed alone in a
    fresh process (_retry_alone, up to `workers` at a time). Only the cases of a task that is lost again are recorded as
    worker_crash (truth 'interrupted'); their features come from this run's checkpoint when one exists
    (feature_recovery 'checkpoint'), else from a nominal-only recovery in a fresh process (feature_recovery
    'driver_nominal_recovery', or 'failed')."""
    import uuid
    P = out_paths(out)
    man = os.path.join(P["man"], f"{cell}_{split}.jsonl")
    if not os.path.exists(man):
        man = os.path.join(MAN, f"{cell}_{split}.jsonl")
    recs = L.read_jsonl_strict(man, key="run_id")
    os.makedirs(P["res"], exist_ok=True)
    rpath = os.path.join(P["res"], f"{cell}_{split}.jsonl")
    done = {r["run_id"] for r in L.read_jsonl_strict(rpath, key="run_id", allow_missing=True)}
    todo = [r for r in recs if r["run_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"[{cell}/{split}] planned {len(recs)}, done {len(done)}, to run {len(todo)}", flush=True)
    if not todo:
        return
    sw = software()
    held = CELLS[cell]["cmd_mode"] == "sampled"
    tasks = [todo[i:i + batch] for i in range(0, len(todo), batch)] if held else [[r] for r in todo]
    fn = work_held if held else work_continuous
    ckdir = os.path.join(P["res"], "feature_checkpoints", f"{cell}_{split}")
    session = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    os.environ[CHECKPOINT_ENV] = ckdir          # inherited by the worker processes
    os.environ[SESSION_ENV] = session
    t0 = time.perf_counter()
    ncomp, cpu = 0, 0.0
    broken = []

    def write(o, fh, lf):
        o["software"] = sw
        o["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        rec_h, phys_h = SER.write_record(fh, o)
        ev = {"ledger_id": f"completed:{o['run_id']}", "event": "completed", "run_id": o["run_id"],
              "case_id": o["case_id"], "cell": cell, "split": split,
              "worker_status": o.get("worker_status"), "cpu_s": o.get("cpu_s"),
              "wall_s": o.get("wall_s"), "record_schema": SER.RECORD_SCHEMA,
              "record_sha256": rec_h, "physics_sha256": phys_h,
              "physics_hash_version": SER.PHYSICS_HASH_VERSION, "engine_sha": sw["engine_sha"],
              "engine_id_scheme": SER.ENGINE_ID_SCHEME, "results_file": os.path.basename(rpath),
              "time": o["completed_at"]}
        if o.get("feature_recovery"):
            ev["feature_recovery"] = o["feature_recovery"]
        SER.write_record(lf, ev, schema=None)
        ck = os.path.join(ckdir, f"{o['run_id']}.json")
        if o.get("worker_status") != "worker_crash" and os.path.exists(ck):
            os.remove(ck)                     # the full record is stored; the checkpoint is no longer needed

    with open(rpath, "a", encoding="utf-8", newline="\n") as fh, open(P["ledger"], "a", encoding="utf-8", newline="\n") as lf:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, t if held else t[0]): t for t in tasks}
            for f in as_completed(futs):
                try:
                    outs = f.result()
                except Exception as exc:  # the pool broke (this task's or another task's process was lost)
                    broken.append((futs[f], exc))
                    continue
                for o in outs:
                    write(o, fh, lf)
                    ncomp += 1
                    cpu += o.get("cpu_s") or 0.0
                if ncomp % max(50, batch) < len(outs):
                    el = time.perf_counter() - t0
                    print(f"[{cell}/{split}] {ncomp}/{len(todo)}  wall {el:.0f}s  cpu {cpu:.0f}s  "
                          f"({cpu / max(ncomp, 1):.2f} cpu-s/case)", flush=True)
        # v3.1.1: every task that failed with the pool is re-executed alone in a fresh process; only a task lost again
        # gets crash records (checkpoint or nominal-only recovery, references excluded). Records are never dropped.
        if broken:
            print(f"[{cell}/{split}] worker pool broke: re-executing {len(broken)} unfinished task(s) one per process", flush=True)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max(1, workers)) as tx:
                rfuts = [tx.submit(_retry_alone, fn, t, held, exc, ckdir, session) for t, exc in broken]
                for f in as_completed(rfuts):
                    for o in f.result():
                        write(o, fh, lf)
                        ncomp += 1
                        cpu += o.get("cpu_s") or 0.0
    el = time.perf_counter() - t0
    print(f"[{cell}/{split}] finished {ncomp} runs in {el:.0f}s wall, {cpu:.0f} worker cpu-s"
          + (f"; {len(broken)} task(s) re-executed after a pool break" if broken else ""), flush=True)


def cmd_status_out(out):
    """Reconcile the v3.1 output root: manifests (own, else v3.0) vs results vs ledger, and every ledger
    record_sha256 against the SHA-256 of the stored line."""
    P = {"man": os.path.join(out, "manifests"), "res": os.path.join(out, "results"), "ledger": os.path.join(out, "RUN_LEDGER.jsonl")}
    if not os.path.isdir(P["res"]):
        print(f"no v3.1 results under {out}")
        return []
    led = {}
    if os.path.exists(P["ledger"]):
        for e in L.read_jsonl_strict(P["ledger"], key="ledger_id"):
            if e.get("event") == "completed":
                led[e["run_id"]] = e
    rows = []
    for f in sorted(os.listdir(P["res"])):
        if not f.endswith(".jsonl"):
            continue
        cs = f[:-len(".jsonl")]
        man = os.path.join(P["man"], f) if os.path.exists(os.path.join(P["man"], f)) else os.path.join(MAN, f)
        recs = L.read_jsonl_strict(man, key="run_id") if os.path.exists(man) else []
        res = L.read_jsonl_strict(os.path.join(P["res"], f), key="run_id")
        rc = L.reconcile([r["run_id"] for r in recs], res, "run_id")
        bad_hash = 0
        with open(os.path.join(P["res"], f), "rb") as fh:
            for line in fh:
                rid = json.loads(line)["run_id"]
                e = led.get(rid)
                if e is None or e.get("record_sha256") != SER.stored_line_sha256(line):
                    bad_hash += 1
        rows.append({"cell_split": cs, **{k: rc[k] for k in ("n_planned", "n_rows", "n_missing", "n_extra", "n_duplicates")},
                     "ledger_hash_mismatch_or_missing": bad_hash})
    print(json.dumps(rows, indent=1))
    return rows


def cmd_status():
    rows = []
    for cell in CELLS:
        for split in ("pilot", "dev", "cal", "test", "audit", "transfer_cohort"):
            man = os.path.join(MAN, f"{cell}_{split}.jsonl")
            if not os.path.exists(man):
                continue
            recs = L.read_jsonl_strict(man, key="run_id")
            res = L.read_jsonl_strict(os.path.join(RES, f"{cell}_{split}.jsonl"), key="run_id", allow_missing=True)
            rc = L.reconcile([r["run_id"] for r in recs], res, "run_id")
            ws = {}
            for r in res:
                ws[r.get("worker_status")] = ws.get(r.get("worker_status"), 0) + 1
            cpu = sum((r.get("cpu_s") or 0.0) for r in res)
            rows.append({"cell": cell, "split": split, **{k: rc[k] for k in ("n_planned", "n_rows", "n_missing", "n_extra",
                                                                              "n_duplicates")},
                         "worker_status": ws, "cpu_h": cpu / 3600.0})
    led = L.read_jsonl_strict(LEDGER, key="ledger_id", allow_missing=True)
    print(json.dumps(rows, indent=1))
    print(f"ledger events: {len(led)}  total worker cpu-h: {sum(r['cpu_h'] for r in rows):.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "plan-cohort", "run", "status"])
    ap.add_argument("--cell")
    ap.add_argument("--split")
    ap.add_argument("--n", type=int)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--out", default=OUT_DEFAULT, help="v3.1 output root (the v3.0 tree is closed)")
    a = ap.parse_args()
    if a.cmd == "plan":
        cmd_plan(a.cell, a.split, a.n, a.out)
    elif a.cmd == "plan-cohort":
        cmd_plan_cohort(a.cell, a.split, a.targets, a.out)
    elif a.cmd == "run":
        cmd_run(a.cell, a.split, a.workers, a.batch, a.limit, a.out)
    else:
        cmd_status()
        cmd_status_out(a.out)


if __name__ == "__main__":
    main()
