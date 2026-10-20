import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
ARCH = os.path.dirname(REPO)
V3 = os.path.join(ARCH, "v3")
MAN = os.path.join(V3, "raw", "manifests")
RES = os.path.join(V3, "raw", "results")
LEDGER = os.path.join(V3, "ledger.jsonl")
OUTDEF = os.path.join(V3, "v31", "raw", "campaign")

import numpy as np

from cwvalid3 import ledger as L
from cwvalid3 import serialization as SER
from cwvalid3.domains import CELLS, makecases

ENGINE_FILES = [os.path.join(REPO, p) for p in (
    "cwvalid/simulate.py", "cwvalid/dynamics.py", "cwvalid/metrics.py", "cwvalid/experiment.py", "cwvalid/control.py",
    "cwvalid/estimate.py", "cwvalid/estimate2.py", "cwvalid/scenario.py", "cwvalid/constants.py",
    "cwvalid3/continuous.py", "cwvalid3/controllers.py", "cwvalid3/sampled.py", "cwvalid3/heldmet.py",
    "cwvalid3/domains.py", "cwvalid3/ledger.py", "experiments/v3/campaign.py")]
ENGINEFILES = sorted(set(ENGINE_FILES) | {os.path.join(REPO, p) for p in (
    "cwvalid/__init__.py", "cwvalid3/__init__.py", "cwvalid3/serialization.py", "src/__init__.py", "src/collision.py",
    "src/controller.py", "src/cw.py", "src/extval.py", "src/lvlh.py", "src/orbit.py",
    "src/perturbations.py", "src/plotting.py", "src/scenarios.py", "src/validation.py")})


def inside(path, root):
    p, r = (os.path.normcase(os.path.realpath(os.path.abspath(x))) for x in (path, root))
    try:
        return os.path.commonpath([p, r]) == r
    except ValueError:
        return False


def v30treeguard(path):
    ap = os.path.abspath(path)
    if ap.startswith("\\\\"):
        raise SystemExit(f"refusing a UNC or extended-length path: {ap}")
    if inside(ap, V3) and not inside(ap, os.path.join(V3, "v31")):
        raise SystemExit(f"refusing to write into the closed v3.0 tree: {ap}")
    return ap


def outpaths(out):
    out = v30treeguard(out)
    return {"man": os.path.join(out, "manifests"), "res": os.path.join(out, "results"),
            "ledger": os.path.join(out, "ledger.jsonl")}
BRANCHKEYS = ("status", "termination", "message", "t_final_s", "nfev", "nsteps", "wall_time_s", "n_sat_switch_events",
              "sat_events_disabled_at_s", "completed_horizon", "numerical_failure", "terminal_inside_at_end",
              "terminal_arrival_time_s", "terminal_dwell_margin_s", "terminal_arrived", "final_pos_m",
              "final_speed_mps", "max_range_m", "max_range_over_R", "sat_total_s", "sat_fraction", "sat_n_intervals",
              "sat_initial_s", "peak_raw_acc_mps2", "dv_aug_mps")
PAIRKEYS = ("max_pos_disc_m", "max_pos_disc_grid_m", "t_max_pos_disc_s", "final_pos_disc_m", "common_horizon_s",
            "max_vel_disc_mps")


def engine_identity():
    return SER.engine_identity(ENGINEFILES, REPO)


def enginever():
    return engine_identity()["sha16"]


def engine():
    return L.codever(ENGINE_FILES)


def software():
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "engine_sha": enginever(), "engine_id_scheme": SER.ENGINEIDSCHEME,
            "engine_sha_v30_formula": engine()}


def runconfig(cell):
    from cwvalid.metrics import OutcCrit
    from cwvalid.simulate import SolverCfg
    c = CELLS[cell]
    base = {"cell": cell, "chief": "circular two-body", "reference": "eci", "crosscheck": "nerm",
            "models": ["cw", "eci", "nerm"], "crit": OutcCrit().asdict2(), "monitor_dt_s": 0.5,
            "altitude_floor_m": 100000.0, "engine_sha": enginever()}
    if c["cmd_mode"] == "continuous":
        base.update({"cmd_mode": "continuous", "Ts": None, "phase_s": None, "solver": SolverCfg().asdict2(),
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


def physkeysarch():
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


def cmdplan(cell, split, N, out=OUTDEF):
    P = outpaths(out)
    os.makedirs(P["man"], exist_ok=True)
    path = os.path.join(P["man"], f"{cell}{split.replace('_', '')}.jsonl")
    if os.path.exists(path):
        sys.exit(f"manifest {path} exists; refusing to replace a planned set")
    if os.path.exists(os.path.join(MAN, f"{cell}{split.replace('_', '')}.jsonl")):
        sys.exit(f"{cell}/{split} names a v3.0 campaign split; plan new work under a new split name")
    recs, rnginfo = makecases(cell, split, N)
    cfg = runconfig(cell)
    keys = physkeysarch()
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
    summ = {"cell": cell, "split": split, "n": len(recs), "rng": rnginfo, "run_config": cfg,
            "manifest_sha256": L.filesha256(path), "archived_overlap": 0, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_archived_physics_keys_checked": len(keys), "engine_identity": engine_identity()}
    json.dump(summ, open(os.path.join(P["man"], f"{cell}{split}summary.json"), "w"), indent=2)
    with open(P["ledger"], "a", encoding="utf-8", newline="\n") as fh:
        for r in recs:
            SER.write_record(fh, {"ledger_id": f"planned:{r['run_id']}", "event": "planned", "run_id": r["run_id"],
                                  "case_id": r["case_id"], "physical_id": r["physical_id"], "cell": cell,
                                  "split": split, "index": r["index"], "config_hash": L.stablehash(cfg),
                                  "manifest_sha256": summ["manifest_sha256"], "time": summ["written"]}, schema=None)
    print(json.dumps({k: summ[k] for k in ("cell", "split", "n", "manifest_sha256")}))


def cmdplancohort(srccell, srcsplit, targets, out=OUTDEF):
    from cwvalid3.domains import makelaw
    P = outpaths(out)
    os.makedirs(P["man"], exist_ok=True)
    sp = os.path.join(P["man"], f"{srccell}{srcsplit.replace('_', '')}.jsonl")
    s = L.readjsonlstrict(sp if os.path.exists(sp) else os.path.join(MAN, f"{srccell}{srcsplit.replace('_', '')}.jsonl"),
                          key="run_id")
    for tgt in targets:
        c = CELLS[tgt]
        if c["altitude"] != CELLS[srccell]["altitude"]:
            sys.exit("cohort targets must share the source altitude (same initial states and n)")
        path = os.path.join(P["man"], f"{tgt}transfercohort.jsonl")
        if os.path.exists(path) or os.path.exists(os.path.join(MAN, f"{tgt}transfercohort.jsonl")):
            sys.exit(f"{tgt} transfer cohort already planned (v3.0 or {path})")
        cfg = runconfig(tgt)
        recs = []
        for r in s:
            law = makelaw(tgt, r["n_radps"], r["draw"]["wn_ratio"], r["draw"]["a_max"])
            q = dict(r)
            q.update(cell=tgt, split="transfer_cohort", K=law.K.tolist(), controller=law.design, cmd_mode=c["cmd_mode"],
                     Ts=c["Ts"], source_case_id=r["case_id"], source_physical_id=r["physical_id"])
            q["physical_id"] = L.physical_id(r["draw"]["rel0"], c["altitude"], 1.0, law.K, r["draw"]["a_max"])
            q["case_id"] = "C" + L.stablehash({"cohort": tgt, "source": r["case_id"]})
            q["run_id"] = L.run_id(q["physical_id"], cfg)
            recs.append(q)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for q in recs:
                SER.write_record(fh, q, schema=None)
        summ = {"cell": tgt, "split": "transfer_cohort", "n": len(recs), "source": f"{srccell}_{srcsplit}",
                "run_config": cfg, "manifest_sha256": L.filesha256(path), "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                "engine_identity": engine_identity()}
        json.dump(summ, open(os.path.join(P["man"], f"{tgt}transfercohortsummary.json"), "w"), indent=2)
        with open(P["ledger"], "a", encoding="utf-8", newline="\n") as fh:
            for q in recs:
                SER.write_record(fh, {"ledger_id": f"planned:{q['run_id']}", "event": "planned", "run_id": q["run_id"],
                                      "case_id": q["case_id"], "physical_id": q["physical_id"], "cell": tgt,
                                      "split": "transfer_cohort", "index": q["index"],
                                      "config_hash": L.stablehash(cfg), "manifest_sha256": summ["manifest_sha256"],
                                      "time": summ["written"]}, schema=None)
        print(json.dumps({k: summ[k] for k in ("cell", "split", "n", "manifest_sha256")}))


def pick(row, prefix, keys):
    return {f"{prefix}_{k}": row.get(f"{prefix}_{k}") for k in keys}


def propag(scn, cfg, law, models):
    from dataclasses import replace
    from cwvalid.control import PDLaw
    from cwvalid3.continuous import runbranchgen
    c = replace(cfg, models=tuple(models))
    if isinstance(law, PDLaw):
        from cwvalid.experiment import runbranch
        sims, _ = runbranch(scn, c)
    else:
        sims, _ = runbranchgen(scn, c, law)
    return sims


def branchfields(scn, cfg, sim, m):
    from cwvalid import metrics as M
    T = scn.horizon
    tr = M.maketrace(sim, M.commongrid(T, cfg.dt_eval))
    row = {}
    for k, v in sim.summary().items():
        if k != "model":
            row[f"{m}_{k}"] = v
    for k, v in M.legacyarr(sim, T, cfg.crit).items():
        row[f"{m}_{k}"] = v
    for k, v in M.branchdiags(sim, tr, T, cfg.crit).items():
        row[f"{m}_{k}"] = v
    return row


def _tb(n=3000):
    import traceback
    return traceback.format_exc()[-n:]


HELDPAIRS = (("cw", "eci"), ("cw", "nerm"), ("nerm", "eci"))
CKPTENV = "CWVALID3_FEATURE_CHECKPOINT_DIR"
SESSENV = "CWVALID3_RUN_SESSION"
FAULTENV = "CWVALID3_FAULT_INJECT"
ENGINESHA = []


def engineshacached():
    if not ENGINESHA:
        ENGINESHA.append(enginever())
    return ENGINESHA[0]


def fault(point, runids=()):
    spec = os.environ.get(FAULTENV, "")
    items = spec.split(",") if spec else []
    if point not in items and not any(f"{point}@{rid}" in items for rid in runids):
        return
    if point.startswith("loss_"):
        import multiprocessing
        if multiprocessing.parent_process() is not None:
            os._exit(97)
        return
    raise RuntimeError(f"injected fault at {point}")


def featckpt(out):
    d = os.environ.get(CKPTENV)
    if not d:
        return
    p = os.path.join(d, f"{out['run_id']}.json")
    try:
        os.makedirs(d, exist_ok=True)
        ck = {**out, "checkpoint_session": os.environ.get(SESSENV), "checkpoint_engine_sha": engineshacached()}
        with open(p + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
            SER.write_record(fh, ck, schema=None)
        os.replace(p + ".tmp", p)
    except Exception as exc:
        out["feature_checkpoint_error"] = repr(exc)[:500]


def contsetup(rec):
    from cwvalid.control import PDLaw
    from cwvalid.experiment import RunCfg
    from cwvalid.scenario import Scenario
    from cwvalid3.controllers import MatLaw
    scn = Scenario.from_dict(rec["scenario"])
    if rec["controller"].get("rule") == "scalar_pd":
        law = PDLaw.bandw(scn.wn_ratio * scn.n, scn.a_max)
    else:
        law = MatLaw(rec["K"], scn.a_max, rec["controller"])
    cfg = RunCfg(models=("cw", "eci", "nerm"), reference="eci", crosscheck="nerm")
    fields = {"sep_m": scn.separation, "wn_ratio": scn.wn_ratio, "a_max": scn.a_max, "n_radps": scn.n, "R_m": scn.R,
              "horizon_s": scn.horizon, "nu": rec["draw"]["nu"], "axis": rec["draw"]["axis"], "velocity": rec["draw"]["velocity"]}
    return scn, law, cfg, fields


def contfeats(scn, cfg, law, cwsim):
    from cwvalid3.continuous import estvargen
    out = {}
    cwrow = branchfields(scn, cfg, cwsim, "cw")
    out.update(pick(cwrow, "cw", BRANCHKEYS))
    out["cw_sat_switch_times"] = [float(t) for t in cwsim.satswitchtimes][:200]
    feats = estvargen(cwsim, law, scn.n, scn.R, scn.horizon)
    for k in ("d1", "d1u", "cres"):
        for f in ("peak_m", "t_peak_s", "status", "nfev", "failure_reason"):
            out[f"{k}_{f}"] = feats.get(f"{k}_{f}")
    out["feature_sat_fraction"] = (cwrow.get("cw_sat_total_s") / scn.horizon) if cwrow.get("cw_sat_total_s") is not None else None
    out["feature_stage"] = "complete"
    return out, cwrow


def workcont(rec):
    from cwvalid3.continuous import evalgen
    c0, w0 = time.process_time(), time.perf_counter()
    out = {k: rec[k] for k in ("case_id", "physical_id", "run_id", "cell", "split", "index")}
    stage = "setup"
    try:
        scn, law, cfg, fields = contsetup(rec)
        out.update(fields)
        stage = "nominal"
        t0 = time.perf_counter()
        sims = propag(scn, cfg, law, ("cw",))
        t1 = time.perf_counter()
        fault("loss_continuous_before_features", [rec["run_id"]])
        stage = "features"
        feats, cwrow = contfeats(scn, cfg, law, sims["cw"])
        out.update(feats)
        featckpt(out)
        fault("loss_continuous_after_features", [rec["run_id"]])
        t2 = time.perf_counter()
        stage = "reference"
        sims.update(propag(scn, cfg, law, ("eci", "nerm")))
        sims = {m: sims[m] for m in cfg.models}
        t3 = time.perf_counter()
        stage = "metrics"
        row, _ = evalgen(scn, cfg, sims, law)
        t4 = time.perf_counter()
        if pick(row, "cw", BRANCHKEYS) != pick(cwrow, "cw", BRANCHKEYS) and \
                SER.normalize(pick(row, "cw", BRANCHKEYS)) != SER.normalize(pick(cwrow, "cw", BRANCHKEYS)):
            raise RuntimeError("CW branch fields differ between the feature stage and the full evaluation")
        out.update({"kp": row["kp"], "kd": row["kd"]})
        for b in ("cw", "eci", "nerm"):
            out.update(pick(row, b, BRANCHKEYS))
        for p in ("cw_vs_eci", "cw_vs_nerm", "nerm_vs_eci"):
            out.update(pick(row, p, PAIRKEYS))
        out["ambiguity_flags"] = row.get("ambiguity_flags", "")
        out.update({"wall_propagate_s": (t1 - t0) + (t3 - t2), "wall_propagate_nominal_s": t1 - t0,
                    "wall_evaluate_s": t4 - t3, "wall_features_s": t2 - t1, "worker_status": "ok"})
    except Exception:
        tb = _tb()
        if stage in ("reference", "metrics"):
            out.update({"worker_status": "reference_failed", "reference_error": tb, "failed_stage": stage})
        else:
            out.update({"worker_status": "exception", "error": tb, "failed_stage": stage})
    out["cpu_s"] = time.process_time() - c0
    out["wall_s"] = time.perf_counter() - w0
    return [out]


def heldprimfeats(cv):
    from cwvalid3.heldmet import estimfeat, satsumm, termarr
    b, i = cv.b, cv.i
    out, errors = {"cw_status": b.status["cw"][i], "cw_t_end_s": float(b.t_end["cw"][i])}, {}
    try:
        for k, v in termarr(cv, "cw").items():
            out[f"cw_{k}"] = v
    except Exception:
        errors["cw_arrival"] = _tb(2000)
    for e in b.ests:
        try:
            fe = estimfeat(cv, e)
        except Exception:
            errors[e] = _tb(2000)
            fe = {"peak_m": float("nan"), "t_peak_s": float("nan"), "status": "feature_extraction_failed"}
        for k, v in fe.items():
            out[f"{e}_{k}"] = v
    try:
        for k, v in satsumm(cv).items():
            out[f"cw_{k}"] = v
    except Exception:
        errors["saturation"] = _tb(2000)
    out["feature_sat_fraction"] = out.get("cw_sat_fraction")
    return out, errors


def naivepassdiag(cvn, cvf):
    return float(np.max(np.linalg.norm(cvn.grid("cw")[:, :3] - cvf.grid("cw")[:, :3], axis=1)))


def heldnaivefields(hn, i, cvf, errhn):
    from cwvalid3.heldmet import CaseView, estimfeat
    nan = float("nan")
    out, errors = {}, {}
    if hn is None:
        out.update({"A_naive_peak_m": nan, "A_naive_t_peak_s": nan, "A_naive_status": "naive_pass_failed",
                    "naive_pass_error": errhn})
        errors["A_naive"] = "the A_naive pass raised (traceback in naive_pass_error)"
        return out, errors
    try:
        cvn = CaseView(hn, i)
        fn = estimfeat(cvn, "A_naive")
    except Exception:
        errors["A_naive"] = _tb(2000)
        out.update({"A_naive_peak_m": nan, "A_naive_t_peak_s": nan, "A_naive_status": "naive_feature_failed"})
        return out, errors
    out["A_naive_peak_m"], out["A_naive_t_peak_s"], out["A_naive_status"] = fn["peak_m"], fn["t_peak_s"], fn["status"]
    try:
        out["cw_naive_pass_max_state_diff_m"] = naivepassdiag(cvn, cvf)
    except Exception:
        errors["cw_naive_pass_diagnostic"] = _tb(2000)
        out["cw_naive_pass_max_state_diff_m"] = nan
    return out, errors


def heldreffields(cv):
    from cwvalid3.heldmet import peakdisc, termarr
    b, i = cv.b, cv.i
    out = {}
    for br in ("nerm", "eci"):
        for k, v in termarr(cv, br).items():
            out[f"{br}_{k}"] = v
        out[f"{br}_exit_step"] = int(b.exit_step[br][i])
        out[f"{br}_exit_fallback"] = bool(b.exit_fallback[br][i])
    for a, r in HELDPAIRS:
        for k, v in peakdisc(cv, a, r).items():
            out[f"{a}_vs_{r}_{k}"] = v
    return out


def heldscenfields(r, batch, Ts):
    d = r["draw"]
    return {"sep_m": d["sep_m"], "wn_ratio": d["wn_ratio"], "a_max": d["a_max"], "n_radps": batch.n, "R_m": batch.R,
            "horizon_s": batch.T, "nu": d["nu"], "axis": d["axis"], "velocity": d["velocity"], "Ts": Ts, "h_main_s": batch.h,
            "h_naive_s": 0.5}


def errs(out, errors):
    if errors:
        out["feature_errors"] = {**out.get("feature_errors", {}), **errors}
    return out


def workheld(recs):
    from cwvalid3.heldmet import CaseView
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
    errhb = errhn = ""
    t0 = time.perf_counter()
    try:
        hb = HeldBatch(rel0, K, am, alt, Ts, models=("cw", "nerm", "eci"), ests=("A_H", "B_H", "C_H"), **kw).run()
    except Exception:
        errhb = _tb()
    t1 = time.perf_counter()
    fault("loss_held_after_combined", [r["run_id"] for r in recs])
    if hb is None and len(recs) > 1:
        outs = []
        for r in recs:
            o = workheld([r])[0]
            o["batch_retry"] = f"case by case after an exception in a batch of {len(recs)}"
            outs.append(o)
        return outs
    views = []
    for i, (r, out) in enumerate(zip(recs, outs)):
        refprob = "" if hb is not None else "main batch raised: " + errhb
        cv = None
        try:
            if hb is not None:
                try:
                    cv = CaseView(hb, i)
                except Exception:
                    refprob = "case view of the main batch raised: " + _tb(2500)
                    cv = None
            if cv is None:
                hc = HeldBatch([rel0[i]], K[i:i + 1], [am[i]], alt, Ts, models=("cw",), ests=("A_H", "B_H", "C_H"), **kw).run()
                cvf = CaseView(hc, 0)
                out["feature_source"] = "cw_estimator_only_fallback"
            else:
                cvf = cv
            out.update(heldscenfields(r, cvf.b, Ts))
            prim, errors = heldprimfeats(cvf)
            out.update(prim)
            errs(out, errors)
            out["feature_stage"] = "complete"
            featckpt(out)
            views.append((cv, cvf, refprob))
        except Exception:
            out.update({"worker_status": "exception", "error": _tb(), "failed_stage": "features"})
            views.append(None)
    fault("loss_held_after_features", [r["run_id"] for r in recs])
    t2a = time.perf_counter()
    try:
        hn = HeldBatch(rel0, K, am, alt, Ts, horizon_orbits=1.0, h=0.5, models=("cw",), ests=("A_naive",)).run()
    except Exception:
        errhn = _tb()
    t2 = time.perf_counter()
    for i, (out, v) in enumerate(zip(outs, views)):
        if v is None:
            continue
        nf, errors = heldnaivefields(hn, i, v[1], errhn)
        out.update(nf)
        errs(out, errors)
        featckpt(out)
    for i, (out, v) in enumerate(zip(outs, views)):
        if v is None:
            continue
        cv, _, refprob = v
        try:
            ts = time.perf_counter()
            if refprob:
                raise RuntimeError("reference branches unavailable: " + refprob)
            out.update(heldreffields(cv))
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


def recfeats(recs):
    from cwvalid3.heldmet import CaseView
    from cwvalid3.sampled import HeldBatch
    fault("fail_recovery", [r["run_id"] for r in recs])
    cell = recs[0]["cell"]
    res = {}
    if CELLS[cell]["cmd_mode"] != "sampled":
        for r in recs:
            try:
                scn, law, cfg, fields = contsetup(r)
                sims = propag(scn, cfg, law, ("cw",))
                feats, _ = contfeats(scn, cfg, law, sims["cw"])
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
                       ests=("A_H", "B_H", "C_H")).run()
    except Exception:
        if len(recs) > 1:
            for r in recs:
                res.update(recfeats([r]))
            return res
        return {recs[0]["run_id"]: {"feature_recovery_error": _tb()}}
    hn, errhn = None, ""
    try:
        hn = HeldBatch(rel0, K, am, alt, Ts, horizon_orbits=1.0, h=0.5, models=("cw",), ests=("A_naive",)).run()
    except Exception:
        errhn = _tb()
    for i, r in enumerate(recs):
        try:
            cv = CaseView(hc, i)
            o = heldscenfields(r, hc, Ts)
            prim, errors = heldprimfeats(cv)
            o.update(prim)
            nf, nerr = heldnaivefields(hn, i, cv, errhn)
            o.update(nf)
            errs(o, {**errors, **nerr})
            o["feature_stage"] = "complete"
            res[r["run_id"]] = o
        except Exception:
            res[r["run_id"]] = {"feature_recovery_error": _tb()}
    return res


# Recovery runs in its own process so a crash there cannot take the driver down
def recproc(recs):
    try:
        with ProcessPoolExecutor(max_workers=1) as ex:
            return ex.submit(recfeats, recs).result()
    except Exception as exc:
        return {r["run_id"]: {"feature_recovery_error": f"recovery process failed: {exc!r}"} for r in recs}


def mergefeats(o, feats, how):
    for k, v in feats.items():
        if k not in o and not SER.volat(k) and not k.startswith("checkpoint_"):
            o[k] = v
    o["feature_recovery"] = how
    if how == "checkpoint":
        o["feature_checkpoint_recovered"] = True
    return o


def crashrec(r, exc, ckdir, session=None):
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
                                ("engine", ck.get("checkpoint_engine_sha") != engineshacached()),
                                ("session", session is not None and ck.get("checkpoint_session") != session)) if bad]
        if why:
            o["stale_checkpoint_ignored"] = ",".join(why)
        else:
            mergefeats(o, ck, "checkpoint")
    return o


# One task per process: if it is lost, only that task is affected
def runalone(fn, arg):
    with ProcessPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, arg).result()


# A broken pool fails every unfinished task, including ones that never started, so each failed task
# is re-run alone. Only a task that is lost again gets crash records.
def retryalone(fn, task, held, exc, ckdir, session):
    try:
        outs = runalone(fn, task if held else task[0])
        for o in outs:
            o["task_retry"] = f"re-executed alone in a fresh process after the worker pool broke ({type(exc).__name__})"
        return outs
    except Exception as exc2:
        outs = []
        for r in task:
            o = crashrec(r, exc2, ckdir, session)
            o["task_retry"] = "lost again when re-executed alone in a fresh process"
            outs.append(o)
        lost = [o for o in outs if o.get("feature_recovery") != "checkpoint"]
        if lost:
            want = {o["run_id"] for o in lost}
            got = recproc([r for r in task if r["run_id"] in want])
            for o in lost:
                fr = got.get(o["run_id"]) or {"feature_recovery_error": "no result from the recovery process"}
                if "feature_recovery_error" in fr:
                    o.update({"feature_recovery": "failed", "feature_recovery_error": fr["feature_recovery_error"]})
                else:
                    mergefeats(o, fr, "driver_nominal_recovery")
        return outs


def cmdrun(cell, split, workers, batch, limit=None, out=OUTDEF):
    import uuid
    P = outpaths(out)
    man = os.path.join(P["man"], f"{cell}{split.replace('_', '')}.jsonl")
    if not os.path.exists(man):
        man = os.path.join(MAN, f"{cell}{split.replace('_', '')}.jsonl")
    recs = L.readjsonlstrict(man, key="run_id")
    os.makedirs(P["res"], exist_ok=True)
    rpath = os.path.join(P["res"], f"{cell}{split.replace('_', '')}.jsonl")
    done = {r["run_id"] for r in L.readjsonlstrict(rpath, key="run_id", allow_missing=True)}
    todo = [r for r in recs if r["run_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"[{cell}/{split}] planned {len(recs)}, done {len(done)}, to run {len(todo)}", flush=True)
    if not todo:
        return
    sw = software()
    held = CELLS[cell]["cmd_mode"] == "sampled"
    tasks = [todo[i:i + batch] for i in range(0, len(todo), batch)] if held else [[r] for r in todo]
    fn = workheld if held else workcont
    ckdir = os.path.join(P["res"], "checkpoints", f"{cell}{split.replace('_', '')}")
    session = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    os.environ[CKPTENV] = ckdir
    os.environ[SESSENV] = session
    t0 = time.perf_counter()
    ncomp, cpu = 0, 0.0
    broken = []

    def write(o, fh, lf):
        o["software"] = sw
        o["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        rech, physh = SER.write_record(fh, o)
        ev = {"ledger_id": f"completed:{o['run_id']}", "event": "completed", "run_id": o["run_id"],
              "case_id": o["case_id"], "cell": cell, "split": split,
              "worker_status": o.get("worker_status"), "cpu_s": o.get("cpu_s"),
              "wall_s": o.get("wall_s"), "record_schema": SER.RECSCHEMA,
              "record_sha256": rech, "physics_sha256": physh,
              "physics_hash_version": SER.PHYSHASHVER, "engine_sha": sw["engine_sha"],
              "engine_id_scheme": SER.ENGINEIDSCHEME, "results_file": os.path.basename(rpath),
              "time": o["completed_at"]}
        if o.get("feature_recovery"):
            ev["feature_recovery"] = o["feature_recovery"]
        SER.write_record(lf, ev, schema=None)
        ck = os.path.join(ckdir, f"{o['run_id']}.json")
        if o.get("worker_status") != "worker_crash" and os.path.exists(ck):
            os.remove(ck)

    with open(rpath, "a", encoding="utf-8", newline="\n") as fh, open(P["ledger"], "a", encoding="utf-8", newline="\n") as lf:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, t if held else t[0]): t for t in tasks}
            for f in as_completed(futs):
                try:
                    outs = f.result()
                except Exception as exc:
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
        if broken:
            print(f"[{cell}/{split}] worker pool broke: re-executing {len(broken)} unfinished task(s) one per process", flush=True)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max(1, workers)) as tx:
                rfuts = [tx.submit(retryalone, fn, t, held, exc, ckdir, session) for t, exc in broken]
                for f in as_completed(rfuts):
                    for o in f.result():
                        write(o, fh, lf)
                        ncomp += 1
                        cpu += o.get("cpu_s") or 0.0
    el = time.perf_counter() - t0
    print(f"[{cell}/{split}] finished {ncomp} runs in {el:.0f}s wall, {cpu:.0f} worker cpu-s"
          + (f"; {len(broken)} task(s) re-executed after a pool break" if broken else ""), flush=True)


def cmdstatusout(out):
    P = {"man": os.path.join(out, "manifests"), "res": os.path.join(out, "results"), "ledger": os.path.join(out, "ledger.jsonl")}
    if not os.path.isdir(P["res"]):
        print(f"no v3.1 results under {out}")
        return []
    led = {}
    if os.path.exists(P["ledger"]):
        for e in L.readjsonlstrict(P["ledger"], key="ledger_id"):
            if e.get("event") == "completed":
                led[e["run_id"]] = e
    rows = []
    for f in sorted(os.listdir(P["res"])):
        if not f.endswith(".jsonl"):
            continue
        cs = f[:-len(".jsonl")]
        man = os.path.join(P["man"], f) if os.path.exists(os.path.join(P["man"], f)) else os.path.join(MAN, f)
        recs = L.readjsonlstrict(man, key="run_id") if os.path.exists(man) else []
        res = L.readjsonlstrict(os.path.join(P["res"], f), key="run_id")
        rc = L.reconcile([r["run_id"] for r in recs], res, "run_id")
        badhash = 0
        with open(os.path.join(P["res"], f), "rb") as fh:
            for line in fh:
                rid = json.loads(line)["run_id"]
                e = led.get(rid)
                if e is None or e.get("record_sha256") != SER.stored_line_sha256(line):
                    badhash += 1
        rows.append({"cell_split": cs, **{k: rc[k] for k in ("n_planned", "n_rows", "n_missing", "n_extra", "n_duplicates")},
                     "ledger_hash_mismatch_or_missing": badhash})
    print(json.dumps(rows, indent=1))
    return rows


def cmdstatus():
    rows = []
    for cell in CELLS:
        for split in ("pilot", "dev", "cal", "test", "audit", "transfer_cohort"):
            man = os.path.join(MAN, f"{cell}{split.replace('_', '')}.jsonl")
            if not os.path.exists(man):
                continue
            recs = L.readjsonlstrict(man, key="run_id")
            res = L.readjsonlstrict(os.path.join(RES, f"{cell}{split.replace('_', '')}.jsonl"), key="run_id", allow_missing=True)
            rc = L.reconcile([r["run_id"] for r in recs], res, "run_id")
            ws = {}
            for r in res:
                ws[r.get("worker_status")] = ws.get(r.get("worker_status"), 0) + 1
            cpu = sum((r.get("cpu_s") or 0.0) for r in res)
            rows.append({"cell": cell, "split": split, **{k: rc[k] for k in ("n_planned", "n_rows", "n_missing", "n_extra",
                                                                              "n_duplicates")},
                         "worker_status": ws, "cpu_h": cpu / 3600.0})
    led = L.readjsonlstrict(LEDGER, key="ledger_id", allow_missing=True)
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
    ap.add_argument("--out", default=OUTDEF)
    a = ap.parse_args()
    if a.cmd == "plan":
        cmdplan(a.cell, a.split, a.n, a.out)
    elif a.cmd == "plan-cohort":
        cmdplancohort(a.cell, a.split, a.targets, a.out)
    elif a.cmd == "run":
        cmdrun(a.cell, a.split, a.workers, a.batch, a.limit, a.out)
    else:
        cmdstatus()
        cmdstatusout(a.out)


if __name__ == "__main__":
    main()
