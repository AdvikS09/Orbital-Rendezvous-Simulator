"""C06: D1 completed-data calibration addendum - a POST-TEST sensitivity analysis.

The 43 D1 calibration records lost to operational interruptions (worker_crash; DEV-01) were re-executed
after the single test evaluation (reruns completed 2026-10-14 06:16:56-06:17:21; test evaluation first run
06:13:34) into raw/results/D1_cal_reruns.jsonl. This script joins the reruns one-to-one to the crashed
records (same run_id, case_id, physical_id, index) and recomputes all ten D1 calibration families
(A, B, C, sep, sat x 1 m, 0.1 m) with the same prespecified candidates, alpha = 0.01, delta = 0.05 and
M = 1012 (per-candidate level delta/M), twice:
  original  : raw/results/D1_cal.jsonl as stored (must reproduce raw/calibration_freeze.json exactly)
  completed : the 43 worker_crash records replaced by their re-executions
with the frozen v3.0 labelling (cwvalid3.calibration) and again with the v3.1 contract (cwvalid3.contracts,
crash -> 'interrupted'); both labellings must give identical selections.

Status: post-test, descriptive. The frozen selection is the one in calibration_freeze.json; the completed
values are not a replacement. Whether the 43 interrupted cases are an outcome-independent subset is not
established by the record (they were the cases in flight when the driver was stopped); the completed
analysis is reported so that the selection's sensitivity to them is visible, not to support that claim.

Outputs (v3_1_correction/results): calibration_completed_D1_addendum.csv (every candidate, both versions)
and calibration_completed_D1_addendum.json (identity checks, census, ledger checks, per-family summary).
Usage: python code/experiments/v3_1/c06_d1_completed_addendum.py
"""

import collections
import csv
import hashlib
import json
import math
import os
import sys

import v31paths as P  # noqa: E402

ROOT = P.ARCH
CODE = P.REPO
sys.path.insert(0, os.path.join(CODE, "experiments", "v3"))

import numpy as np  # noqa: E402

from cwvalid3 import calibration as CAL  # noqa: E402
from cwvalid3 import contracts as C31  # noqa: E402
from cwvalid3 import ledger as L  # noqa: E402
import v3data as VD  # noqa: E402  (only VD.adverse and VD.methods_for are used; no file is read at import)

OUT = P.V31_RES
V3 = os.path.join(ROOT, "v3_extension")
PLAN = os.path.join(V3, "PROSPECTIVE_PLAN.json")
FREEZE = os.path.join(V3, "raw", "calibration_freeze.json")
MAN = os.path.join(V3, "raw", "manifests", "D1_cal.jsonl")
RES = os.path.join(V3, "raw", "results", "D1_cal.jsonl")
RER = os.path.join(V3, "raw", "results", "D1_cal_reruns.jsonl")
LEDGER = os.path.join(V3, "RUN_LEDGER.jsonl")
CANDCSV = os.path.join(V3, "results", "calibration_candidates.csv")
RERUNCHK = os.path.join(V3, "raw", "rerun_check_D1_cal.json")
CELL, SPLIT, KIND = "D1", "cal", "continuous"
ADVERSE_LABELS = ("violation", "exit", "failure", "unresolved")
IDENT = ("case_id", "physical_id", "run_id", "cell", "split", "index")


def stream_jsonl(path, key="run_id"):
    """Strict streaming reader: yields records one line at a time; raises on malformed, truncated,
    empty or duplicate-key lines (same contract as ledger.read_jsonl_strict, without loading the file)."""
    seen = {}
    with open(path, "rb") as fh:
        i = 0
        for raw in fh:
            i += 1
            if not raw.endswith(b"\n"):
                raise ValueError(f"{path}:{i}: final line has no newline")
            line = raw.decode("utf-8").strip()
            if not line:
                raise ValueError(f"{path}:{i}: empty line")
            r = json.loads(line)
            if not isinstance(r, dict):
                raise ValueError(f"{path}:{i}: not an object")
            if key is not None:
                if key not in r:
                    raise ValueError(f"{path}:{i}: missing {key}")
                if r[key] in seen:
                    raise ValueError(f"{path}:{i}: duplicate {key}={r[key]} (first at {seen[r[key]]})")
                seen[r[key]] = i
            yield r


def result_hash(o):
    """Exactly as campaign.cmd_run computes the ledger 'result_sha256_16' field."""
    return L.stable_hash({k: v for k, v in o.items() if k not in ("software", "completed_at")}, 32)


def derive(r, methods):
    """Per-case derived quantities used by calibrate.py (via v3data.load_split)."""
    tr = CAL.truth_record(r)
    d = {"worker_status": r.get("worker_status"), "truth_status": tr["truth_status"],
         "label_1m": CAL.criterion_label(tr, 1.0), "label_0.1m": CAL.criterion_label(tr, 0.1),
         "truth31": C31.truth_record(r, KIND)["truth_status"],
         "label31_1m": C31.criterion_label(C31.truth_record(r, KIND), 1.0),
         "label31_0.1m": C31.criterion_label(C31.truth_record(r, KIND), 0.1),
         **{f"f31_{m}": C31.feature_value(r, m, KIND) for m in methods},
         "arrival_status": CAL.arrival_label(r, tr)["arrival_status"], "tr": tr}
    for m in methods:
        d[f"f_{m}"] = CAL.feature_value(r, m, KIND)
    return d


def jnum(x):
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, (np.integer,)):
        x = int(x)
    if isinstance(x, (np.bool_,)):
        x = bool(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def main():
    log = []
    plan = json.load(open(PLAN, encoding="utf-8"))
    frz = json.load(open(FREEZE, encoding="utf-8"))
    mult = plan["multiplicity"]
    level, alpha = mult["per_candidate_level"], mult["alpha"]
    fams = [f for f in plan["families"] if f["cell"] == CELL]
    methods = VD.methods_for(CELL)

    # ---------------- provenance / frozen-code identity ----------------
    prov = {"plan_sha256": L.file_sha256(PLAN), "freeze_plan_sha256": frz["plan_sha256"],
            "D1_cal_results_sha256": L.file_sha256(RES),
            "freeze_D1_cal_results_sha256": frz["calibration_results_sha256"][CELL],
            "D1_cal_reruns_sha256": L.file_sha256(RER), "D1_cal_manifest_sha256": L.file_sha256(MAN),
            "calibration_freeze_sha256": L.file_sha256(FREEZE), "ledger_sha256": L.file_sha256(LEDGER),
            "frozen_code_check": {}}
    for rel in ("cwvalid3/calibration.py", "experiments/v3/v3data.py", "experiments/v3/calibrate.py", "cwvalid3/ledger.py",
                "cwvalid3/domains.py"):
        h = L.text_sha256(os.path.join(CODE, rel))
        prov["frozen_code_check"][rel] = {"text_sha256": h, "plan_sha256": plan["implementation"]["code_sha256"].get(rel),
                                          "match": h == plan["implementation"]["code_sha256"].get(rel)}
    prov["plan_sha_match"] = prov["plan_sha256"] == prov["freeze_plan_sha256"]
    prov["D1_cal_results_sha_match"] = prov["D1_cal_results_sha256"] == prov["freeze_D1_cal_results_sha256"]
    prov["alpha"], prov["per_candidate_level"], prov["M_total_candidates"], prov["delta"] = alpha, level, mult["M_total_candidates"], mult["delta"]
    prov["freeze_alpha_level_match"] = (frz["alpha"] == alpha and frz["per_candidate_level"] == level)

    # ---------------- manifest (streamed) ----------------
    man_order, man = [], {}
    for m in stream_jsonl(MAN):
        man_order.append(m["run_id"])
        man[m["run_id"]] = {k: m.get(k) for k in IDENT}

    # ---------------- original results (streamed) ----------------
    orig = {}
    crash_raw = {}
    orig_hash = {}
    orig_ident_mismatch = []
    orig_engine = collections.Counter()
    for r in stream_jsonl(RES):
        rid = r["run_id"]
        orig[rid] = derive(r, methods)
        orig_hash[rid] = result_hash(r)
        orig_engine[(r.get("software") or {}).get("engine_sha")] += 1
        if rid in man and any(r.get(k) != man[rid][k] for k in IDENT):
            orig_ident_mismatch.append(rid)
        if r.get("worker_status") == "worker_crash":
            crash_raw[rid] = {k: r.get(k) for k in IDENT + ("worker_status", "error", "completed_at")}
    rc_orig = L.reconcile(man_order, [{"run_id": k} for k in orig], "run_id")

    # ---------------- reruns (streamed) ----------------
    rer = {}
    rer_raw_ident = {}
    rer_info = {}
    for r in stream_jsonl(RER):
        rid = r["run_id"]
        rer[rid] = derive(r, methods)
        rer_raw_ident[rid] = {k: r.get(k) for k in IDENT}
        rer_info[rid] = {"worker_status": r.get("worker_status"), "cpu_s": r.get("cpu_s"), "wall_s": r.get("wall_s"),
                         "engine_sha": (r.get("software") or {}).get("engine_sha"), "software": r.get("software"),
                         "rerun_of": r.get("rerun_of"), "completed_at": r.get("completed_at"),
                         "result_hash32": result_hash(r)}

    crashed = set(crash_raw)
    rset = set(rer)
    ident = {
        "n_original_records": len(orig), "n_manifest": len(man_order),
        "original_vs_manifest_reconcile": {k: rc_orig[k] for k in ("n_planned", "n_rows", "n_missing", "n_extra", "n_duplicates")},
        "original_identity_mismatch_vs_manifest": len(orig_ident_mismatch),
        "n_original_worker_crash": len(crashed),
        "original_worker_status": dict(collections.Counter(d["worker_status"] for d in orig.values())),
        "n_reruns": len(rer),
        "rerun_ids_equal_crash_ids": rset == crashed,
        "reruns_not_in_crash_set": sorted(rset - crashed), "crash_ids_without_rerun": sorted(crashed - rset),
        "reruns_not_in_manifest": sorted(rset - set(man)),
        "rerun_identity_mismatch_vs_manifest": sorted(rid for rid in rset if rid in man and any(rer_raw_ident[rid][k] != man[rid][k] for k in IDENT)),
        "rerun_identity_mismatch_vs_original_record": sorted(rid for rid in rset & crashed if any(rer_raw_ident[rid][k] != crash_raw[rid][k] for k in IDENT)),
        "rerun_worker_status": dict(collections.Counter(v["worker_status"] for v in rer_info.values())),
        "rerun_engine_sha": dict(collections.Counter(v["engine_sha"] for v in rer_info.values())),
        "original_engine_sha": {str(k): v for k, v in orig_engine.items()},
        "plan_engine_sha": plan["implementation"].get("engine_sha"),
        "rerun_software_distinct": [json.loads(s) for s in sorted({json.dumps(v["software"], sort_keys=True) for v in rer_info.values()})],
        "original_crash_error_distinct": sorted({str(v["error"]) for v in crash_raw.values()}),
        "crash_index_range": [min(v["index"] for v in crash_raw.values()), max(v["index"] for v in crash_raw.values())] if crash_raw else None,
        "crash_indices": sorted(v["index"] for v in crash_raw.values()),
    }
    ident["one_to_one_ok"] = (ident["n_original_worker_crash"] == 43 and ident["n_reruns"] == 43 and ident["rerun_ids_equal_crash_ids"]
                              and not ident["reruns_not_in_manifest"] and not ident["rerun_identity_mismatch_vs_manifest"]
                              and not ident["rerun_identity_mismatch_vs_original_record"])
    assert ident["n_original_worker_crash"] == 43, ident["n_original_worker_crash"]
    assert ident["n_reruns"] == 43, ident["n_reruns"]
    assert ident["rerun_ids_equal_crash_ids"]
    assert not ident["reruns_not_in_manifest"]
    assert not ident["rerun_identity_mismatch_vs_manifest"]
    assert not ident["rerun_identity_mismatch_vs_original_record"]
    assert rc_orig["n_missing"] == 0 and rc_orig["n_extra"] == 0 and rc_orig["n_duplicates"] == 0

    # ---------------- arrays in manifest order (as v3data.load_split) ----------------
    versions = {"original": [orig[k] for k in man_order],
                "completed": [rer.get(k, orig[k]) for k in man_order]}
    is_rerun = np.array([k in rset for k in man_order])

    # ---------------- census ----------------
    census = {}
    for ver, recs in versions.items():
        census[ver] = {
            "truth_status": dict(collections.Counter(d["truth_status"] for d in recs)),
            "label_1m": dict(collections.Counter(d["label_1m"] for d in recs)),
            "label_0.1m": dict(collections.Counter(d["label_0.1m"] for d in recs)),
            "worker_status": dict(collections.Counter(d["worker_status"] for d in recs)),
            "arrival_status": dict(collections.Counter(d["arrival_status"] for d in recs)),
            "feature_unavailable": {m: int(sum(1 for d in recs if not math.isfinite(d[f"f_{m}"]))) for m in methods},
        }
    census["reruns_only"] = {
        "truth_status": dict(collections.Counter(d["truth_status"] for d in rer.values())),
        "label_1m": dict(collections.Counter(d["label_1m"] for d in rer.values())),
        "label_0.1m": dict(collections.Counter(d["label_0.1m"] for d in rer.values())),
        "arrival_status": dict(collections.Counter(d["arrival_status"] for d in rer.values())),
        "feature_unavailable": {m: int(sum(1 for d in rer.values() if not math.isfinite(d[f"f_{m}"]))) for m in methods},
    }
    fc = frz["census"][CELL]
    census["original_matches_freeze_census"] = {k: census["original"][k] == fc[k] for k in ("truth_status", "label_1m", "label_0.1m", "worker_status")}

    # ---------------- stored per-candidate table (for exact comparison) ----------------
    stored = {}
    with open(CANDCSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["cell"] == CELL:
                stored[(row["family_id"], int(row["cid"]))] = row

    # ---------------- calibration, both versions ----------------
    csv_rows, fam_summary = [], []
    bench = {}
    for f in fams:
        fid = f["family_id"]
        per = {}
        for ver, recs in versions.items():
            v = np.array([d[f"f_{f['method']}"] for d in recs], float)
            labels = np.array([d[f"label_{f['criterion']}"] for d in recs], object)
            adv = VD.adverse(labels)
            # independent adverse definition as in rerun_interrupted.py (tol from the family) for a cross-check
            adv2 = np.array([CAL.criterion_label(d["tr"], f["tol_m"]) in ADVERSE_LABELS for d in recs])
            assert np.array_equal(adv, adv2)
            rows, best = CAL.calibrate_family(v, adv, f["candidates"], alpha, level)
            # v3.1 contract labelling: same features, crash -> 'interrupted' (still adverse)
            v31 = np.array([d[f"f31_{f['method']}"] for d in recs], float)
            lab31 = [d[f"label31_{f['criterion']}"] for d in recs]
            C31.validate_labels(lab31)
            adv31 = np.array([C31.is_adverse(x) for x in lab31])
            rows31, best31 = CAL.calibrate_family(v31, adv31, f["candidates"], alpha, level)
            assert np.array_equal(np.isnan(v), np.isnan(v31)) and np.array_equal(v[np.isfinite(v)], v31[np.isfinite(v31)])
            assert [(r["n_accepted"], r["n_adverse"], r["ucb"]) for r in rows] == [(r["n_accepted"], r["n_adverse"], r["ucb"]) for r in rows31]
            assert (best is None and best31 is None) or (best is not None and best31 is not None and best["cid"] == best31["cid"])
            for r in rows:
                acc = CAL.accept_mask(v, r["threshold"])
                C31.check_label_partition(C31.label_counts(list(acc), lab31), int(acc.sum()))
            for r in rows:
                acc = CAL.accept_mask(v, r["threshold"])
                lab_acc = collections.Counter(labels[acc])
                r["acc_from_reruns"] = int((acc & is_rerun).sum())
                r["adv_from_reruns"] = int((acc & adv & is_rerun).sum())
                r["acc_by_label"] = {k: int(lab_acc.get(k, 0)) for k in ("ok",) + ADVERSE_LABELS}
            per[ver] = {"rows": rows, "best": best, "n_cal": int(v.size), "n_unavail": int((~np.isfinite(v)).sum())}
        # compare original with freeze selection and stored candidate table
        s = frz["selection"][fid]
        bo, bc = per["original"]["best"], per["completed"]["best"]
        orig_match = {
            "status": s["status"] == ("selected" if bo else "abstain_no_admissible_candidate"),
            "threshold": s["threshold"] == (None if bo is None else bo["threshold"]),
            "cid": s["cid"] == (None if bo is None else bo["cid"]),
            "cal_accepted": s["cal_accepted"] == (None if bo is None else bo["n_accepted"]),
            "cal_adverse": s["cal_adverse"] == (None if bo is None else bo["n_adverse"]),
            "cal_ucb_exact": s["cal_ucb"] == (None if bo is None else bo["ucb"]),
            "cal_feature_unavailable": s["cal_feature_unavailable"] == per["original"]["n_unavail"],
            "n_cal": s["n_cal"] == per["original"]["n_cal"],
        }
        cand_table_mismatch = []
        for r in per["original"]["rows"]:
            st = stored.get((fid, r["cid"]))
            sel = bo is not None and r["cid"] == bo["cid"]
            if st is None:
                cand_table_mismatch.append((r["cid"], "missing in calibration_candidates.csv"))
                continue
            checks = {"threshold": float(st["threshold"]) == r["threshold"], "n_accepted": int(st["n_accepted"]) == r["n_accepted"],
                      "n_adverse": int(st["n_adverse"]) == r["n_adverse"], "ucb": float(st["ucb"]) == r["ucb"],
                      "admissible": (st["admissible"] == "True") == bool(r["admissible"]), "selected": (st["selected"] == "True") == sel,
                      "source": st["source"] == r["source"]}
            if not all(checks.values()):
                cand_table_mismatch.append((r["cid"], {k: v for k, v in checks.items() if not v}))
        n_stored = sum(1 for k in stored if k[0] == fid)
        for ver in ("original", "completed"):
            best = per[ver]["best"]
            for r in per[ver]["rows"]:
                ro = next(x for x in per["original"]["rows"] if x["cid"] == r["cid"])
                csv_rows.append({
                    "version": ver, "family_id": fid, "cell": f["cell"], "method": f["method"], "criterion": f["criterion"],
                    "tol_m": f["tol_m"], "cid": r["cid"], "source": r["source"], "threshold": repr(float(r["threshold"])),
                    "n_cal": per[ver]["n_cal"], "n_feature_unavailable": per[ver]["n_unavail"],
                    "n_accepted": r["n_accepted"], "n_adverse": r["n_adverse"], "ucb": repr(float(r["ucb"])),
                    "alpha": alpha, "per_candidate_level": repr(level), "admissible": bool(r["admissible"]),
                    "selected": best is not None and r["cid"] == best["cid"],
                    "acc_ok": r["acc_by_label"]["ok"], "acc_violation": r["acc_by_label"]["violation"],
                    "acc_exit": r["acc_by_label"]["exit"], "acc_failure": r["acc_by_label"]["failure"],
                    "acc_unresolved": r["acc_by_label"]["unresolved"],
                    "accepted_from_reruns": r["acc_from_reruns"], "adverse_from_reruns": r["adv_from_reruns"],
                    "delta_n_accepted_vs_original": r["n_accepted"] - ro["n_accepted"],
                    "delta_n_adverse_vs_original": r["n_adverse"] - ro["n_adverse"],
                    "admissibility_changed_vs_original": bool(r["admissible"]) != bool(ro["admissible"]),
                })

        def bsum(b):
            return None if b is None else {"cid": b["cid"], "source": b["source"], "threshold": b["threshold"],
                                           "n_accepted": b["n_accepted"], "n_adverse": b["n_adverse"], "ucb": b["ucb"]}
        adm_o = [r["cid"] for r in per["original"]["rows"] if r["admissible"]]
        adm_c = [r["cid"] for r in per["completed"]["rows"] if r["admissible"]]
        flips = [{"cid": ro["cid"], "threshold": ro["threshold"], "original": {"n": ro["n_accepted"], "b": ro["n_adverse"], "ucb": ro["ucb"]},
                  "completed": {"n": rc["n_accepted"], "b": rc["n_adverse"], "ucb": rc["ucb"]},
                  "direction": "became_admissible" if rc["admissible"] else "became_inadmissible"}
                 for ro, rc in zip(per["original"]["rows"], per["completed"]["rows"]) if bool(ro["admissible"]) != bool(rc["admissible"])]
        # closest-to-alpha inadmissible candidate (does completion move any candidate across alpha?)
        fam_summary.append({
            "family_id": fid, "method": f["method"], "criterion": f["criterion"], "tol_m": f["tol_m"],
            "n_candidates": len(f["candidates"]), "n_candidates_in_stored_csv": n_stored,
            "freeze_selection": {k: s[k] for k in ("status", "cid", "source", "threshold", "cal_accepted", "cal_adverse", "cal_ucb",
                                                   "cal_feature_unavailable")},
            "original_selected": bsum(bo), "completed_selected": bsum(bc),
            "original_reproduces_freeze": orig_match, "original_reproduces_freeze_all": all(orig_match.values()),
            "original_matches_stored_candidate_table": not cand_table_mismatch, "stored_candidate_table_mismatches": cand_table_mismatch,
            "same_selected_candidate": (None if bo is None else bo["cid"]) == (None if bc is None else bc["cid"]),
            "same_selected_threshold": (None if bo is None else bo["threshold"]) == (None if bc is None else bc["threshold"]),
            "selected_delta_accepted": None if (bo is None or bc is None) else bc["n_accepted"] - bo["n_accepted"],
            "selected_delta_adverse": None if (bo is None or bc is None) else bc["n_adverse"] - bo["n_adverse"],
            "selected_ucb_original": None if bo is None else bo["ucb"], "selected_ucb_completed": None if bc is None else bc["ucb"],
            "selected_ucb_margin_to_alpha_original": None if bo is None else alpha - bo["ucb"],
            "selected_ucb_margin_to_alpha_completed": None if bc is None else alpha - bc["ucb"],
            "n_feature_unavailable_original": per["original"]["n_unavail"], "n_feature_unavailable_completed": per["completed"]["n_unavail"],
            "admissible_cids_original": adm_o, "admissible_cids_completed": adm_c,
            "admissibility_flips": flips,
            "max_candidate_delta_accepted": max(rc["n_accepted"] - ro["n_accepted"] for ro, rc in zip(per["original"]["rows"], per["completed"]["rows"])),
            "max_candidate_delta_adverse": max(rc["n_adverse"] - ro["n_adverse"] for ro, rc in zip(per["original"]["rows"], per["completed"]["rows"])),
            "min_ucb_inadmissible_original": min([r["ucb"] for r in per["original"]["rows"] if not r["admissible"]], default=None),
            "min_ucb_inadmissible_completed": min([r["ucb"] for r in per["completed"]["rows"] if not r["admissible"]], default=None),
        })
        if fid == "D1|A|1m":
            bench = {"original": bsum(bo), "completed": bsum(bc),
                     "stated_original": {"n_accepted": 3328, "n_adverse": 0, "ucb_approx": 0.002974958},
                     "stated_completed": {"n_accepted": 3363, "n_adverse": 0, "ucb_approx": 0.002944042},
                     "closed_form_check": {"original": 1.0 - level ** (1.0 / 3328), "completed": 1.0 - level ** (1.0 / 3363)}}
            bench["original_ok"] = (bo is not None and bo["n_accepted"] == 3328 and bo["n_adverse"] == 0 and abs(bo["ucb"] - 0.002974958) < 5e-10)
            bench["completed_ok"] = (bc is not None and bc["n_accepted"] == 3363 and bc["n_adverse"] == 0 and abs(bc["ucb"] - 0.002944042) < 5e-10)

    # ---------------- comparison with the repository's own rerun check ----------------
    rchk = json.load(open(RERUNCHK, encoding="utf-8"))
    rchk_cmp = []
    for e in rchk["selection_sensitivity"]:
        fs = next(x for x in fam_summary if x["family_id"] == e["family_id"])
        bc = fs["completed_selected"]
        rchk_cmp.append({"family_id": e["family_id"],
                         "threshold_match": e["threshold_with_reruns"] == (None if bc is None else bc["threshold"]),
                         "accepted_match": e["accepted_with_reruns"] == (None if bc is None else bc["n_accepted"]),
                         "adverse_match": e["adverse_with_reruns"] == (None if bc is None else bc["n_adverse"])})

    # ---------------- ledger (streamed) ----------------
    ev = collections.defaultdict(list)
    n_lines = 0
    for r in stream_jsonl(LEDGER, key="ledger_id"):
        n_lines += 1
        if r.get("cell") == CELL and r.get("split") == SPLIT:
            ev[r["event"]].append(r)
    comp_ids = collections.Counter(e["run_id"] for e in ev["completed"])
    crec_ids = collections.Counter(e["run_id"] for e in ev["completed_reconciled"])
    rerun_ids = collections.Counter(e["run_id"] for e in ev["rerun"])
    plan_ids = collections.Counter(e["run_id"] for e in ev["planned"])
    per_run = collections.Counter()
    for c in (comp_ids, crec_ids):
        per_run.update(c)
    man_set = set(man_order)
    hash_mismatch = [e["run_id"] for e in ev["completed"] + ev["completed_reconciled"]
                     if e.get("result_sha256_16") is not None and orig_hash.get(e["run_id"]) != e["result_sha256_16"]]
    ws_mismatch = [e["run_id"] for e in ev["completed"] + ev["completed_reconciled"]
                   if orig.get(e["run_id"], {}).get("worker_status") != e.get("worker_status")]
    rerun_ev_mismatch = [e["run_id"] for e in ev["rerun"]
                         if e["run_id"] not in rer_info or e.get("worker_status") != rer_info[e["run_id"]]["worker_status"]
                         or e.get("cpu_s") != rer_info[e["run_id"]]["cpu_s"] or e.get("wall_s") != rer_info[e["run_id"]]["wall_s"]
                         or e.get("case_id") != man.get(e["run_id"], {}).get("case_id")]
    crash_event_kind = collections.Counter(("completed" if rid in comp_ids else "") + ("completed_reconciled" if rid in crec_ids else "")
                                           for rid in crashed)
    ledger = {
        "n_lines_total": n_lines,
        "D1_cal_event_counts": {k: len(v) for k, v in ev.items()},
        "n_rerun_events": len(ev["rerun"]), "n_completed_reconciled_events": len(ev["completed_reconciled"]),
        "rerun_event_run_ids_equal_crash_ids": set(rerun_ids) == crashed and all(v == 1 for v in rerun_ids.values()),
        "rerun_event_worker_status": dict(collections.Counter(e.get("worker_status") for e in ev["rerun"])),
        "rerun_event_engine_sha": dict(collections.Counter(e.get("engine_sha") for e in ev["rerun"])),
        "rerun_events_with_result_hash": sum(1 for e in ev["rerun"] if "result_sha256_16" in e),
        "rerun_event_vs_rerun_record_mismatch": rerun_ev_mismatch,
        "completed_reconciled": [{k: e.get(k) for k in ("run_id", "case_id", "worker_status", "result_sha256_16", "time", "note")} for e in ev["completed_reconciled"]],
        "completed_reconciled_run_in_crash_set": all(e["run_id"] in crashed for e in ev["completed_reconciled"]),
        "completed_reconciled_hash_matches_record": all(orig_hash.get(e["run_id"]) == e.get("result_sha256_16") for e in ev["completed_reconciled"]),
        "planned_events_equal_manifest": set(plan_ids) == man_set and all(v == 1 for v in plan_ids.values()),
        "completed_or_reconciled_cover_manifest_exactly_once": set(per_run) == man_set and all(v == 1 for v in per_run.values()),
        "n_run_ids_with_completed": len(comp_ids), "n_run_ids_with_completed_reconciled": len(crec_ids),
        "n_run_ids_with_both": len(set(comp_ids) & set(crec_ids)),
        "n_manifest_ids_without_completion_event": len(man_set - set(per_run)),
        "n_completion_ids_not_in_manifest": len(set(per_run) - man_set),
        "max_completion_events_per_run": max(per_run.values()) if per_run else None,
        "completed_event_worker_status": dict(collections.Counter(e.get("worker_status") for e in ev["completed"])),
        "crash_runs_by_completion_event_kind": dict(crash_event_kind),
        "completion_result_hash_mismatches_vs_stored_records": len(hash_mismatch),
        "completion_result_hash_mismatch_ids": hash_mismatch[:20],
        "completion_worker_status_mismatches_vs_stored_records": len(ws_mismatch),
    }

    summary = {
        "finding_id": "C06", "cell": CELL, "split": SPLIT,
        "command": "python code/experiments/v3_1/c06_d1_completed_addendum.py",
        "provenance": prov, "identity_checks": ident, "benchmark_D1_A_1m": bench,
        "families": fam_summary,
        "all_original_reproduce_freeze": all(x["original_reproduces_freeze_all"] for x in fam_summary),
        "all_original_match_stored_candidate_table": all(x["original_matches_stored_candidate_table"] for x in fam_summary),
        "n_families_selection_changed": sum(1 for x in fam_summary if not x["same_selected_candidate"]),
        "n_candidates_admissibility_flipped": sum(len(x["admissibility_flips"]) for x in fam_summary),
        "n_candidates_total": sum(x["n_candidates"] for x in fam_summary),
        "n_candidate_rows_gaining_adverse_cases": sum(1 for r in csv_rows if r["version"] == "completed" and r["delta_n_adverse_vs_original"] > 0),
        "n_candidate_rows_gaining_adverse_cases_admissible_original": sum(
            1 for r in csv_rows if r["version"] == "completed" and r["delta_n_adverse_vs_original"] > 0 and
            next(x for x in csv_rows if x["version"] == "original" and x["family_id"] == r["family_id"] and x["cid"] == r["cid"])["admissible"]),
        "contract_v31_labelling_gives_identical_counts_and_selections": True,
        "status": "post-test sensitivity (descriptive); the frozen selection is unchanged and remains the reported one",
        "outcome_independence_of_the_43": "not established by the record; not relied on",
        "repository_rerun_check_comparison": rchk_cmp,
        "census": census, "ledger": ledger,
    }
    with open(os.path.join(OUT, "calibration_completed_D1_addendum.json"), "w", encoding="utf-8") as fh:
        json.dump(L.jsonable(summary), fh, indent=1, default=jnum)
    cols = list(csv_rows[0].keys())
    with open(os.path.join(OUT, "calibration_completed_D1_addendum.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)
    # console summary
    print("plan/freeze sha match:", prov["plan_sha_match"], " D1_cal results sha match freeze:", prov["D1_cal_results_sha_match"])
    print("frozen code:", {k: v["match"] for k, v in prov["frozen_code_check"].items()})
    print("identity one-to-one:", ident["one_to_one_ok"], " rerun ws:", ident["rerun_worker_status"])
    print("all original reproduce freeze:", summary["all_original_reproduce_freeze"],
          " match stored csv:", summary["all_original_match_stored_candidate_table"])
    for x in fam_summary:
        o, c = x["original_selected"], x["completed_selected"]
        f = lambda b: "abstain" if b is None else f"cid={b['cid']} t={b['threshold']!r} n={b['n_accepted']} b={b['n_adverse']} U={b['ucb']:.9f}"
        print(f"{x['family_id']:12s} orig[{f(o)}]  comp[{f(c)}]  same={x['same_selected_candidate']} flips={len(x['admissibility_flips'])} "
              f"repro={x['original_reproduces_freeze_all']}")
    print("benchmark:", bench.get("original_ok"), bench.get("completed_ok"))
    print("census original:", census["original"]["truth_status"], census["original"]["label_1m"], census["original"]["label_0.1m"])
    print("census completed:", census["completed"]["truth_status"], census["completed"]["label_1m"], census["completed"]["label_0.1m"])
    print("census reruns:", census["reruns_only"])
    print("ledger:", json.dumps(L.jsonable(ledger), default=jnum)[:3000])
    print("rerun-check comparison:", rchk_cmp)
    print("csv rows:", len(csv_rows))


if __name__ == "__main__":
    main()
