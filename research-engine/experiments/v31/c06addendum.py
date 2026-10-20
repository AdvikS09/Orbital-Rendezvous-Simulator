import collections
import csv
import json
import math
import os
import sys

import v31paths as P

ROOT = P.ARCH
CODE = P.REPO
sys.path.insert(0, os.path.join(CODE, "experiments", "v3"))

import numpy as np

from cwvalid3 import calibration as CAL
from cwvalid3 import contracts as C31
from cwvalid3 import ledger as L
import v3data as VD

OUT = P.V31RES
V3 = os.path.join(ROOT, "v3")
PLAN = os.path.join(V3, "plan.json")
FREEZE = os.path.join(V3, "raw", "freeze.json")
MAN = os.path.join(V3, "raw", "manifests", "D1cal.jsonl")
RES = os.path.join(V3, "raw", "results", "D1cal.jsonl")
RER = os.path.join(V3, "raw", "results", "D1calreruns.jsonl")
LEDGER = os.path.join(V3, "ledger.jsonl")
CANDCSV = os.path.join(V3, "results", "candidates.csv")
RERUNCHK = os.path.join(V3, "raw", "rerunD1cal.json")
CELL, SPLIT, KIND = "D1", "cal", "continuous"
ADVLABELS = ("violation", "exit", "failure", "unresolved")
IDENT = ("case_id", "physical_id", "run_id", "cell", "split", "index")


def streamjsonl(path, key="run_id"):
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


def reshash(o):
    return L.stablehash({k: v for k, v in o.items() if k not in ("software", "completed_at")}, 32)


def derive(r, methods):
    tr = CAL.truth_record(r)
    d = {"worker_status": r.get("worker_status"), "truth_status": tr["truth_status"],
         "label_1m": CAL.critlabel(tr, 1.0), "label_0.1m": CAL.critlabel(tr, 0.1),
         "truth31": C31.truth_record(r, KIND)["truth_status"],
         "label31_1m": C31.critlabel(C31.truth_record(r, KIND), 1.0),
         "label31_0.1m": C31.critlabel(C31.truth_record(r, KIND), 0.1),
         **{f"f31_{m}": C31.feature_value(r, m, KIND) for m in methods},
         "arrival_status": CAL.arrlabel(r, tr)["arrival_status"], "tr": tr}
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
    plan = json.load(open(PLAN, encoding="utf-8"))
    frz = json.load(open(FREEZE, encoding="utf-8"))
    mult = plan["multiplicity"]
    level, alpha = mult["per_candidate_level"], mult["alpha"]
    fams = [f for f in plan["families"] if f["cell"] == CELL]
    methods = VD.meth2(CELL)

    prov = {"plan_sha256": L.filesha256(PLAN), "freeze_plan_sha256": frz["plan_sha256"],
            "D1_cal_results_sha256": L.filesha256(RES),
            "freeze_D1_cal_results_sha256": frz["calibration_results_sha256"][CELL],
            "D1_cal_reruns_sha256": L.filesha256(RER), "D1_cal_manifest_sha256": L.filesha256(MAN),
            "calibration_freeze_sha256": L.filesha256(FREEZE), "ledger_sha256": L.filesha256(LEDGER),
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

    manorder, man = [], {}
    for m in streamjsonl(MAN):
        manorder.append(m["run_id"])
        man[m["run_id"]] = {k: m.get(k) for k in IDENT}

    orig = {}
    crashraw = {}
    orighash = {}
    origmism = []
    origengine = collections.Counter()
    for r in streamjsonl(RES):
        rid = r["run_id"]
        orig[rid] = derive(r, methods)
        orighash[rid] = reshash(r)
        origengine[(r.get("software") or {}).get("engine_sha")] += 1
        if rid in man and any(r.get(k) != man[rid][k] for k in IDENT):
            origmism.append(rid)
        if r.get("worker_status") == "worker_crash":
            crashraw[rid] = {k: r.get(k) for k in IDENT + ("worker_status", "error", "completed_at")}
    rcorig = L.reconcile(manorder, [{"run_id": k} for k in orig], "run_id")

    rer = {}
    rerident = {}
    rerinfo = {}
    for r in streamjsonl(RER):
        rid = r["run_id"]
        rer[rid] = derive(r, methods)
        rerident[rid] = {k: r.get(k) for k in IDENT}
        rerinfo[rid] = {"worker_status": r.get("worker_status"), "cpu_s": r.get("cpu_s"), "wall_s": r.get("wall_s"),
                        "engine_sha": (r.get("software") or {}).get("engine_sha"), "software": r.get("software"),
                        "rerun_of": r.get("rerun_of"), "completed_at": r.get("completed_at"),
                        "result_hash32": reshash(r)}

    crashed = set(crashraw)
    rset = set(rer)
    ident = {
        "n_original_records": len(orig), "n_manifest": len(manorder),
        "original_vs_manifest_reconcile": {k: rcorig[k] for k in ("n_planned", "n_rows", "n_missing", "n_extra", "n_duplicates")},
        "original_identity_mismatch_vs_manifest": len(origmism),
        "n_original_worker_crash": len(crashed),
        "original_worker_status": dict(collections.Counter(d["worker_status"] for d in orig.values())),
        "n_reruns": len(rer),
        "rerun_ids_equal_crash_ids": rset == crashed,
        "reruns_not_in_crash_set": sorted(rset - crashed), "crash_ids_without_rerun": sorted(crashed - rset),
        "reruns_not_in_manifest": sorted(rset - set(man)),
        "rerun_identity_mismatch_vs_manifest": sorted(rid for rid in rset if rid in man and any(rerident[rid][k] != man[rid][k] for k in IDENT)),
        "rerun_identity_mismatch_vs_original_record": sorted(rid for rid in rset & crashed if any(rerident[rid][k] != crashraw[rid][k] for k in IDENT)),
        "rerun_worker_status": dict(collections.Counter(v["worker_status"] for v in rerinfo.values())),
        "rerun_engine_sha": dict(collections.Counter(v["engine_sha"] for v in rerinfo.values())),
        "original_engine_sha": {str(k): v for k, v in origengine.items()},
        "plan_engine_sha": plan["implementation"].get("engine_sha"),
        "rerun_software_distinct": [json.loads(s) for s in sorted({json.dumps(v["software"], sort_keys=True) for v in rerinfo.values()})],
        "original_crash_error_distinct": sorted({str(v["error"]) for v in crashraw.values()}),
        "crash_index_range": [min(v["index"] for v in crashraw.values()), max(v["index"] for v in crashraw.values())] if crashraw else None,
        "crash_indices": sorted(v["index"] for v in crashraw.values()),
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
    assert rcorig["n_missing"] == 0 and rcorig["n_extra"] == 0 and rcorig["n_duplicates"] == 0

    versions = {"original": [orig[k] for k in manorder],
                "completed": [rer.get(k, orig[k]) for k in manorder]}
    isrerun = np.array([k in rset for k in manorder])

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

    stored = {}
    with open(CANDCSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["cell"] == CELL:
                stored[(row["family_id"], int(row["cid"]))] = row

    csvrows, famsumm = [], []
    bench = {}
    for f in fams:
        fid = f["family_id"]
        per = {}
        for ver, recs in versions.items():
            v = np.array([d[f"f_{f['method']}"] for d in recs], float)
            labels = np.array([d[f"label_{f['criterion']}"] for d in recs], object)
            adv = VD.adverse(labels)
            adv2 = np.array([CAL.critlabel(d["tr"], f["tol_m"]) in ADVLABELS for d in recs])
            assert np.array_equal(adv, adv2)
            rows, best = CAL.calfam(v, adv, f["candidates"], alpha, level)
            v31 = np.array([d[f"f31_{f['method']}"] for d in recs], float)
            lab31 = [d[f"label31_{f['criterion']}"] for d in recs]
            C31.vallabels(lab31)
            adv31 = np.array([C31.isadverse(x) for x in lab31])
            rows31, best31 = CAL.calfam(v31, adv31, f["candidates"], alpha, level)
            assert np.array_equal(np.isnan(v), np.isnan(v31)) and np.array_equal(v[np.isfinite(v)], v31[np.isfinite(v31)])
            assert [(r["n_accepted"], r["n_adverse"], r["ucb"]) for r in rows] == [(r["n_accepted"], r["n_adverse"], r["ucb"]) for r in rows31]
            assert (best is None and best31 is None) or (best is not None and best31 is not None and best["cid"] == best31["cid"])
            for r in rows:
                acc = CAL.acceptmask(v, r["threshold"])
                C31.checklabelpart(C31.labelcounts(list(acc), lab31), int(acc.sum()))
            for r in rows:
                acc = CAL.acceptmask(v, r["threshold"])
                labacc = collections.Counter(labels[acc])
                r["acc_from_reruns"] = int((acc & isrerun).sum())
                r["adv_from_reruns"] = int((acc & adv & isrerun).sum())
                r["acc_by_label"] = {k: int(labacc.get(k, 0)) for k in ("ok",) + ADVLABELS}
            per[ver] = {"rows": rows, "best": best, "n_cal": int(v.size), "n_unavail": int((~np.isfinite(v)).sum())}
        s = frz["selection"][fid]
        bo, bc = per["original"]["best"], per["completed"]["best"]
        origmatch = {
            "status": s["status"] == ("selected" if bo else "abstain_no_admissible_candidate"),
            "threshold": s["threshold"] == (None if bo is None else bo["threshold"]),
            "cid": s["cid"] == (None if bo is None else bo["cid"]),
            "cal_accepted": s["cal_accepted"] == (None if bo is None else bo["n_accepted"]),
            "cal_adverse": s["cal_adverse"] == (None if bo is None else bo["n_adverse"]),
            "cal_ucb_exact": s["cal_ucb"] == (None if bo is None else bo["ucb"]),
            "cal_feature_unavailable": s["cal_feature_unavailable"] == per["original"]["n_unavail"],
            "n_cal": s["n_cal"] == per["original"]["n_cal"],
        }
        candmism = []
        for r in per["original"]["rows"]:
            st = stored.get((fid, r["cid"]))
            sel = bo is not None and r["cid"] == bo["cid"]
            if st is None:
                candmism.append((r["cid"], "missing in candidates.csv"))
                continue
            checks = {"threshold": float(st["threshold"]) == r["threshold"], "n_accepted": int(st["n_accepted"]) == r["n_accepted"],
                      "n_adverse": int(st["n_adverse"]) == r["n_adverse"], "ucb": float(st["ucb"]) == r["ucb"],
                      "admissible": (st["admissible"] == "True") == bool(r["admissible"]), "selected": (st["selected"] == "True") == sel,
                      "source": st["source"] == r["source"]}
            if not all(checks.values()):
                candmism.append((r["cid"], {k: v for k, v in checks.items() if not v}))
        nstored = sum(1 for k in stored if k[0] == fid)
        for ver in ("original", "completed"):
            best = per[ver]["best"]
            for r in per[ver]["rows"]:
                ro = next(x for x in per["original"]["rows"] if x["cid"] == r["cid"])
                csvrows.append({
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
        admo = [r["cid"] for r in per["original"]["rows"] if r["admissible"]]
        admc = [r["cid"] for r in per["completed"]["rows"] if r["admissible"]]
        flips = [{"cid": ro["cid"], "threshold": ro["threshold"], "original": {"n": ro["n_accepted"], "b": ro["n_adverse"], "ucb": ro["ucb"]},
                  "completed": {"n": rc["n_accepted"], "b": rc["n_adverse"], "ucb": rc["ucb"]},
                  "direction": "became_admissible" if rc["admissible"] else "became_inadmissible"}
                 for ro, rc in zip(per["original"]["rows"], per["completed"]["rows"]) if bool(ro["admissible"]) != bool(rc["admissible"])]
        famsumm.append({
            "family_id": fid, "method": f["method"], "criterion": f["criterion"], "tol_m": f["tol_m"],
            "n_candidates": len(f["candidates"]), "n_candidates_in_stored_csv": nstored,
            "freeze_selection": {k: s[k] for k in ("status", "cid", "source", "threshold", "cal_accepted", "cal_adverse", "cal_ucb",
                                                   "cal_feature_unavailable")},
            "original_selected": bsum(bo), "completed_selected": bsum(bc),
            "original_reproduces_freeze": origmatch, "original_reproduces_freeze_all": all(origmatch.values()),
            "original_matches_stored_candidate_table": not candmism, "stored_candidate_table_mismatches": candmism,
            "same_selected_candidate": (None if bo is None else bo["cid"]) == (None if bc is None else bc["cid"]),
            "same_selected_threshold": (None if bo is None else bo["threshold"]) == (None if bc is None else bc["threshold"]),
            "selected_delta_accepted": None if (bo is None or bc is None) else bc["n_accepted"] - bo["n_accepted"],
            "selected_delta_adverse": None if (bo is None or bc is None) else bc["n_adverse"] - bo["n_adverse"],
            "selected_ucb_original": None if bo is None else bo["ucb"], "selected_ucb_completed": None if bc is None else bc["ucb"],
            "selected_ucb_margin_to_alpha_original": None if bo is None else alpha - bo["ucb"],
            "selected_ucb_margin_to_alpha_completed": None if bc is None else alpha - bc["ucb"],
            "n_feature_unavailable_original": per["original"]["n_unavail"], "n_feature_unavailable_completed": per["completed"]["n_unavail"],
            "admissible_cids_original": admo, "admissible_cids_completed": admc,
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

    rchk = json.load(open(RERUNCHK, encoding="utf-8"))
    rchkcmp = []
    for e in rchk["selection_sensitivity"]:
        fs = next(x for x in famsumm if x["family_id"] == e["family_id"])
        bc = fs["completed_selected"]
        rchkcmp.append({"family_id": e["family_id"],
                        "threshold_match": e["threshold_with_reruns"] == (None if bc is None else bc["threshold"]),
                        "accepted_match": e["accepted_with_reruns"] == (None if bc is None else bc["n_accepted"]),
                        "adverse_match": e["adverse_with_reruns"] == (None if bc is None else bc["n_adverse"])})

    ev = collections.defaultdict(list)
    nlines = 0
    for r in streamjsonl(LEDGER, key="ledger_id"):
        nlines += 1
        if r.get("cell") == CELL and r.get("split") == SPLIT:
            ev[r["event"]].append(r)
    compids = collections.Counter(e["run_id"] for e in ev["completed"])
    crecids = collections.Counter(e["run_id"] for e in ev["completed_reconciled"])
    rerunids = collections.Counter(e["run_id"] for e in ev["rerun"])
    planids = collections.Counter(e["run_id"] for e in ev["planned"])
    perrun = collections.Counter()
    for c in (compids, crecids):
        perrun.update(c)
    manset = set(manorder)
    hashmism = [e["run_id"] for e in ev["completed"] + ev["completed_reconciled"]
                if e.get("result_sha256_16") is not None and orighash.get(e["run_id"]) != e["result_sha256_16"]]
    wsmism = [e["run_id"] for e in ev["completed"] + ev["completed_reconciled"]
              if orig.get(e["run_id"], {}).get("worker_status") != e.get("worker_status")]
    rerunmism = [e["run_id"] for e in ev["rerun"]
                 if e["run_id"] not in rerinfo or e.get("worker_status") != rerinfo[e["run_id"]]["worker_status"]
                 or e.get("cpu_s") != rerinfo[e["run_id"]]["cpu_s"] or e.get("wall_s") != rerinfo[e["run_id"]]["wall_s"]
                 or e.get("case_id") != man.get(e["run_id"], {}).get("case_id")]
    crashkind = collections.Counter(("completed" if rid in compids else "") + ("completed_reconciled" if rid in crecids else "")
                                    for rid in crashed)
    ledg = {
        "n_lines_total": nlines,
        "D1_cal_event_counts": {k: len(v) for k, v in ev.items()},
        "n_rerun_events": len(ev["rerun"]), "n_completed_reconciled_events": len(ev["completed_reconciled"]),
        "rerun_event_run_ids_equal_crash_ids": set(rerunids) == crashed and all(v == 1 for v in rerunids.values()),
        "rerun_event_worker_status": dict(collections.Counter(e.get("worker_status") for e in ev["rerun"])),
        "rerun_event_engine_sha": dict(collections.Counter(e.get("engine_sha") for e in ev["rerun"])),
        "rerun_events_with_result_hash": sum(1 for e in ev["rerun"] if "result_sha256_16" in e),
        "rerun_event_vs_rerun_record_mismatch": rerunmism,
        "completed_reconciled": [{k: e.get(k) for k in ("run_id", "case_id", "worker_status", "result_sha256_16", "time", "note")} for e in ev["completed_reconciled"]],
        "completed_reconciled_run_in_crash_set": all(e["run_id"] in crashed for e in ev["completed_reconciled"]),
        "completed_reconciled_hash_matches_record": all(orighash.get(e["run_id"]) == e.get("result_sha256_16") for e in ev["completed_reconciled"]),
        "planned_events_equal_manifest": set(planids) == manset and all(v == 1 for v in planids.values()),
        "completed_or_reconciled_cover_manifest_exactly_once": set(perrun) == manset and all(v == 1 for v in perrun.values()),
        "n_run_ids_with_completed": len(compids), "n_run_ids_with_completed_reconciled": len(crecids),
        "n_run_ids_with_both": len(set(compids) & set(crecids)),
        "n_manifest_ids_without_completion_event": len(manset - set(perrun)),
        "n_completion_ids_not_in_manifest": len(set(perrun) - manset),
        "max_completion_events_per_run": max(perrun.values()) if perrun else None,
        "completed_event_worker_status": dict(collections.Counter(e.get("worker_status") for e in ev["completed"])),
        "crash_runs_by_completion_event_kind": dict(crashkind),
        "completion_result_hash_mismatches_vs_stored_records": len(hashmism),
        "completion_result_hash_mismatch_ids": hashmism[:20],
        "completion_worker_status_mismatches_vs_stored_records": len(wsmism),
    }

    summary = {
        "finding_id": "C06", "cell": CELL, "split": SPLIT,
        "command": "python code/experiments/v31/c06addendum.py",
        "provenance": prov, "identity_checks": ident, "benchmark_D1_A_1m": bench,
        "families": famsumm,
        "all_original_reproduce_freeze": all(x["original_reproduces_freeze_all"] for x in famsumm),
        "all_original_match_stored_candidate_table": all(x["original_matches_stored_candidate_table"] for x in famsumm),
        "n_families_selection_changed": sum(1 for x in famsumm if not x["same_selected_candidate"]),
        "n_candidates_admissibility_flipped": sum(len(x["admissibility_flips"]) for x in famsumm),
        "n_candidates_total": sum(x["n_candidates"] for x in famsumm),
        "n_candidate_rows_gaining_adverse_cases": sum(1 for r in csvrows if r["version"] == "completed" and r["delta_n_adverse_vs_original"] > 0),
        "n_candidate_rows_gaining_adverse_cases_admissible_original": sum(
            1 for r in csvrows if r["version"] == "completed" and r["delta_n_adverse_vs_original"] > 0 and
            next(x for x in csvrows if x["version"] == "original" and x["family_id"] == r["family_id"] and x["cid"] == r["cid"])["admissible"]),
        "contract_v31_labelling_gives_identical_counts_and_selections": True,
        "status": "post-test sensitivity (descriptive); the frozen selection is unchanged and remains the reported one",
        "outcome_independence_of_the_43": "not established by the record; not relied on",
        "repository_rerun_check_comparison": rchkcmp,
        "census": census, "ledger": ledg,
    }
    with open(os.path.join(OUT, "D1addendum.json"), "w", encoding="utf-8") as fh:
        json.dump(L.jsonable(summary), fh, indent=1, default=jnum)
    cols = list(csvrows[0].keys())
    with open(os.path.join(OUT, "D1addendum.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in csvrows:
            w.writerow(r)
    print("plan/freeze sha match:", prov["plan_sha_match"], " D1cal results sha match freeze:", prov["D1_cal_results_sha_match"])
    print("frozen code:", {k: v["match"] for k, v in prov["frozen_code_check"].items()})
    print("identity one-to-one:", ident["one_to_one_ok"], " rerun ws:", ident["rerun_worker_status"])
    print("all original reproduce freeze:", summary["all_original_reproduce_freeze"],
          " match stored csv:", summary["all_original_match_stored_candidate_table"])
    for x in famsumm:
        o, c = x["original_selected"], x["completed_selected"]
        f = lambda b: "abstain" if b is None else f"cid={b['cid']} t={b['threshold']!r} n={b['n_accepted']} b={b['n_adverse']} U={b['ucb']:.9f}"
        print(f"{x['family_id']:12s} orig[{f(o)}]  comp[{f(c)}]  same={x['same_selected_candidate']} flips={len(x['admissibility_flips'])} "
              f"repro={x['original_reproduces_freeze_all']}")
    print("benchmark:", bench.get("original_ok"), bench.get("completed_ok"))
    print("census original:", census["original"]["truth_status"], census["original"]["label_1m"], census["original"]["label_0.1m"])
    print("census completed:", census["completed"]["truth_status"], census["completed"]["label_1m"], census["completed"]["label_0.1m"])
    print("census reruns:", census["reruns_only"])
    print("ledger:", json.dumps(L.jsonable(ledg), default=jnum)[:3000])
    print("rerun-check comparison:", rchkcmp)
    print("csv rows:", len(csvrows))


if __name__ == "__main__":
    main()
