"""C01 targeted replay: every held record with a physical-domain exit (D3/D4, all splits incl. the
transfer cohort; 341 records in v3.0) is re-executed

  (a) with the v3.1 campaign worker (corrected exit storage, staged worker, canonical record), and
  (b) with the v3.0 engine extracted from git (default commit 2aa8c44, engine ae0c780aaa672ad4), main
      batch and metrics only, to show that the replay reproduces the stored v3.0 values exactly.

Cases are re-batched per cell in chunks of --chunk in manifest order (cell, split, index); the original
campaign batches are not reproduced (stored v3.0 values were reproduced bit for bit under re-batching
in the audit). Records are keyed by the original run_id / case_id / physical_id; the v3.1 engine identity
is recorded separately (software.engine_sha), the v3.0 ids are not recomputed.

Outputs (v3_1_correction):
  raw/results/held_exit_replay_v31.jsonl        canonical v3.1 records (one per exit case)
  raw/results/held_exit_replay_v30_engine.jsonl canonical case summaries from the v3.0 engine
  raw/REPLAY_LEDGER.jsonl                       one event per v3.1 record (record/physics SHA-256)
  results/exit_correction_impact.csv            old/new values per record and comparison flags
  results/exit_correction_summary.json          totals, maxima, unexpected changes
Usage: python code/experiments/v3_1/c01_exit_replay.py --v30-src <dir with git archive of 2aa8c44> [--workers 4]
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import v31paths as P  # noqa: E402

HELD_SPLITS = ("pilot", "dev", "cal", "test", "transfer_cohort")
PAIRS = (("cw", "eci"), ("cw", "nerm"), ("nerm", "eci"))
PK = ("peak_m", "t_peak_s", "peak_grid_m", "common_end_s")
EXPECTED_CHANGE = {f"{a}_vs_{b}_{k}" for a, b in PAIRS for k in ("peak_m", "t_peak_s", "peak_grid_m")}
NEW_V31 = {"nerm_exit_step", "nerm_exit_fallback", "eci_exit_step", "eci_exit_fallback", "feature_stage"}


# ------------------------------------------------------------------ worker processes
def _init(src):
    sys.path.insert(0, src)
    sys.path.insert(0, os.path.join(src, "experiments", "v3"))


_GOT = []


def _capture_batches():
    """Keep a reference to every HeldBatch run in this process (patched once per process)."""
    import cwvalid3.sampled as SA
    if not getattr(SA.HeldBatch, "_v31_capture", False):
        orig = SA.HeldBatch.run

        def run(self):
            out = orig(self)
            _GOT.append(self)
            return out
        SA.HeldBatch.run = run
        SA.HeldBatch._v31_capture = True
    _GOT.clear()
    return _GOT


def _exit_diag(hb, i):
    """Exit step, position in step and dense radius minus floor at t_end for nerm and eci."""
    import numpy as np
    d = {}
    for b in ("nerm", "eci"):
        te = float(hb.t_end[b][i])
        if hb.status[b][i] != "domain_exit":
            d.update({f"{b}_exit_step_diag": -1, f"{b}_s_in_step": math.nan, f"{b}_floor_gap_m": math.nan,
                      f"{b}_stored_end_equals_start": None})
            continue
        j = int(np.clip(np.searchsorted(hb.t, te, side="right") - 1, 0, hb.S - 1))
        if te == hb.t[j] and j > 0 and getattr(hb, "exit_step", None) is not None:
            j = int(hb.exit_step[b][i])
        y = hb.dense(i, [te])
        cols = hb.cols(b)
        d.update({f"{b}_exit_step_diag": j, f"{b}_s_in_step": (te - hb.t[j]) / (hb.t[j + 1] - hb.t[j]),
                  f"{b}_floor_gap_m": float(hb.radius(y, b)[0] - hb.r_floor),
                  f"{b}_stored_end_equals_start": bool(np.array_equal(hb.Y[j + 1, i, cols], hb.Y[j, i, cols]))})
    return d


def task_v31(chunk):
    import campaign as CP
    got = _capture_batches()
    t0 = time.process_time()
    outs = CP.work_held(chunk)
    main = [b for b in got if b.has_eci]
    if len(main) != 1 or main[0].N != len(chunk):
        raise RuntimeError(f"expected one main batch of {len(chunk)} cases, got {[b.N for b in main]}")
    diag = [_exit_diag(main[0], i) for i in range(len(chunk))]
    return outs, diag, CP.software(), time.process_time() - t0


def task_v30(chunk):
    import numpy as np
    from cwvalid3.domains import CELLS
    from cwvalid3.held_metrics import case_summary
    from cwvalid3.sampled import HeldBatch
    import campaign as CP
    t0 = time.process_time()
    cell = chunk[0]["cell"]
    Ts, alt = CELLS[cell]["Ts"], CELLS[cell]["altitude"]
    hb = HeldBatch([r["scenario"]["rel0"] for r in chunk], np.array([r["K"] for r in chunk]),
                   [r["scenario"]["a_max"] for r in chunk], alt, Ts, horizon_orbits=1.0, h=min(Ts, 2.0),
                   models=("cw", "nerm", "eci"), estimators=("A_H", "B_H", "C_H")).run()
    outs = []
    for i, r in enumerate(chunk):
        sm, _ = case_summary(hb, i, pairs=PAIRS)
        outs.append({**{k: r[k] for k in ("case_id", "physical_id", "run_id", "cell", "split", "index")}, **sm})
    diag = [_exit_diag(hb, i) for i in range(len(chunk))]
    return outs, diag, {"engine_sha": CP.engine_version(), "engine_formula": "v3.0 ledger.code_version"}, time.process_time() - t0


# ------------------------------------------------------------------ main process
def load_exit_cases():
    cases = []
    for cell in ("D3", "D4"):
        for sp in HELD_SPLITS:
            res = {}
            with open(os.path.join(P.RES, f"{cell}_{sp}.jsonl"), encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get("nerm_status") == "domain_exit" or r.get("eci_status") == "domain_exit":
                        res[r["run_id"]] = r
            with open(os.path.join(P.MAN, f"{cell}_{sp}.jsonl"), encoding="utf-8") as fh:
                for line in fh:
                    m = json.loads(line)
                    if m["run_id"] in res:
                        cases.append((m, res.pop(m["run_id"])))
            if res:
                raise SystemExit(f"{cell}/{sp}: {len(res)} exit records without a manifest entry")
    return cases


def git_head():
    r = subprocess.run(["git", "-C", P.REPO, "rev-parse", "HEAD"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git rev-parse failed: {r.stderr}")
    dirty = subprocess.run(["git", "-C", P.REPO, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
    return r.stdout.strip(), bool(dirty.stdout.strip())


def nz(x):
    return None if (isinstance(x, float) and math.isnan(x)) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v30-src", required=True, help="directory holding `git archive 2aa8c44` of code/")
    ap.add_argument("--v30-commit", default="2aa8c44")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=45)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    from cwvalid3 import calibration as CAL
    from cwvalid3 import contracts as C
    from cwvalid3 import serialization as SER
    head, dirty = git_head()
    if dirty and not a.limit:
        raise SystemExit("commit the v3.1 code before the replay (tracked files are modified)")
    cases = load_exit_cases()
    if a.limit:
        cases = cases[:a.limit]
    chunks = []
    for cell in ("D3", "D4"):
        cc = [m for m, _ in cases if m["cell"] == cell]
        chunks += [cc[i:i + a.chunk] for i in range(0, len(cc), a.chunk)]
    stored = {m["run_id"]: r for m, r in cases}
    print(f"{len(cases)} exit cases in {len(chunks)} chunks; v3.1 HEAD {head}", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(P.REPO,)) as ex1, \
            ProcessPoolExecutor(a.workers, initializer=_init, initargs=(os.path.abspath(a.v30_src),)) as ex0:
        f1 = [ex1.submit(task_v31, c) for c in chunks]
        f0 = [ex0.submit(task_v30, c) for c in chunks]
        r1 = [f.result() for f in f1]
        r0 = [f.result() for f in f0]
    wall = time.perf_counter() - t0
    cpu1, cpu0 = sum(x[3] for x in r1), sum(x[3] for x in r0)
    sw1, sw0 = r1[0][2], r0[0][2]
    print(f"done in {wall:.0f} s wall; worker cpu v3.1 {cpu1:.0f} s, v3.0 {cpu0:.0f} s", flush=True)

    os.makedirs(os.path.join(P.V31_RAW, "results"), exist_ok=True)
    f31 = P.guard_v30(os.path.join(P.V31_RAW, "results", "held_exit_replay_v31.jsonl"))
    f30 = P.guard_v30(os.path.join(P.V31_RAW, "results", "held_exit_replay_v30_engine.jsonl"))
    fl = P.guard_v30(os.path.join(P.V31_RAW, "REPLAY_LEDGER.jsonl"))
    rows = []
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(f31, "w", encoding="utf-8", newline="\n") as h31, open(f30, "w", encoding="utf-8", newline="\n") as h30, \
            open(fl, "w", encoding="utf-8", newline="\n") as hl:
        for k, (chunk, (o1, d1, _, _), (o0, d0, _, _)) in enumerate(zip(chunks, r1, r0)):
            for m, n31, g31, n30, g30 in zip(chunk, o1, d1, o0, d0):
                old = stored[m["run_id"]]
                rep = {"kind": "targeted_replay_C01", "of_run_id": m["run_id"], "chunk": k, "chunk_size": len(chunk),
                       "v31_git_head": head, "engine_sha_v31": sw1["engine_sha"], "stored_engine_sha":
                       (old.get("software") or {}).get("engine_sha"), "replayed_at": stamp}
                n31 = {**n31, "software": sw1, "completed_at": stamp, "replay": rep}
                rh, ph = SER.write_record(h31, n31)
                SER.write_record(h30, {**n30, "replay": {**rep, "kind": "v30_engine_fidelity", "v30_commit": a.v30_commit,
                                                         "engine_sha_v30": sw0["engine_sha"]}}, schema=None)
                SER.write_record(hl, {"ledger_id": f"replay_c01:{m['run_id']}", "event": "replayed", "of_run_id": m["run_id"],
                                      "case_id": m["case_id"], "cell": m["cell"], "split": m["split"],
                                      "worker_status": n31.get("worker_status"), "record_sha256": rh, "physics_sha256": ph,
                                      "record_schema": SER.RECORD_SCHEMA, "engine_sha": sw1["engine_sha"],
                                      "results_file": os.path.basename(f31), "time": stamp}, schema=None)
                A, B, S = SER.normalize(n31), SER.normalize(n30), SER.normalize(old)
                vol = SER.is_volatile
                d30 = sorted(x for x in B if not vol(x) and x not in ("replay",) and x in S and B[x] != S[x])
                d31 = sorted(x for x in A if not vol(x) and x not in NEW_V31 and x in S and A[x] != S[x])
                miss31 = sorted(x for x in S if not vol(x) and x not in A)
                unexpected = sorted(set(d31) - EXPECTED_CHANGE) + [f"missing:{x}" for x in miss31]
                # a field that differs from the stored value identically in the v3.0-engine replay is a
                # replay (re-batching / producing-engine) artefact, not an effect of the C01 correction
                not_explained = [x for x in unexpected if x.startswith("missing:") or x not in B or A.get(x) != B.get(x)]
                rel30 = max((abs(B[x] - S[x]) / max(abs(S[x]), 1e-300) for x in d30
                             if isinstance(B[x], float) and isinstance(S[x], float)), default=0.0)
                kind = "held"
                t_old, t_new = C.truth_record(old, kind), C.truth_record(n31, kind)
                c_old, c_new = CAL.truth_record(old), CAL.truth_record(n31)
                row = {"cell": m["cell"], "split": m["split"], "index": m["index"], "case_id": m["case_id"],
                       "physical_id": m["physical_id"], "run_id": m["run_id"], "stored_engine_sha": rep["stored_engine_sha"],
                       "engine_sha_v31": sw1["engine_sha"], "engine_sha_v30_replay": sw0["engine_sha"],
                       "worker_status_v31": n31.get("worker_status"), "chunk": k}
                for b in ("nerm", "eci"):
                    row.update({f"{b}_status_stored": old.get(f"{b}_status"), f"{b}_status_v31": n31.get(f"{b}_status"),
                                f"{b}_t_end_stored_s": old.get(f"{b}_t_end_s"), f"{b}_t_end_v31_s": n31.get(f"{b}_t_end_s"),
                                f"{b}_t_end_equal": old.get(f"{b}_t_end_s") == n31.get(f"{b}_t_end_s"),
                                f"{b}_exit_step": n31.get(f"{b}_exit_step"), f"{b}_exit_fallback": n31.get(f"{b}_exit_fallback"),
                                f"{b}_s_in_step": g31[f"{b}_s_in_step"],
                                f"{b}_floor_gap_v30_m": g30[f"{b}_floor_gap_m"], f"{b}_floor_gap_v31_m": g31[f"{b}_floor_gap_m"],
                                f"{b}_v30_stored_end_equals_start": g30[f"{b}_stored_end_equals_start"],
                                f"{b}_v31_stored_end_equals_start": g31[f"{b}_stored_end_equals_start"]})
                for p in ("cw_vs_eci", "cw_vs_nerm", "nerm_vs_eci"):
                    for f in PK:
                        row[f"{p}_{f}_stored"] = old.get(f"{p}_{f}")
                        row[f"{p}_{f}_v31"] = n31.get(f"{p}_{f}")
                    row[f"{p}_peak_delta_m"] = n31[f"{p}_peak_m"] - old[f"{p}_peak_m"]
                row.update({"disc_m_stored": nz(c_old["disc_m"]), "disc_m_v31": nz(c_new["disc_m"]),
                            "u_num_m_stored": nz(c_old["u_num_m"]), "u_num_m_v31": nz(c_new["u_num_m"]),
                            "truth_v30_stored": c_old["truth_status"], "truth_v30_rules_on_v31": c_new["truth_status"],
                            "truth_contract_stored": t_old["truth_status"], "truth_contract_v31": t_new["truth_status"],
                            "label_1m_stored": CAL.criterion_label(c_old, 1.0), "label_1m_v31": C.criterion_label(t_new, 1.0),
                            "label_0.1m_stored": CAL.criterion_label(c_old, 0.1), "label_0.1m_v31": C.criterion_label(t_new, 0.1)})
                for meth in ("A_H", "B_H", "A_naive", "C_H", "sep", "sat"):
                    fo, fn = CAL.feature_value(old, meth, "held"), C.feature_value(n31, meth, "held")
                    row[f"f_{meth}_equal"] = (fo == fn) or (math.isnan(fo) and math.isnan(fn))
                row.update({"v30_replay_fields_differing_from_stored": ";".join(d30), "n_v30_replay_diff": len(d30),
                            "v31_changed_fields": ";".join(d31), "v31_unexpected_changes": ";".join(unexpected),
                            "n_v31_unexpected": len(unexpected), "v31_unexpected_not_in_v30_replay": ";".join(not_explained),
                            "n_v31_unexpected_not_in_v30_replay": len(not_explained), "v30_replay_max_rel_diff": rel30,
                            "record_sha256_v31": rh, "physics_sha256_v31": ph})
                rows.append(row)
    with open(P.guard_v30(os.path.join(P.V31_RES, "exit_correction_impact.csv")), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def rng(key, sub=None):
        v = [r[key] for r in rows if (sub is None or sub(r)) and r[key] is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
        return {"n": len(v), "min": min(v), "max": max(v), "median": sorted(v)[len(v) // 2]} if v else {"n": 0}
    ae0 = lambda r: r["stored_engine_sha"] == "ae0c780aaa672ad4"
    summ = {"n_records": len(rows), "by_cell_split": {}, "wall_s": wall, "worker_cpu_s": {"v31": cpu1, "v30": cpu0},
            "v31_git_head": head, "engine_sha_v31": sw1["engine_sha"], "engine_sha_v30_replay": sw0["engine_sha"],
            "v30_commit": a.v30_commit, "chunk": a.chunk,
            "stored_engine_sha_counts": {}, "n_v30_replay_differs_from_stored": sum(r["n_v30_replay_diff"] > 0 for r in rows),
            "n_v31_unexpected_changes": sum(r["n_v31_unexpected"] > 0 for r in rows),
            "n_v31_unexpected_not_in_v30_replay": sum(r["n_v31_unexpected_not_in_v30_replay"] > 0 for r in rows),
            "v30_replay_max_rel_diff": max(r["v30_replay_max_rel_diff"] for r in rows),
            "v30_replay_diff_by_stored_engine": {e: sum(r["n_v30_replay_diff"] > 0 for r in rows if r["stored_engine_sha"] == e)
                                                 for e in sorted({r["stored_engine_sha"] for r in rows})},
            "statuses_unchanged": all(r["nerm_status_stored"] == r["nerm_status_v31"] and r["eci_status_stored"] == r["eci_status_v31"] for r in rows),
            "t_end_bitwise_unchanged": all(r["nerm_t_end_equal"] and r["eci_t_end_equal"] for r in rows),
            "truth_unchanged": all(r["truth_contract_stored"] == r["truth_contract_v31"] == "physical_domain_exit" for r in rows),
            "labels_unchanged": all(r["label_1m_stored"] == r["label_1m_v31"] == "exit" and r["label_0.1m_stored"] == r["label_0.1m_v31"] == "exit" for r in rows),
            "features_unchanged": all(r[f"f_{m}_equal"] for r in rows for m in ("A_H", "B_H", "A_naive", "C_H", "sep", "sat")),
            "exit_fallbacks": sum(bool(r["nerm_exit_fallback"]) or bool(r["eci_exit_fallback"]) for r in rows),
            "v30_stored_end_equals_start": sum(bool(r["nerm_v30_stored_end_equals_start"]) and bool(r["eci_v30_stored_end_equals_start"]) for r in rows),
            "v31_stored_end_equals_start": sum(bool(r["nerm_v31_stored_end_equals_start"]) or bool(r["eci_v31_stored_end_equals_start"]) for r in rows),
            "floor_gap_v30_m": {b: rng(f"{b}_floor_gap_v30_m") for b in ("nerm", "eci")},
            "floor_gap_v31_abs_max_m": {b: max(abs(r[f"{b}_floor_gap_v31_m"]) for r in rows) for b in ("nerm", "eci")},
            "cw_vs_eci_peak_delta_m": {"all": rng("cw_vs_eci_peak_delta_m"), "ae0c780_records": rng("cw_vs_eci_peak_delta_m", ae0),
                                      "D3": rng("cw_vs_eci_peak_delta_m", lambda r: r["cell"] == "D3"),
                                      "D4": rng("cw_vs_eci_peak_delta_m", lambda r: r["cell"] == "D4")},
            "nerm_vs_eci_peak_m": {"stored": rng("nerm_vs_eci_peak_m_stored"), "v31": rng("nerm_vs_eci_peak_m_v31")},
            "u_num_m": {"stored": rng("u_num_m_stored"), "v31": rng("u_num_m_v31")},
            "n_peak_at_common_end": {"stored": sum(r["cw_vs_eci_t_peak_s_stored"] == r["cw_vs_eci_common_end_s_stored"] for r in rows),
                                     "v31": sum(r["cw_vs_eci_t_peak_s_v31"] == r["cw_vs_eci_common_end_s_v31"] for r in rows)},
            "outputs": {"v31_records": os.path.relpath(f31, P.V3), "v30_engine_records": os.path.relpath(f30, P.V3),
                        "ledger": os.path.relpath(fl, P.V3)}}
    for r in rows:
        key = f"{r['cell']}/{r['split']}"
        summ["by_cell_split"][key] = summ["by_cell_split"].get(key, 0) + 1
        summ["stored_engine_sha_counts"][r["stored_engine_sha"]] = summ["stored_engine_sha_counts"].get(r["stored_engine_sha"], 0) + 1
    with open(P.guard_v30(os.path.join(P.V31_RES, "exit_correction_summary.json")), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
