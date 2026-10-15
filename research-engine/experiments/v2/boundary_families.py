"""v2 designed challenge: 24 new arrival-boundary families.

Step 1  declare     write the predeclared family list, search protocol and diagnostic-point
                    rule (families.json, SHA-256) BEFORE any nonlinear propagation.
Step 2  search      for every family: log sweep of the initial separation, dense interior
                    sampling of every grid cell within +/-3 cells of a label change or a CW/
                    reference disagreement, then bisection of every CW and every reference
                    arrival-label change to a relative bracket width of 2e-5. EVERY propagation
                    (sweep, interior, bisection) is appended to search/<family>.jsonl with its
                    purpose, statuses and termination reasons.
Step 2b refine2     (added) second bisection stage to a relative width of 1e-7 for boundary pairs whose
                    offset is below 4 x the combined bracket half-width (refine where feasible)
Step 2c explore     (EXPLORATORY, decided after the search) extend the separation range beyond 80 km for
                    families whose CW boundary lies outside the predeclared range
Step 3  diagnostic  place the predeclared diagnostic points (see DIAG_RULE) and run them as an
                    ordinary paired campaign (CW, inertial reference, NERM cross-check, second
                    order) in diagnostic/.

Families use one-orbit horizons, the production solver and the primary terminal-arrival
criterion (1 m, 0.01 m/s, >= 300 s remaining), exactly as the main study. Initial velocity
for scaled-velocity families is v0 = nu * n * rho0 * w (w a fixed unit vector), so that
every family is a one-parameter line in rho0.

This is a designed challenge that uses nonlinear information to place cases; its mismatch
frequency is not a mission probability, and nearby points of one family are dependent.

Step 3b supplementary (EXPLORATORY) diagnostic points for disagreement regions the predeclared rule does
                    not cover (CW-only arrival islands; bands bounded by an exploratory boundary)

Usage: python experiments/v2/boundary_families.py declare|search|refine2|explore|diagnostic|supplementary [--workers 7]
Output: results/v2/raw/boundary_families/{families.json, families_record.json, search/, brackets.csv, diagnostic/}
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ext"))
import v2_common as C  # noqa: E402

OUT = os.path.join(C.RAW, "boundary_families")
FAM_FILE = os.path.join(OUT, "families.json")

SWEEP = {"lo_m": 200.0, "hi_m": 80000.0, "n": 64}            # ~10% log spacing
INTERIOR = {"cells_each_side": 3, "points_per_cell": 9}       # ~1% spacing near boundaries
BISECT_TOL = 2e-5                                            # relative bracket width
DIAG_RULE = ("for each resolved CW/reference boundary pair (|offset| >= 2 x combined bracket half-width): "
             "points 0.5% below the lower end and 0.5% above the upper end of the disagreement band, plus, "
             "if the band is wider than 4 x BISECT_TOL, points at 1/4, 1/2 and 3/4 of the band (log scale); "
             "for an unresolved pair: the two points 0.5% below and above the merged boundary; "
             "families without a boundary in the search range get no diagnostic points (retained in the report)")


def _u(v):
    v = np.asarray(v, float)
    return (v / np.linalg.norm(v)).tolist()


MA, MB, MC = _u([1, 1, 1]), _u([-1, 2, -1]), _u([2, -1, 2])
# (id, label, direction, velocity rule, nu, velocity direction, wn/n, a_max)
FAMILIES = [
    ("F01", "+R", [1, 0, 0], "zero", 0.0, None, 4.5, 0.035),
    ("F02", "-R", [-1, 0, 0], "zero", 0.0, None, 4.5, 0.035),
    ("F03", "+R", [1, 0, 0], "zero", 0.0, None, 6.5, 0.075),
    ("F04", "-R", [-1, 0, 0], "zero", 0.0, None, 6.5, 0.075),
    ("F05", "+R", [1, 0, 0], "zero", 0.0, None, 7.5, 0.015),
    ("F06", "-R", [-1, 0, 0], "zero", 0.0, None, 7.5, 0.015),
    ("F07", "+V", [0, 1, 0], "zero", 0.0, None, 5.5, 0.015),
    ("F08", "-V", [0, -1, 0], "zero", 0.0, None, 5.5, 0.015),
    ("F09", "+V", [0, 1, 0], "zero", 0.0, None, 7.5, 0.075),
    ("F10", "-V", [0, -1, 0], "zero", 0.0, None, 6.5, 0.035),
    ("F11", "+H", [0, 0, 1], "zero", 0.0, None, 4.5, 0.075),
    ("F12", "-H", [0, 0, -1], "zero", 0.0, None, 5.5, 0.035),
    ("F13", "+H", [0, 0, 1], "zero", 0.0, None, 7.5, 0.035),
    ("F14", "mixed A", MA, "zero", 0.0, None, 4.5, 0.015),
    ("F15", "mixed B", MB, "zero", 0.0, None, 6.5, 0.015),
    ("F16", "mixed C", MC, "zero", 0.0, None, 5.5, 0.075),
    ("F17", "+R", [1, 0, 0], "em1", None, None, 5.5, 0.035),
    ("F18", "-R", [-1, 0, 0], "em1", None, None, 6.5, 0.035),
    ("F19", "-R", [-1, 0, 0], "em1", None, None, 7.5, 0.035),
    ("F20", "mixed A", MA, "scaled", 0.5, [-v for v in MA], 7.5, 0.075),
    ("F21", "mixed B", MB, "scaled", 1.0, _u(np.cross(MB, [0, 0, 1])), 4.5, 0.015),
    ("F22", "+V", [0, 1, 0], "scaled", 0.5, [0, -1, 0], 6.5, 0.015),
    ("F23", "+H", [0, 0, 1], "scaled", 0.5, [0, 1, 0], 5.5, 0.075),
    ("F24", "mixed C", MC, "scaled", 0.3, MC, 4.5, 0.035),
]
COVERAGE = (
    "24 one-orbit families, none sharing a (wn/n, a_max) pair with any archived scenario (checked against all "
    "766 settings in results/**/scenarios.jsonl). Controller settings: every combination of wn/n in {4.5, 5.5, 6.5, "
    "7.5} and a_max in {0.015, 0.035, 0.075} m/s^2 appears at least once (12 combinations, 10 of them twice). "
    "Directions: +radial and -radial with zero velocity at three settings each (F01-F06); +/-along-track (F07-F10); "
    "+/-cross-track (F11-F13); three fixed mixed directions A=(1,1,1)/sqrt3, B=(-1,2,-1)/sqrt6, C=(2,-1,2)/3 with zero "
    "velocity (F14-F16). Nonzero initial velocity: first-order energy-matched radial offsets ydot0 = -2 n x0 for both "
    "radial signs (F17-F19) and scaled velocities v0 = nu n rho0 w: closing along -rho (F20, nu 0.5), transverse "
    "(F21, nu 1.0), closing along-track (F22, nu 0.5), along-track drift of a cross-track offset (F23, nu 0.5) and "
    "opening along +rho (F24, nu 0.3). Radial families are over-represented because the earlier study found the "
    "largest disagreement bands there; along-track, cross-track and mixed families test whether the near-zero offsets "
    "seen there persist.")


def family_scenario(fam, sep):
    from cwvalid.constants import DEFAULT_RADIUS, mean_motion
    from cwvalid.scenario import Scenario
    fid, lab, d, vrule, nu, w, wn, am = fam
    n = mean_motion(DEFAULT_RADIUS)
    pos = np.asarray(d, float) * sep
    if vrule == "zero":
        vel = np.zeros(3)
        nu_eff = 0.0
    elif vrule == "em1":
        vel = np.array([0.0, -2.0 * n * pos[0], 0.0])
        nu_eff = float(np.linalg.norm(vel) / (n * sep))
    else:
        vel = nu * n * sep * np.asarray(w, float)
        nu_eff = nu
    meta = {"family_id": fid, "axis": lab if lab in ("+R", "-R", "+V", "-V", "+H", "-H") else "mixed",
            "direction_label": lab, "sep_m": float(sep), "velocity": vrule, "nu": nu_eff, "dir": [float(v) for v in d],
            "variant": fid, "in_dev_domain": True}
    return Scenario("v2fam", tuple(np.concatenate((pos, vel))), wn, am, 1.0, meta=meta)


def cmd_declare():
    if os.path.exists(FAM_FILE):
        sys.exit("families already declared; refusing to overwrite")
    os.makedirs(OUT, exist_ok=True)
    fams = [{"id": f[0], "direction_label": f[1], "direction": f[2], "velocity_rule": f[3], "nu": f[4],
             "velocity_direction": f[5], "wn_ratio": f[6], "a_max": f[7]} for f in FAMILIES]
    # archive check: no archived scenario shares a (wn, a_max) pair with a family
    pairs = set()
    import glob
    for p in glob.glob(os.path.join(C.RESULTS, "**", "scenarios.jsonl"), recursive=True):
        if os.sep + "v2" + os.sep in p:
            continue
        for line in open(p):
            if line.strip():
                s = json.loads(line)
                pairs.add((round(s["wn_ratio"], 9), None if s["a_max"] is None else round(s["a_max"], 9)))
    clash = [f["id"] for f in fams if (round(f["wn_ratio"], 9), round(f["a_max"], 9)) in pairs]
    if clash:
        sys.exit(f"families clash with archived settings: {clash}")
    doc = {"declared_at": time.strftime("%Y-%m-%d %H:%M:%S"), "families": fams, "coverage": COVERAGE,
           "search": {"sweep": SWEEP, "interior": INTERIOR, "bisect_rel_tol": BISECT_TOL,
                      "labels": "primary terminal arrival (1 m, 0.01 m/s, >= 300 s remaining), one orbit",
                      "models": "CW and inertial reference (+ NERM cross-check) on sweep and interior points; CW alone "
                                "for CW-boundary bisection; reference + NERM for reference-boundary bisection"},
           "diagnostic_rule": DIAG_RULE, "frozen_v1_sha256": C.sha256(C.FROZEN_V1),
           "archive_settings_checked": len(pairs), "note": "declared before any nonlinear propagation of these families"}
    C.write_json(FAM_FILE, doc)
    rec = {"families_sha256": C.sha256(FAM_FILE), "code": C.code_revision(), "written": doc["declared_at"]}
    C.write_json(os.path.join(OUT, "families_record.json"), rec)
    print(json.dumps(rec, indent=2))


def _run(scn, models):
    from cwvalid.experiment import RunConfig, run_scenario
    if models == "pair":
        cfg = RunConfig(models=("cw", "eci", "nerm"), reference="eci", crosscheck="nerm")
    elif models == "cw":
        cfg = RunConfig(models=("cw",), reference="cw", crosscheck=None)
    else:
        cfg = RunConfig(models=("eci", "nerm"), reference="eci", crosscheck="nerm")
    t0 = time.perf_counter()
    try:
        row, _ = run_scenario(scn, cfg)
        row["run_ok"] = True
    except Exception:
        import traceback
        row = {"sid": scn.sid, "run_ok": False, "error": traceback.format_exc()[-2000:]}
    row["run_wall_s"] = time.perf_counter() - t0
    return row


def search_family(fam):
    fid = fam[0]
    path = os.path.join(OUT, "search", f"{fid}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {}
    for r in C.read_jsonl(path):
        done[(r["purpose"], r["models"], round(r["sep_m"], 9))] = r
    fh = open(path, "a", encoding="utf-8")
    t0 = time.perf_counter()

    def prop(sep, purpose, models):
        key = (purpose, models, round(float(sep), 9))
        if key in done:
            return done[key]
        scn = family_scenario(fam, float(sep))
        row = _run(scn, models)
        row.update({"family_id": fid, "purpose": purpose, "models": models, "sep_m": float(sep)})
        row = C.jsonable(row)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        done[key] = row
        return row

    def lab(row, m):
        return bool(row.get(f"{m}_terminal_arrived")) if row.get("run_ok") else None

    seps = list(np.geomspace(SWEEP["lo_m"], SWEEP["hi_m"], SWEEP["n"]))
    pts = {}
    for s in seps:
        pts[s] = prop(s, "sweep", "pair")

    def changes(m):
        ss = sorted(pts)
        return [i for i in range(len(ss) - 1) if lab(pts[ss[i]], m) != lab(pts[ss[i + 1]], m)]

    # interior sampling around every label change and every disagreement of the sweep
    flag = set(changes("cw")) | set(changes("eci"))
    flag |= {i for i, s in enumerate(seps) if lab(pts[s], "cw") != lab(pts[s], "eci")}
    cells = set()
    for i in flag:
        for j in range(i - INTERIOR["cells_each_side"], i + INTERIOR["cells_each_side"] + 1):
            if 0 <= j < len(seps) - 1:
                cells.add(j)
    k = INTERIOR["points_per_cell"]
    for j in sorted(cells):
        a, b = math.log(seps[j]), math.log(seps[j + 1])
        for q in range(1, k + 1):
            s = math.exp(a + q * (b - a) / (k + 1))
            pts[s] = prop(s, "interior", "pair")
    # bisection of every change, per model, on the merged sample list
    brackets = []
    for m, models in (("cw", "cw"), ("eci", "ref")):
        ss = sorted(pts)
        labs = [lab(pts[s], m) for s in ss]
        for i in range(len(ss) - 1):
            if labs[i] == labs[i + 1] or labs[i] is None or labs[i + 1] is None:
                continue
            lo, hi, llo = ss[i], ss[i + 1], labs[i]
            nb = 0
            while hi / lo - 1.0 > BISECT_TOL:
                mid = math.exp(0.5 * (math.log(lo) + math.log(hi)))
                r = prop(mid, f"bisect_{models}", models)
                lm = lab(r, m)
                nb += 1
                if lm is None:
                    break
                if lm == llo:
                    lo = mid
                else:
                    hi = mid
            brackets.append({"family_id": fid, "model": "cw" if m == "cw" else "reference", "lo_m": lo, "hi_m": hi,
                             "label_below": llo, "label_above": not llo, "rel_width": hi / lo - 1.0, "n_bisections": nb,
                             "kind": "arrive->fail" if llo else "fail->arrive"})
    fh.close()
    rows = C.read_jsonl(path)
    return {"family_id": fid, "n_propagations": len(rows), "n_sweep": len(seps), "n_interior_cells": len(cells),
            "n_run_errors": sum(1 for r in rows if not r.get("run_ok")), "brackets": brackets,
            "wall_s": time.perf_counter() - t0}


def cmd_search(workers):
    if not os.path.exists(FAM_FILE):
        sys.exit("declare the families first")
    rec = json.load(open(os.path.join(OUT, "families_record.json")))
    if C.sha256(FAM_FILE) != rec["families_sha256"]:
        sys.exit("families.json changed after declaration; refusing")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(search_family, FAMILIES))
    import pandas as pd
    br = pd.DataFrame([b for r in res for b in r["brackets"]])
    br.to_csv(os.path.join(OUT, "brackets.csv"), index=False)
    summ = pd.DataFrame([{k: v for k, v in r.items() if k != "brackets"} for r in res])
    summ.to_csv(os.path.join(OUT, "search_summary.csv"), index=False)
    C.log_run("boundary_families", "python experiments/v2/boundary_families.py search", "completed",
              time.perf_counter() - t0, f"{int(summ.n_propagations.sum())} propagations, {int(summ.n_run_errors.sum())} run errors")
    print(summ.to_string())
    print(br.to_string())


STAGE2_TOL = 1e-7
STAGE2_IF_BELOW = 4.0      # refine further when |offset| < 4 x combined half-width
EXPLORE = {"F11": {"lo_m": 80000.0, "hi_m": 300000.0, "n": 24}}


def _bisect_more(args):
    """Stage-2 bisection of one stored bracket (added after the search: the instruction asks to refine
    until the uncertainty is clearly smaller than the offset where feasible). Every propagation is
    recorded with purpose bisect2_cw / bisect2_ref."""
    fid, model, lo, hi, llo = args
    fam = {f[0]: f for f in FAMILIES}[fid]
    path = os.path.join(OUT, "search", f"{fid}.jsonl")
    models = "cw" if model == "cw" else "ref"
    m = "cw" if model == "cw" else "eci"
    nb = 0
    while hi / lo - 1.0 > STAGE2_TOL:
        mid = math.exp(0.5 * (math.log(lo) + math.log(hi)))
        row = _run(family_scenario(fam, mid), models)
        row.update({"family_id": fid, "purpose": f"bisect2_{models}", "models": models, "sep_m": float(mid)})
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(C.jsonable(row)) + "\n")
        nb += 1
        if not row.get("run_ok"):
            break
        if bool(row.get(f"{m}_terminal_arrived")) == llo:
            lo = mid
        else:
            hi = mid
    return {"family_id": fid, "model": model, "lo_m": lo, "hi_m": hi, "label_below": llo, "label_above": not llo,
            "rel_width": hi / lo - 1.0, "n_bisections": nb, "kind": "arrive->fail" if llo else "fail->arrive", "stage": 2}


def cmd_refine2(workers):
    import pandas as pd
    br = pd.read_csv(os.path.join(OUT, "brackets.csv"))
    off = offsets(br)
    marg = off[(off.offset_rel.notna()) & (off.offset_rel.abs() < STAGE2_IF_BELOW * off.offset_unc_rel)]
    jobs = []
    for r in marg.itertuples():
        for model, lo, hi in (("cw", r.cw_lo_m, r.cw_hi_m), ("reference", r.ref_lo_m, r.ref_hi_m)):
            jobs.append((r.family_id, model, lo, hi, r.kind == "arrive->fail"))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_bisect_more, jobs))
    new = pd.DataFrame(res)
    br["stage"] = 1
    keep = br.merge(new[["family_id", "model", "kind"]].assign(_r=1), on=["family_id", "model", "kind"], how="left")
    out = pd.concat([keep[keep._r.isna()].drop(columns="_r"), new], ignore_index=True).sort_values(["family_id", "model", "lo_m"])
    out.to_csv(os.path.join(OUT, "brackets_final.csv"), index=False)
    C.log_run("boundary_families", "python experiments/v2/boundary_families.py refine2", "completed", time.perf_counter() - t0,
              f"stage-2 refinement of {len(jobs)} brackets in {sorted(set(marg.family_id))}")
    print(new.to_string())


def cmd_explore(workers):
    """EXPLORATORY (decided after inspecting the predeclared search): extend the separation range for
    families whose CW boundary lies beyond the predeclared 80 km limit. Recorded with purpose
    explore_sweep / explore_bisect_*; never mixed into the predeclared statistics."""
    import pandas as pd
    rows = []
    for fid, rg in EXPLORE.items():
        fam = {f[0]: f for f in FAMILIES}[fid]
        path = os.path.join(OUT, "search", f"{fid}.jsonl")
        seps = list(np.geomspace(rg["lo_m"], rg["hi_m"], rg["n"]))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_run, [family_scenario(fam, s) for s in seps], ["pair"] * len(seps)))
        with open(path, "a", encoding="utf-8") as fh:
            for s, r in zip(seps, res):
                r.update({"family_id": fid, "purpose": "explore_sweep", "models": "pair", "sep_m": float(s)})
                fh.write(json.dumps(C.jsonable(r)) + "\n")
        for m, models in (("cw", "cw"), ("eci", "ref")):
            labs = [bool(r.get(f"{m}_terminal_arrived")) if r.get("run_ok") else None for r in res]
            for i in range(len(seps) - 1):
                if labs[i] is None or labs[i + 1] is None or labs[i] == labs[i + 1]:
                    continue
                lo, hi, llo = seps[i], seps[i + 1], labs[i]
                while hi / lo - 1.0 > BISECT_TOL:
                    mid = math.exp(0.5 * (math.log(lo) + math.log(hi)))
                    r = _run(family_scenario(fam, mid), models)
                    r.update({"family_id": fid, "purpose": f"explore_bisect_{models}", "models": models, "sep_m": float(mid)})
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(C.jsonable(r)) + "\n")
                    if not r.get("run_ok"):
                        break
                    if bool(r.get(f"{m}_terminal_arrived")) == llo:
                        lo = mid
                    else:
                        hi = mid
                rows.append({"family_id": fid, "model": "cw" if m == "cw" else "reference", "lo_m": lo, "hi_m": hi,
                             "kind": "arrive->fail" if llo else "fail->arrive", "rel_width": hi / lo - 1.0, "exploratory": True})
        for s, r in zip(seps, res):
            print(fid, f"{s:9.0f}", r.get("cw_terminal_arrived"), r.get("eci_terminal_arrived"), r.get("eci_termination"))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "brackets_exploratory.csv"), index=False)
    C.log_run("boundary_families", "python experiments/v2/boundary_families.py explore", "completed", 0.0,
              "EXPLORATORY range extension " + ", ".join(EXPLORE))
    print(pd.DataFrame(rows).to_string())


def cmd_supplementary(workers):
    """EXPLORATORY supplementary diagnostic points (decided after inspecting the search) for disagreement
    regions the predeclared rule does not cover: bands bounded by unpaired boundaries (e.g. a CW-only
    arrival island, or a CW boundary beyond the predeclared range found by cmd_explore)."""
    import pandas as pd
    from main_grid import CONFIG, write_scenarios
    from cwvalid.runner import run_campaign
    fam_by = {f[0]: f for f in FAMILIES}
    br = pd.read_csv(os.path.join(OUT, "brackets_final.csv"))
    ex = pd.read_csv(os.path.join(OUT, "brackets_exploratory.csv")) if os.path.exists(os.path.join(OUT, "brackets_exploratory.csv")) else pd.DataFrame()
    regions = []
    # CW-only islands: a CW fail->arrive followed by arrive->fail with no reference boundary in between
    for fid, g in br[br.model == "cw"].groupby("family_id"):
        g = g.sort_values("lo_m")
        recs = g.to_dict("records")
        for a, b in zip(recs[:-1], recs[1:]):
            if a["kind"] == "fail->arrive" and b["kind"] == "arrive->fail":
                rf = br[(br.family_id == fid) & (br.model == "reference") & (br.lo_m > a["lo_m"]) & (br.hi_m < b["hi_m"])]
                if rf.empty:
                    regions.append((fid, a["hi_m"], b["lo_m"], "CW-only arrival island"))
    # reference boundary paired only with an exploratory CW boundary beyond the predeclared range
    for fid in EXPLORE:
        rf = br[(br.family_id == fid) & (br.model == "reference")]
        cw = ex[(ex.family_id == fid) & (ex.model == "cw")] if not ex.empty else ex
        if len(rf) and len(cw):
            regions.append((fid, float(rf.hi_m.iloc[0]), float(cw.lo_m.iloc[0]), "band beyond predeclared range"))
        elif len(rf):
            regions.append((fid, float(rf.hi_m.iloc[0]), SWEEP["hi_m"], "band to the end of the predeclared range"))
    scns = []
    for fid, lo, hi, kind in regions:
        pts = [(lo * 0.995, f"below {kind}")] + [(math.exp(math.log(lo) + f * (math.log(hi) - math.log(lo))), f"inside {kind} {f:g}")
                                                  for f in (0.25, 0.5, 0.75)] + [(hi * 1.005, f"above {kind}")]
        for s, lab in pts:
            sc = family_scenario(fam_by[fid], float(f"{s:.10g}"))
            sc.meta.update({"diag_label": lab, "boundary_kind": kind, "boundary_status": "exploratory supplementary"})
            scns.append(sc)
    d = os.path.join(OUT, "diagnostic_supplementary")
    os.makedirs(d, exist_ok=True)
    write_scenarios(os.path.join(d, "scenarios.jsonl"), scns)
    C.write_json(os.path.join(d, "set_definition.json"), {"set": "v2 boundary families, EXPLORATORY supplementary diagnostic points",
                 "regions": regions, "n": len(scns), "frozen_v1_sha256_at_run": C.sha256(C.FROZEN_V1), "config": CONFIG,
                 "note": "decided after inspecting the predeclared search; not part of the predeclared statistics",
                 "written": time.strftime("%Y-%m-%d %H:%M:%S")})
    t0 = time.perf_counter()
    run_campaign(d, scns, CONFIG, workers=workers, keep_hist=False)
    C.log_run("boundary_families", "python experiments/v2/boundary_families.py supplementary", "completed", time.perf_counter() - t0,
              f"{len(scns)} exploratory points in {len(regions)} regions")


def pair_boundaries(br, fid):
    """Pair each CW label change with the nearest reference change of the same kind."""
    b = br[br.family_id == fid]
    cw = b[b.model == "cw"].to_dict("records")
    rf = b[b.model == "reference"].to_dict("records")
    out, used = [], set()
    for c in cw:
        cand = [(abs(math.log(math.sqrt(r["lo_m"] * r["hi_m"]) / math.sqrt(c["lo_m"] * c["hi_m"]))), j)
                for j, r in enumerate(rf) if r["kind"] == c["kind"] and j not in used]
        if not cand:
            out.append({"cw": c, "ref": None})
            continue
        _, j = min(cand)
        used.add(j)
        out.append({"cw": c, "ref": rf[j]})
    for j, r in enumerate(rf):
        if j not in used:
            out.append({"cw": None, "ref": r})
    return out


def offsets(br):
    rows = []
    for fam in FAMILIES:
        fid = fam[0]
        pairs = pair_boundaries(br, fid)
        if not pairs:
            rows.append({"family_id": fid, "status": "no boundary in search range"})
            continue
        for p in pairs:
            c, r = p["cw"], p["ref"]
            if c is None or r is None:
                rows.append({"family_id": fid, "status": "unpaired " + ("reference" if c is None else "CW") + " boundary",
                             "kind": (r or c)["kind"],
                             "cw_lo_m": c["lo_m"] if c else np.nan, "cw_hi_m": c["hi_m"] if c else np.nan,
                             "ref_lo_m": r["lo_m"] if r else np.nan, "ref_hi_m": r["hi_m"] if r else np.nan})
                continue
            cm, rm = math.sqrt(c["lo_m"] * c["hi_m"]), math.sqrt(r["lo_m"] * r["hi_m"])
            off = cm / rm - 1.0
            unc = 0.5 * (c["hi_m"] / c["lo_m"] - 1.0) + 0.5 * (r["hi_m"] / r["lo_m"] - 1.0)
            resolved = abs(off) >= 2.0 * unc
            rows.append({"family_id": fid, "kind": c["kind"], "cw_lo_m": c["lo_m"], "cw_hi_m": c["hi_m"],
                         "ref_lo_m": r["lo_m"], "ref_hi_m": r["hi_m"], "offset_rel": off, "offset_unc_rel": unc,
                         "status": "resolved" if resolved else "unresolved (|offset| < 2 x uncertainty)"})
    import pandas as pd
    return pd.DataFrame(rows)


def cmd_diagnostic(workers):
    import pandas as pd
    from main_grid import CONFIG, write_scenarios
    from cwvalid.runner import run_campaign
    bf = os.path.join(OUT, "brackets_final.csv")
    br = pd.read_csv(bf if os.path.exists(bf) else os.path.join(OUT, "brackets.csv"))
    off = offsets(br)
    off.to_csv(os.path.join(OUT, "offsets.csv"), index=False)
    fam_by = {f[0]: f for f in FAMILIES}
    scns = []
    for r in off.itertuples():
        if not isinstance(r.status, str) or r.status == "no boundary in search range" or r.status.startswith("unpaired"):
            continue
        fam = fam_by[r.family_id]
        cm, rm = math.sqrt(r.cw_lo_m * r.cw_hi_m), math.sqrt(r.ref_lo_m * r.ref_hi_m)
        if r.status == "resolved":
            lo, hi = min(cm, rm), max(cm, rm)
            pts = [(lo * 0.995, "below band")]
            if hi / lo - 1.0 > 4 * BISECT_TOL:
                pts += [(math.exp(math.log(lo) + f * (math.log(hi) - math.log(lo))), f"inside band {f:g}") for f in (0.25, 0.5, 0.75)]
            pts.append((hi * 1.005, "above band"))
        else:
            mid = math.sqrt(cm * rm)
            pts = [(mid * 0.995, "below unresolved boundary"), (mid * 1.005, "above unresolved boundary")]
        for s, lab in pts:
            sc = family_scenario(fam, float(f"{s:.10g}"))
            sc.meta.update({"diag_label": lab, "boundary_kind": r.kind, "boundary_status": r.status})
            scns.append(sc)
    d = os.path.join(OUT, "diagnostic")
    os.makedirs(d, exist_ok=True)
    write_scenarios(os.path.join(d, "scenarios.jsonl"), scns)
    C.write_json(os.path.join(d, "set_definition.json"), {"set": "v2 boundary-family diagnostic points", "n": len(scns),
                 "rule": DIAG_RULE, "frozen_v1_sha256_at_run": C.sha256(C.FROZEN_V1), "config": CONFIG,
                 "scenarios_sha256": C.sha256(os.path.join(d, "scenarios.jsonl")), "written": time.strftime("%Y-%m-%d %H:%M:%S")})
    t0 = time.perf_counter()
    run_campaign(d, scns, CONFIG, workers=workers, keep_hist=False)
    C.log_run("boundary_families", "python experiments/v2/boundary_families.py diagnostic", "completed", time.perf_counter() - t0,
              f"{len(scns)} diagnostic points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["declare", "search", "refine2", "explore", "diagnostic", "supplementary"])
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    if a.cmd == "declare":
        cmd_declare()
    elif a.cmd == "search":
        cmd_search(a.workers)
    elif a.cmd == "refine2":
        cmd_refine2(a.workers)
    elif a.cmd == "explore":
        cmd_explore(a.workers)
    elif a.cmd == "supplementary":
        cmd_supplementary(a.workers)
    else:
        cmd_diagnostic(a.workers)


if __name__ == "__main__":
    main()
