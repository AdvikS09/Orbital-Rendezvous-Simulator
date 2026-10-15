"""Convergence reporting over a ladder of solver settings (SPEC section 7).

Reads the JSON lines written by ``python -m indverify`` and reports, for every
scenario and solver family, the change of every compared quantity between
consecutive settings of the ladder (coarse -> fine: Radau/DOP853 by decreasing
rtol, RK4 by decreasing h), and optionally the difference of every setting
from a chosen reference setting.

    python -m indverify.convergence OUTDIR/indverify_results.jsonl \
        --out OUTDIR/convergence.csv [--reference radau_1e-12]

Reading the ladders (accuracy limits found in the verification review):

* Reference-branch quantities (``ref_*`` and ``*_vs_ref_*``; column
  ``ref_derived``).  With the SPEC section 7 tolerances the inertial
  reference integrates |r| ~ 6.8e6 m with relative tolerance rtol, and under
  feedback the target and chaser errors do not cancel in rho; its relative-
  state error is tens to thousands of times that of the CW / quadratic
  branches at the same rtol (50-5600x measured on two saturated cases), so
  e.g. so_vs_ref_peak_pos_m is solver error, not model error, at Radau 1e-8
  to 1e-10.  Treat ref-derived quantities as converged only if
  the finest rungs agree (Radau 1e-11 -> 1e-12, DOP853 1e-12 -> 1e-13), take
  reference values from DOP853 1e-13 or Radau 1e-12, or tighten only that
  branch with the setting key ``ref_rtol``.
* Radau.  SciPy's Radau dense output (the cubic collocation polynomial) is
  1-2 orders of magnitude less accurate between steps than at step ends, and
  every assessed quantity is sampled from dense output, so Radau ladders
  converge more slowly than its step-end error suggests.  A smaller
  ``max_step`` or the DOP853 rungs give the tighter reference values.
* RK4 has no switch events (SPEC section 7.3): a step may straddle a
  saturation switch, so in saturated cases the RK4 ladder is not a clean
  fourth-order sequence (the h -> h/2 error ratio depends on where switches
  fall inside steps).  Judge switch, entry and saturated-time ladders of RK4
  accordingly; an extra rung such as h = 0.125 s helps.
* Failure times (numerical_failure, incl. the divergence guard of the CW /
  quadratic branches) are step-start times: they are defined only to within
  one step and must not be compared across solvers more finely than h or
  max_step.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict

# Bookkeeping / run-configuration keys that are not compared quantities.
_META = {"rtol", "ref_rtol", "h", "max_step", "wall_s", "wall_integrate_s", "horizon_s",
         "T_orb_s", "n_radps", "wn_ratio", "a_max", "fine_dt"}
_META_SUFFIX = ("_n_steps", "_nfev", "_n_restart_fallback")
LIST_SUFFIX = "_switch_times_s"


def is_ref_derived(quantity):
    """True for quantities computed from the inertial reference branch."""
    return quantity.startswith("ref_") or "_vs_ref_" in quantity


def load_results(path, stats=None):
    """Successful records of a results file, one per (sid, setting_label).

    If a pair has several records (e.g. an error line from a crashed run
    followed by a successful re-run), the last successful one is used and the
    error lines are dropped; pairs with only error records are left out.
    Unparseable lines are skipped.  If ``stats`` is a dict it receives the
    counts of lines, unparseable lines, error-only pairs and dropped records.
    """
    last_ok, errored, order = {}, set(), []
    n_lines = n_bad = n_dropped = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError
            except ValueError:
                n_bad += 1
                continue
            key = (str(rec.get("sid")), str(rec.get("setting_label")))
            if "error" in rec:
                errored.add(key)
                n_dropped += 1
                continue
            if key in last_ok:
                n_dropped += 1
            else:
                order.append(key)
            last_ok[key] = rec
    if stats is not None:
        stats.update(lines=n_lines, unparseable=n_bad, dropped=n_dropped,
                     error_only_pairs=len(errored - set(last_ok)))
    return [last_ok[k] for k in order]


def compared_quantities(rec):
    keys = []
    for k, v in rec.items():
        if k in _META or k.endswith(_META_SUFFIX):
            continue
        if isinstance(v, (bool, int, float)) or k.endswith(LIST_SUFFIX):
            keys.append(k)
    return sorted(keys)


def _fineness(rec):
    """Sort key: larger value = coarser setting."""
    if rec.get("solver") == "rk4":
        return float(rec.get("h", math.nan))
    return float(rec.get("rtol", math.nan))


def quantity_change(a, b):
    """(change b - a, note).  Booleans as 0/1; lists of switch times by max |diff|."""
    if isinstance(a, list) or isinstance(b, list):
        a = a or []
        b = b or []
        if len(a) != len(b):
            return math.nan, "count %d->%d" % (len(a), len(b))
        if not a:
            return 0.0, ""
        return max(abs(float(x) - float(y)) for x, y in zip(a, b)), "max|dt|"
    if isinstance(a, bool) or isinstance(b, bool):
        return float(int(bool(b)) - int(bool(a))), ""
    if isinstance(a, str) or isinstance(b, str):
        return (0.0, "") if a == b else (math.nan, "%s->%s" % (a, b))
    fa, fb = float(a), float(b)
    if math.isnan(fa) and math.isnan(fb):
        return 0.0, "both NaN"
    if math.isnan(fa) or math.isnan(fb):
        return math.nan, "NaN on one side"
    return fb - fa, ""


def _rows_between(r0, r1, kind):
    rows = []
    keys = sorted(set(compared_quantities(r0)) | set(compared_quantities(r1)))
    for k in keys + [b + "_status" for b in ("ref", "cw", "so") if b + "_status" in r0]:
        v0, v1 = r0.get(k), r1.get(k)
        ch, note = quantity_change(v0, v1)
        rows.append({"kind": kind, "sid": r0.get("sid"), "solver_from": r0.get("solver"),
                     "solver_to": r1.get("solver"), "from_label": r0.get("setting_label"),
                     "to_label": r1.get("setting_label"), "quantity": k,
                     "ref_derived": is_ref_derived(k),
                     "from_value": json.dumps(v0), "to_value": json.dumps(v1),
                     "change": ch, "abs_change": abs(ch) if not math.isnan(ch) else math.nan,
                     "note": note})
    return rows


def ladder_changes(records):
    """Changes between consecutive settings of each (sid, solver) ladder."""
    groups = defaultdict(list)
    for r in records:
        groups[(r.get("sid"), r.get("solver"))].append(r)
    rows = []
    for key in sorted(groups, key=lambda x: (str(x[0]), str(x[1]))):
        ladder = sorted(groups[key], key=_fineness, reverse=True)
        for r0, r1 in zip(ladder[:-1], ladder[1:]):
            rows.extend(_rows_between(r0, r1, "ladder"))
    return rows


def changes_vs_reference(records, reference_label):
    """Difference of every setting from the reference setting of the same sid."""
    ref = {r.get("sid"): r for r in records if r.get("setting_label") == reference_label}
    rows = []
    for r in records:
        r0 = ref.get(r.get("sid"))
        if r0 is None or r is r0:
            continue
        rows.extend(_rows_between(r0, r, "vs_reference"))
    return rows


def summarise(rows):
    """Max |change| over scenarios per (kind, from_label, to_label, quantity)."""
    agg = {}
    for row in rows:
        key = (row["kind"], row["from_label"], row["to_label"], row["quantity"])
        cur = agg.setdefault(key, {"ref_derived": is_ref_derived(row["quantity"]),
                                   "max_abs_change": 0.0, "n": 0, "n_undefined": 0})
        cur["n"] += 1
        if math.isnan(row["abs_change"]):
            cur["n_undefined"] += 1
        else:
            cur["max_abs_change"] = max(cur["max_abs_change"], row["abs_change"])
    return [dict(kind=k[0], from_label=k[1], to_label=k[2], quantity=k[3], **v)
            for k, v in sorted(agg.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]


def _write_csv(path, rows):
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m indverify.convergence")
    p.add_argument("results")
    p.add_argument("--out", required=True, help="CSV of per-scenario changes")
    p.add_argument("--reference", default=None, help="label of the reference setting")
    args = p.parse_args(argv)
    lstats = {}
    recs = load_results(args.results, lstats)
    if lstats["unparseable"] or lstats["dropped"] or lstats["error_only_pairs"]:
        print("load: %(lines)d line(s); %(unparseable)d unparseable, %(dropped)d superseded or "
              "error record(s) dropped, %(error_only_pairs)d pair(s) with only errors" % lstats)
    rows = ladder_changes(recs)
    if args.reference:
        rows += changes_vs_reference(recs, args.reference)
    _write_csv(args.out, rows)
    summary_path = args.out[:-4] + "_summary.csv" if args.out.endswith(".csv") else args.out + "_summary.csv"
    _write_csv(summary_path, summarise(rows))
    print("%d records, %d change rows -> %s (summary: %s)" % (len(recs), len(rows), args.out, summary_path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
