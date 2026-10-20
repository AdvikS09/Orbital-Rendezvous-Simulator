import argparse
import csv
import json
import math
import sys
from collections import defaultdict

META = {"rtol", "ref_rtol", "h", "max_step", "wall_s", "wall_integrate_s", "horizon_s",
        "T_orb_s", "n_radps", "wn_ratio", "a_max", "fine_dt"}
METASFX = ("_n_steps", "_nfev", "_n_restart_fallback")
LISTSFX = "_switch_times_s"


def refderived(quantity):
    return quantity.startswith("ref_") or "_vs_ref_" in quantity


def loadres(path, stats=None):
    lastok, errored, order = {}, set(), []
    nlines = nbad = ndropped = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            nlines += 1
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError
            except ValueError:
                nbad += 1
                continue
            key = (str(rec.get("sid")), str(rec.get("setting_label")))
            if "error" in rec:
                errored.add(key)
                ndropped += 1
                continue
            if key in lastok:
                ndropped += 1
            else:
                order.append(key)
            lastok[key] = rec
    if stats is not None:
        stats.update(lines=nlines, unparseable=nbad, dropped=ndropped,
                     error_only_pairs=len(errored - set(lastok)))
    return [lastok[k] for k in order]


def cmpquant(rec):
    keys = []
    for k, v in rec.items():
        if k in META or k.endswith(METASFX):
            continue
        if isinstance(v, (bool, int, float)) or k.endswith(LISTSFX):
            keys.append(k)
    return sorted(keys)


def fin(rec):
    if rec.get("solver") == "rk4":
        return float(rec.get("h", math.nan))
    return float(rec.get("rtol", math.nan))


def quantchange(a, b):
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


def rowsbetw(r0, r1, kind):
    rows = []
    keys = sorted(set(cmpquant(r0)) | set(cmpquant(r1)))
    for k in keys + [b + "_status" for b in ("ref", "cw", "so") if b + "_status" in r0]:
        v0, v1 = r0.get(k), r1.get(k)
        ch, note = quantchange(v0, v1)
        rows.append({"kind": kind, "sid": r0.get("sid"), "solver_from": r0.get("solver"),
                     "solver_to": r1.get("solver"), "from_label": r0.get("setting_label"),
                     "to_label": r1.get("setting_label"), "quantity": k,
                     "ref_derived": refderived(k),
                     "from_value": json.dumps(v0), "to_value": json.dumps(v1),
                     "change": ch, "abs_change": abs(ch) if not math.isnan(ch) else math.nan,
                     "note": note})
    return rows


def ladderchang(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r.get("sid"), r.get("solver"))].append(r)
    rows = []
    for key in sorted(groups, key=lambda x: (str(x[0]), str(x[1]))):
        ladder = sorted(groups[key], key=fin, reverse=True)
        for r0, r1 in zip(ladder[:-1], ladder[1:]):
            rows.extend(rowsbetw(r0, r1, "ladder"))
    return rows


def changref(records, reflabel):
    ref = {r.get("sid"): r for r in records if r.get("setting_label") == reflabel}
    rows = []
    for r in records:
        r0 = ref.get(r.get("sid"))
        if r0 is None or r is r0:
            continue
        rows.extend(rowsbetw(r0, r, "vs_reference"))
    return rows


def summar(rows):
    agg = {}
    for row in rows:
        key = (row["kind"], row["from_label"], row["to_label"], row["quantity"])
        cur = agg.setdefault(key, {"ref_derived": refderived(row["quantity"]),
                                   "max_abs_change": 0.0, "n": 0, "n_undefined": 0})
        cur["n"] += 1
        if math.isnan(row["abs_change"]):
            cur["n_undefined"] += 1
        else:
            cur["max_abs_change"] = max(cur["max_abs_change"], row["abs_change"])
    return [dict(kind=k[0], from_label=k[1], to_label=k[2], quantity=k[3], **v)
            for k, v in sorted(agg.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]


def writecsv(path, rows):
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
    p.add_argument("--out", required=True)
    p.add_argument("--reference", default=None)
    args = p.parse_args(argv)
    lstats = {}
    recs = loadres(args.results, lstats)
    if lstats["unparseable"] or lstats["dropped"] or lstats["error_only_pairs"]:
        print("load: %(lines)d line(s); %(unparseable)d unparseable, %(dropped)d superseded or "
              "error record(s) dropped, %(error_only_pairs)d pair(s) with only errors" % lstats)
    rows = ladderchang(recs)
    if args.reference:
        rows += changref(recs, args.reference)
    writecsv(args.out, rows)
    summpath = args.out[:-4] + "summary.csv" if args.out.endswith(".csv") else args.out + "summary.csv"
    writecsv(summpath, summar(rows))
    print("%d records, %d change rows -> %s (summary: %s)" % (len(recs), len(rows), args.out, summpath))
    return 0


if __name__ == "__main__":
    sys.exit(main())
