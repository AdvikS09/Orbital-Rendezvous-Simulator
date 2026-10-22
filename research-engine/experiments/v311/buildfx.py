import hashlib
import json
import os
import shutil
import time

import v311paths as P
from cwvalid3 import calibration as CAL

LINES = [
    ("raw/manifests/D0test.jsonl", "first 25 lines", lambda k, r, ctx: k <= 25),
    ("raw/manifests/D3cal.jsonl", "first 25 lines", lambda k, r, ctx: k <= 25),
    ("raw/manifests/D5dev.jsonl", "first 25 lines", lambda k, r, ctx: k <= 25),
    ("raw/manifests/D4test.jsonl", "first 2 lines (planned indices 0 and 1)", lambda k, r, ctx: k <= 2),
    ("raw/manifests/D4dev.jsonl", "planned index 56", lambda k, r, ctx: r["index"] == 56),
    ("raw/results/D0test.jsonl", "first 3 lines; records of the first 3 planned cases; first record with eci_terminal_arrived "
     "true; first completed non-arrival (eci_status ok, eci_terminal_arrived false)", None),
    ("raw/results/D0cal.jsonl", "first 40 lines; first record with eci_status terminated", None),
    ("raw/results/D1cal.jsonl", "first 40 lines; every worker_crash record; first 10 records with eci_status terminated; first 10 "
     "records labelled violation at 1 m by the frozen v3.0 labelling (cwvalid3.calibration)", None),
    ("raw/results/D3cal.jsonl", "first 50 lines; first record with eci_status domain_exit", None),
    ("raw/results/D4cal.jsonl", "first 40 lines; first 10 records with eci_status domain_exit", None),
    ("raw/results/D5cal.jsonl", "first 40 lines", lambda k, r, ctx: k <= 40),
    ("raw/results/D5test.jsonl", "first record with kp null", None),
    ("raw/results/D4test.jsonl", "records of planned indices 0 and 1", lambda k, r, ctx: r.get("index") in (0, 1)),
    ("raw/results/D4dev.jsonl", "record of planned index 56", lambda k, r, ctx: r.get("index") == 56),
    ("v31/raw/results/heldexit31.jsonl", "C01 replay record of D4 dev index 56",
     lambda k, r, ctx: r.get("replay", {}).get("of_run_id") in ctx["d4dev56"]),
    ("ledger.jsonl", "planned events of every fixture manifest line; completed, completed_reconciled and rerun events "
     "of every fixture result record", lambda k, r, ctx: r.get("run_id") in ctx["ledger_ids"] and
     (r.get("event") in ("completed", "completed_reconciled", "rerun") or (r.get("event") == "planned" and r.get("run_id") in ctx["planned_ids"]))),
]
WHOLE = ["raw/freeze.json", "raw/manifests/D0testsummary.json", "raw/manifests/D3calsummary.json",
         "raw/manifests/D5devsummary.json", "raw/manifests/D4testsummary.json", "raw/manifests/D4devsummary.json"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def filesha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def firstrules(rel):
    preds = {
        "raw/results/D0test.jsonl": [("first", 3), ("pred", lambda r: r.get("eci_terminal_arrived") is True),
                                      ("pred", lambda r: r.get("eci_terminal_arrived") is False and r.get("eci_status") == "ok")],
        "raw/results/D0cal.jsonl": [("first", 40), ("pred", lambda r: r.get("eci_status") == "terminated")],
        "raw/results/D1cal.jsonl": [("first", 40), ("all", lambda r: r.get("worker_status") == "worker_crash"),
                                     ("firstk", (10, lambda r: r.get("eci_status") == "terminated")),
                                     ("firstk", (10, lambda r: r.get("worker_status") == "ok" and
                                                 CAL.critlabel(CAL.truth_record(r), 1.0) == "violation"))],
        "raw/results/D4cal.jsonl": [("first", 40), ("firstk", (10, lambda r: r.get("eci_status") == "domain_exit"))],
        "raw/results/D3cal.jsonl": [("first", 50), ("pred", lambda r: r.get("eci_status") == "domain_exit")],
        "raw/results/D5test.jsonl": [("pred", lambda r: r.get("kp") is None)],
    }
    return preds.get(rel)


def select(rel, rule, ctx):
    s = os.path.join(P.V3, rel.replace("/", os.sep))
    raw = open(s, "rb").read()
    parts = raw.split(b"\n")
    if parts and parts[-1] == b"":
        parts = parts[:-1]
    keep = []
    rules = firstrules(rel)
    done = set()
    taken = {}
    for k, line in enumerate(parts, 1):
        if rel == "ledger.jsonl" and not any(x in line for x in ctx["ledger_needles"]):
            continue
        r = json.loads(line)
        ok = False
        if rules is None:
            ok = rule(k, r, ctx)
        else:
            for j, (kind, arg) in enumerate(rules):
                if kind == "first" and k <= arg:
                    ok = True
                elif kind == "pred" and j not in done and arg(r):
                    ok = True
                    done.add(j)
                elif kind == "all" and arg(r):
                    ok = True
                elif kind == "firstk" and taken.get(j, 0) < arg[0] and arg[1](r):
                    ok = True
                    taken[j] = taken.get(j, 0) + 1
                if kind == "pred" and arg(r):
                    done.add(j)
        if ok:
            keep.append((k, line, r))
    if rel == "raw/results/D0test.jsonl":
        have = {k for k, _, _ in keep}
        for k, line in enumerate(parts, 1):
            if k not in have and json.loads(line)["run_id"] in ctx["d0test_first3"]:
                keep.append((k, line, json.loads(line)))
        keep.sort(key=lambda x: x[0])
    return s, raw, len(parts), keep


def ids(r):
    out = {k: r.get(k) for k in ("index", "case_id", "physical_id", "run_id", "worker_status", "ledger_id", "event") if k in r}
    if "replay" in r:
        out["replay_of_run_id"] = r["replay"].get("of_run_id")
    return out


def main():
    root = P.FIXTURES
    if os.path.isdir(root):
        shutil.rmtree(P.guard(root))
    os.makedirs(P.guard(os.path.join(root, "lines")), exist_ok=True)
    m56 = next(json.loads(x) for x in open(os.path.join(P.MAN, "D4dev.jsonl"), encoding="utf-8") if json.loads(x)["index"] == 56)
    d0 = [json.loads(x) for _, x in zip(range(3), open(os.path.join(P.MAN, "D0test.jsonl"), encoding="utf-8"))]
    ctx = {"d4dev56": {m56["run_id"]}, "d0test_first3": {m["run_id"] for m in d0}, "ledger_ids": set(), "planned_ids": set(),
           "ledger_needles": []}
    index = {"created_local": time.strftime("%Y-%m-%d %H:%M:%S"), "builder": "code/experiments/v311/buildfx.py",
             "rule": "exact source lines in source order (LF-split), never regenerated; whole files are byte copies",
             "source_root": "v3 (archive root of CW_Rendezvous_CEAS_Extension_v2)", "files": [], "whole_files": []}
    for rel, desc, rule in LINES:
        if rel == "ledger.jsonl":
            ctx["ledger_needles"] = [rid.encode("ascii") for rid in sorted(ctx["ledger_ids"])]
        s, raw, nlines, keep = select(rel, rule, ctx)
        out = os.path.join(root, "lines", rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(P.guard(out), "wb") as fh:
            for _, line, _ in keep:
                fh.write(line + b"\n")
        entry = {"fixture": "lines/" + rel, "source": "v3/" + rel, "source_sha256": sha(raw), "source_bytes": len(raw),
                 "source_lines": nlines, "selection": desc, "n_lines": len(keep), "fixture_sha256": filesha(out),
                 "lines": [{"line": k, "sha256": sha(line), **ids(r)} for k, line, r in keep]}
        if rel.startswith("raw/manifests/"):
            sp = os.path.join(P.MAN, os.path.basename(rel).replace(".jsonl", "_summary.json"))
            summ = json.load(open(sp, encoding="utf-8"))
            entry["parent_manifest_sha256_in_summary"] = summ.get("manifest_sha256")
            entry["manifest_sha256_matches_summary"] = summ.get("manifest_sha256") == entry["source_sha256"]
            entry["run_config_in"] = "whole/" + "raw/manifests/" + os.path.basename(sp)
            ctx["planned_ids"] |= {r["run_id"] for _, _, r in keep}
            ctx["ledger_ids"] |= {r["run_id"] for _, _, r in keep}
        if rel.startswith("raw/results/"):
            mp = os.path.join(P.MAN, os.path.basename(rel))
            entry["manifest_of_these_records"] = {"source": "v3/raw/manifests/" + os.path.basename(rel), "sha256": filesha(mp)}
            ctx["ledger_ids"] |= {r["run_id"] for _, _, r in keep}
        index["files"].append(entry)
        print(f"{rel}: {len(keep)} of {nlines} lines")
    for rel in WHOLE:
        s = os.path.join(P.V3, rel.replace("/", os.sep))
        out = os.path.join(root, "whole", rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(s, P.guard(out))
        index["whole_files"].append({"fixture": "whole/" + rel, "source": "v3/" + rel, "sha256": filesha(s),
                                     "bytes": os.path.getsize(s)})
    zp = os.path.join(os.path.dirname(P.ARCH), "CW_Rendezvous_CEAS_Extension_v3.zip")
    index["baseline_archive"] = {"name": os.path.basename(zp), "bytes": os.path.getsize(zp) if os.path.exists(zp) else None,
                                 "sha256": filesha(zp) if os.path.exists(zp) else None,
                                 "v3_extension_inventory": {"path": "v3/OUTPUT_INVENTORY.csv",
                                                            "sha256": filesha(os.path.join(P.V3, "OUTPUT_INVENTORY.csv"))},
                                 "v31_inventory": {"path": "v3/v31/OUTPUT_INVENTORY_v31.csv",
                                                   "sha256": filesha(os.path.join(P.V31, "OUTPUT_INVENTORY_v31.csv"))}}
    with open(P.guard(os.path.join(root, "index.json")), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(index, fh, indent=1)
    tot = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(root) for f in fs)
    print(f"fixtures: {len(index['files'])} line files, {len(index['whole_files'])} whole files, {tot / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
