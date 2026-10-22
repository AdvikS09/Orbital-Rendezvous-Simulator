import argparse
import json
import os
import sys
import time

ARCHIVE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
V3 = os.path.join(ARCHIVE, "v3")
MAN = os.path.join(V3, "raw", "manifests")
RES = os.path.join(V3, "raw", "results")
REPLAY = os.path.join(V3, "v31", "raw", "results", "heldexit31.jsonl")


def read(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def sample():
    groups = []
    for cell in ("D0", "D1", "D2", "D5"):
        man = read(os.path.join(MAN, f"{cell}test.jsonl"))
        groups += [(f"{cell}_test_first6", [m]) for m in man[:6]]
    for cell in ("D3", "D4"):
        man = read(os.path.join(MAN, f"{cell}test.jsonl"))
        res = {r["run_id"]: r for r in read(os.path.join(RES, f"{cell}test.jsonl"))}
        pick = [m for m in man if res[m["run_id"]].get("eci_status") == "completed" and res[m["run_id"]].get("nerm_status") == "completed"][:8]
        groups.append((f"{cell}_test_first8_nonexit", pick))
    rep = [r for r in read(REPLAY) if r["replay"]["chunk"] == 3]
    want = [r["replay"]["of_run_id"] for r in rep]
    cell = rep[0]["cell"]
    mans = {}
    for sp in ("pilot", "dev", "cal", "test", "transfer_cohort"):
        p = os.path.join(MAN, f"{cell}{sp.replace('_', '')}.jsonl")
        if os.path.exists(p):
            for m in read(p):
                if m["run_id"] in want:
                    mans[m["run_id"]] = m
    groups.append(("replay_chunk3", [mans[r] for r in want]))
    d4t = read(os.path.join(MAN, "D4test.jsonl"))[:2]
    d4d = [m for m in read(os.path.join(MAN, "D4dev.jsonl")) if m["index"] == 56]
    groups.append(("followup_D4_test_0_1_dev_56", d4t + d4d))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--dump", required=True)
    a = ap.parse_args()
    root = os.path.abspath(a.code_root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "experiments", "v3"))
    os.environ.pop("CWVALID3_FEATURE_CHECKPOINT_DIR", None)
    import campaign as CP
    from cwvalid3 import serialization as SER
    assert os.path.dirname(os.path.abspath(CP.__file__)) == os.path.join(root, "experiments", "v3"), CP.__file__
    t0 = time.perf_counter()
    c0 = time.process_time()
    n = 0
    with open(a.dump, "w", encoding="utf-8", newline="\n") as fh:
        for group, recs in sample():
            held = CP.CELLS[recs[0]["cell"]]["cmd_mode"] == "sampled"
            outs = CP.workheld(recs) if held else CP.workcont(recs[0])
            for o in outs:
                o["c11_group"] = group
                SER.write_record(fh, o, schema=None)
                n += 1
    meta = {"code_root": root, "engine_sha": CP.enginever(), "n_records": n, "wall_s": time.perf_counter() - t0,
            "cpu_s": time.process_time() - c0}
    json.dump(meta, open(a.dump + ".meta.json", "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
