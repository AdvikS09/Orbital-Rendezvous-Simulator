"""v3.1.1: normal-path records of the worker code in a given code tree, for the old/new comparison.

Runs the fixed sample of experiments/v3_1/c02_worker_equivalence.py (first 6 planned indices of every continuous test
split; first 8 non-exit planned indices of every held test split, one batch per cell; every case of C01 replay chunk 3
in its stored membership and order) plus the two follow-up cases D4 test indices 0-1 and D4 dev index 56 (one batch),
with the campaign workers imported from --code-root (e.g. a worktree of tag v3.1, or the v3.1.1 tree), and writes the
records (canonical form) to --dump. Inputs are read from the archive read-only. No checkpoint directory is set.

Usage: python c11_normal_path.py --code-root <repo code dir> --dump <records.jsonl>
"""

import argparse
import json
import os
import sys
import time

ARCHIVE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
V3 = os.path.join(ARCHIVE, "v3_extension")
MAN = os.path.join(V3, "raw", "manifests")
RES = os.path.join(V3, "raw", "results")
REPLAY = os.path.join(V3, "v3_1_correction", "raw", "results", "held_exit_replay_v31.jsonl")


def read(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def sample():
    """[(group, [manifest records])] in run order; held groups are run as one batch each."""
    groups = []
    for cell in ("D0", "D1", "D2", "D5"):
        man = read(os.path.join(MAN, f"{cell}_test.jsonl"))
        groups += [(f"{cell}_test_first6", [m]) for m in man[:6]]
    for cell in ("D3", "D4"):
        man = read(os.path.join(MAN, f"{cell}_test.jsonl"))
        res = {r["run_id"]: r for r in read(os.path.join(RES, f"{cell}_test.jsonl"))}
        pick = [m for m in man if res[m["run_id"]].get("eci_status") == "completed" and res[m["run_id"]].get("nerm_status") == "completed"][:8]
        groups.append((f"{cell}_test_first8_nonexit", pick))
    rep = [r for r in read(REPLAY) if r["replay"]["chunk"] == 3]
    want = [r["replay"]["of_run_id"] for r in rep]
    cell = rep[0]["cell"]
    mans = {}
    for sp in ("pilot", "dev", "cal", "test", "transfer_cohort"):
        p = os.path.join(MAN, f"{cell}_{sp}.jsonl")
        if os.path.exists(p):
            for m in read(p):
                if m["run_id"] in want:
                    mans[m["run_id"]] = m
    groups.append(("replay_chunk3", [mans[r] for r in want]))
    d4t = read(os.path.join(MAN, "D4_test.jsonl"))[:2]
    d4d = [m for m in read(os.path.join(MAN, "D4_dev.jsonl")) if m["index"] == 56]
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
            outs = CP.work_held(recs) if held else CP.work_continuous(recs[0])
            for o in outs:
                o["c11_group"] = group
                SER.write_record(fh, o, schema=None)
                n += 1
    meta = {"code_root": root, "engine_sha": CP.engine_version(), "n_records": n, "wall_s": time.perf_counter() - t0,
            "cpu_s": time.process_time() - c0}
    json.dump(meta, open(a.dump + ".meta.json", "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
