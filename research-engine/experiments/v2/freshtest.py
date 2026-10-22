import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ext"))
import v2common as C

OUT = os.path.join(C.RAW, "freshtest")
SEED = 12001
N = 300


def physkeys():
    keys = {}
    for p in glob.glob(os.path.join(C.RESULTS, "**", "scenarios.jsonl"), recursive=True):
        if os.path.abspath(p).startswith(os.path.abspath(OUT)):
            continue
        for line in open(p):
            if line.strip():
                s = json.loads(line)
                k = (tuple(round(float(v), 9) for v in s["rel0"]), round(float(s["wn_ratio"]), 9),
                     None if s["a_max"] is None else round(float(s["a_max"]), 9),
                     float(s.get("horizon_orbits", 1.0)), float(s.get("altitude", 400000.0)))
                keys.setdefault(k, []).append(os.path.relpath(p, C.RESULTS))
    return keys


def cmdman():
    from maingrid import CONFIG, writescen
    from screensets import SETS, sampleone
    if not os.path.exists(C.FROZENV2):
        sys.exit("refusing: freeze the v2 estimator method first (experiments/v2/estfit.py)")
    if os.path.exists(os.path.join(OUT, "setdefinition.json")):
        sys.exit("fresh test manifest already exists; refusing to replace the test set")
    spec = dict(SETS["test"], seed=SEED, N=N)
    rng = np.random.default_rng(SEED)
    scns = [sampleone(rng, spec, "fresh", i) for i in range(N)]
    keys = physkeys()
    overlap = []
    for s in scns:
        k = (tuple(round(float(v), 9) for v in s.rel0), round(float(s.wn_ratio), 9), round(float(s.a_max), 9),
             float(s.horizon_orbits), float(s.altitude))
        if k in keys:
            overlap.append({"sid": s.sid, "found_in": keys[k]})
    os.makedirs(OUT, exist_ok=True)
    scpath = os.path.join(OUT, "scenarios.jsonl")
    writescen(scpath, scns)
    sd = {"set": "fresh_test (v2 confirmatory)", "seed": SEED, "n": N, "spec": spec,
          "distribution_source": "experiments/ext/screensets.py SETS['test'] (seed and N replaced)",
          "scenarios_sha256": C.sha256(scpath), "estimator_v2_frozen_sha256": C.sha256(C.FROZENV2),
          "v1_frozen_sha256": C.sha256(C.FROZENV1), "overlap_with_archive": overlap,
          "n_archive_scenarios_checked": sum(len(v) for v in keys.values()),
          "seeds_in_archive": [7001, 9001, 9101, 424242, 20261010, 20261011], "config": CONFIG,
          "written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": C.coderev(),
          "note": "manifest written and hashed before any reference label was computed"}
    C.writejson(os.path.join(OUT, "setdefinition.json"), sd)
    C.logrun("fresh_test", "python experiments/v2/freshtest.py manifest", "completed", 0.0,
             f"sha256 {sd['scenarios_sha256']}; overlap {len(overlap)}")
    print(json.dumps({k: sd[k] for k in ("seed", "n", "scenarios_sha256", "estimator_v2_frozen_sha256", "n_archive_scenarios_checked")}, indent=2))
    print("overlap with archive:", len(overlap))


def cmdrun(workers):
    from maingrid import CONFIG
    from cwvalid.runner import runcamp
    from cwvalid.scenario import Scenario
    sd = json.load(open(os.path.join(OUT, "setdefinition.json")))
    scpath = os.path.join(OUT, "scenarios.jsonl")
    if C.sha256(scpath) != sd["scenarios_sha256"]:
        sys.exit("manifest changed after hashing; refusing")
    if C.sha256(C.FROZENV2) != sd["estimator_v2_frozen_sha256"]:
        sys.exit("v2 freeze file changed after the manifest was written; refusing")
    if sd["overlap_with_archive"]:
        sys.exit("manifest overlaps archived scenarios; refusing")
    scns = [Scenario.from_dict(json.loads(l)) for l in open(scpath) if l.strip()]
    t0 = time.perf_counter()
    runcamp(OUT, scns, CONFIG, workers=workers, keephist=False)
    C.logrun("fresh_test", "python experiments/v2/freshtest.py run", "completed", time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "run"])
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    cmdman() if a.cmd == "manifest" else cmdrun(a.workers)


if __name__ == "__main__":
    main()
