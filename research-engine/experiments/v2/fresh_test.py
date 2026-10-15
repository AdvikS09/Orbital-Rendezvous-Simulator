"""Fresh 300-case confirmatory test set for the v2 estimator comparison.

Distribution: exactly the documented broad-test distribution (experiments/ext/screening_sets.py,
SETS["test"]: separation log-uniform 50 m - 50 km; direction 70 % uniform on the sphere, 30 % a
random signed LVLH axis; velocity 20 % zero, 10 % first-order bounded, 70 % random direction
with nu ~ U[0, 3]; omega_n/n ~ U[3, 10]; a_max log-uniform [0.005, 0.2] m/s^2) with N = 300 and
the unused seed 12001 (seeds in the archive: 7001, 9001, 9101, 424242, 20261010, 20261011).
No case is selected or rejected using nonlinear outcomes.

Protocol:
  manifest  refuses unless the v2 method is frozen; draws the 300 scenarios; checks them for
            overlap with every archived scenario (identical physics key: initial state, gains,
            cap, horizon, altitude); writes scenarios.jsonl and set_definition.json with the
            SHA-256 of the manifest, of the v2 freeze file and of the v1 freeze file. No
            propagation is run.
  run       refuses unless the manifest and freeze hashes are unchanged; then runs the paired
            campaign (CW, inertial reference, NERM cross-check, second order) with the
            production configuration. This is the first time reference labels are computed.

Usage: python experiments/v2/fresh_test.py manifest|run [--workers 7]
Output: results/v2/raw/fresh_test/
"""

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
import v2_common as C  # noqa: E402

OUT = os.path.join(C.RAW, "fresh_test")
SEED = 12001
N = 300


def physics_keys():
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


def cmd_manifest():
    from main_grid import CONFIG, write_scenarios
    from screening_sets import SETS, sample_one
    if not os.path.exists(C.FROZEN_V2):
        sys.exit("refusing: freeze the v2 estimator method first (experiments/v2/estimator_fit.py)")
    if os.path.exists(os.path.join(OUT, "set_definition.json")):
        sys.exit("fresh test manifest already exists; refusing to replace the test set")
    spec = dict(SETS["test"], seed=SEED, N=N)
    rng = np.random.default_rng(SEED)
    scns = [sample_one(rng, spec, "fresh", i) for i in range(N)]
    keys = physics_keys()
    overlap = []
    for s in scns:
        k = (tuple(round(float(v), 9) for v in s.rel0), round(float(s.wn_ratio), 9), round(float(s.a_max), 9),
             float(s.horizon_orbits), float(s.altitude))
        if k in keys:
            overlap.append({"sid": s.sid, "found_in": keys[k]})
    os.makedirs(OUT, exist_ok=True)
    sc_path = os.path.join(OUT, "scenarios.jsonl")
    write_scenarios(sc_path, scns)
    sd = {"set": "fresh_test (v2 confirmatory)", "seed": SEED, "n": N, "spec": spec,
          "distribution_source": "experiments/ext/screening_sets.py SETS['test'] (seed and N replaced)",
          "scenarios_sha256": C.sha256(sc_path), "estimator_v2_frozen_sha256": C.sha256(C.FROZEN_V2),
          "v1_frozen_sha256": C.sha256(C.FROZEN_V1), "overlap_with_archive": overlap,
          "n_archive_scenarios_checked": sum(len(v) for v in keys.values()),
          "seeds_in_archive": [7001, 9001, 9101, 424242, 20261010, 20261011], "config": CONFIG,
          "written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": C.code_revision(),
          "note": "manifest written and hashed before any reference label was computed"}
    C.write_json(os.path.join(OUT, "set_definition.json"), sd)
    C.log_run("fresh_test", "python experiments/v2/fresh_test.py manifest", "completed", 0.0,
              f"sha256 {sd['scenarios_sha256']}; overlap {len(overlap)}")
    print(json.dumps({k: sd[k] for k in ("seed", "n", "scenarios_sha256", "estimator_v2_frozen_sha256", "n_archive_scenarios_checked")}, indent=2))
    print("overlap with archive:", len(overlap))


def cmd_run(workers):
    from main_grid import CONFIG
    from cwvalid.runner import run_campaign
    from cwvalid.scenario import Scenario
    sd = json.load(open(os.path.join(OUT, "set_definition.json")))
    sc_path = os.path.join(OUT, "scenarios.jsonl")
    if C.sha256(sc_path) != sd["scenarios_sha256"]:
        sys.exit("manifest changed after hashing; refusing")
    if C.sha256(C.FROZEN_V2) != sd["estimator_v2_frozen_sha256"]:
        sys.exit("v2 freeze file changed after the manifest was written; refusing")
    if sd["overlap_with_archive"]:
        sys.exit("manifest overlaps archived scenarios; refusing")
    scns = [Scenario.from_dict(json.loads(l)) for l in open(sc_path) if l.strip()]
    t0 = time.perf_counter()
    run_campaign(OUT, scns, CONFIG, workers=workers, keep_hist=False)
    C.log_run("fresh_test", "python experiments/v2/fresh_test.py run", "completed", time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "run"])
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    cmd_manifest() if a.cmd == "manifest" else cmd_run(a.workers)


if __name__ == "__main__":
    main()
