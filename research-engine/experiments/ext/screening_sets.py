"""Step 7 data - development, broad test and boundary-stress scenario sets.

Sampling (all seeds fixed and recorded in results/screening/<set>/set_definition.json):

  dev    (seed 7001, N=200): separation log-uniform [50 m, 50 km]; direction 70%
         uniform on the sphere, 30% a random signed LVLH axis; initial velocity
         20% zero, 70% random direction with nu = |v0|/(n rho0) ~ U[0, 2],
         10% first-order bounded (ydot0 = -2 n x0); omega_n/n ~ U[3.5, 8.5];
         a_max log-uniform [0.01, 0.1] m/s^2.
  test   (seed 9001, N=400): same structure but wider: nu ~ U[0, 3],
         omega_n/n ~ U[3, 10], a_max log-uniform [0.005, 0.2] m/s^2. Cases
         outside the development ranges are flagged ``in_dev_domain = False``.
  stress (seed 9101, N=150): drawn from the test distribution and kept only if
         the FROZEN screening score (computed from a CW-only run) lies within
         a band around the frozen decision threshold. Generated after freezing.

These distributions are study design choices for exercising the method; they
are not mission probability distributions.

The test and stress sets refuse to run unless results/screening/frozen/
screening_frozen.json exists; its SHA-256 is recorded with the test results.

Usage (repository root):
  python experiments/ext/screening_sets.py {dev|test|stress} [--workers 7]
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(os.path.dirname(ROOT), "results", "screening")
FROZEN = os.path.join(RES, "frozen", "screening_frozen.json")

from main_grid import CONFIG, write_scenarios  # noqa: E402

from cwvalid.constants import DEFAULT_RADIUS, mean_motion  # noqa: E402
from cwvalid.runner import run_campaign  # noqa: E402
from cwvalid.scenario import AXES, Scenario  # noqa: E402

SETS = {
    "dev": {"seed": 7001, "N": 200, "sep": [50.0, 50_000.0], "nu": [0.0, 2.0], "wn": [3.5, 8.5],
            "amax": [0.01, 0.1], "p_axis": 0.3, "p_zero": 0.2, "p_em1": 0.1},
    "test": {"seed": 9001, "N": 400, "sep": [50.0, 50_000.0], "nu": [0.0, 3.0], "wn": [3.0, 10.0],
             "amax": [0.005, 0.2], "p_axis": 0.3, "p_zero": 0.2, "p_em1": 0.1},
}
STRESS = {"seed": 9101, "N": 150, "base": "test", "max_candidates": 20000}
DEV_DOMAIN = {"wn": [3.5, 8.5], "amax": [0.01, 0.1], "nu_max": 2.0}


def _unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def sample_one(rng, spec, family, index):
    n = mean_motion(DEFAULT_RADIUS)
    sep = float(math.exp(rng.uniform(math.log(spec["sep"][0]), math.log(spec["sep"][1]))))
    if rng.uniform() < spec["p_axis"]:
        ax = list(AXES)[int(rng.integers(6))]
        u = np.array(AXES[ax])
        dlabel = ax
    else:
        u = _unit(rng)
        dlabel = "mixed"
    pos = u * sep
    r = rng.uniform()
    if r < spec["p_zero"]:
        vel = np.zeros(3)
        vlabel, nu = "zero", 0.0
    elif r < spec["p_zero"] + spec["p_em1"]:
        vel = np.array([0.0, -2.0 * n * pos[0], 0.0])
        vlabel = "em1"
        nu = float(np.linalg.norm(vel) / (n * sep))
    else:
        nu = float(rng.uniform(*spec["nu"]))
        vel = _unit(rng) * nu * n * sep
        vlabel = "random"
    wn = float(rng.uniform(*spec["wn"]))
    am = float(math.exp(rng.uniform(math.log(spec["amax"][0]), math.log(spec["amax"][1]))))
    in_dom = (DEV_DOMAIN["wn"][0] <= wn <= DEV_DOMAIN["wn"][1] and DEV_DOMAIN["amax"][0] <= am <= DEV_DOMAIN["amax"][1]
              and nu <= DEV_DOMAIN["nu_max"] + 1e-12)
    meta = {"axis": dlabel, "sep_m": sep, "velocity": vlabel, "nu": nu, "set_index": index,
            "in_dev_domain": bool(in_dom), "dir": [float(v) for v in u]}
    return Scenario(family, tuple(np.concatenate((pos, vel))), wn, am, 1.0, meta=meta)


def make_set(name):
    spec = SETS[name]
    rng = np.random.default_rng(spec["seed"])
    return [sample_one(rng, spec, name, i) for i in range(spec["N"])], spec


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=["dev", "test", "stress"])
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    d = os.path.join(RES, a.set)
    os.makedirs(d, exist_ok=True)
    if a.set in ("test", "stress"):
        if not os.path.exists(FROZEN):
            sys.exit("refusing to run: screening method not frozen (missing %s)" % FROZEN)
        frozen_hash = sha256(FROZEN)
    else:
        frozen_hash = None
    if a.set == "stress" and not os.path.exists(os.path.join(d, "set_definition.json")):
        from screening import stress_candidates  # noqa: E402  (uses the frozen rule)
        scns, spec = stress_candidates(STRESS, SETS[STRESS["base"]], sample_one, FROZEN)
    elif a.set == "stress":
        scns, spec = [], STRESS
    else:
        scns, spec = make_set(a.set)
    sd_path = os.path.join(d, "set_definition.json")
    sc_path = os.path.join(d, "scenarios.jsonl")
    if os.path.exists(sd_path):
        # resuming: the original record (scenario list, frozen hash, first-run time) is kept
        sd = json.load(open(sd_path))
        if a.set in ("test", "stress") and sd.get("frozen_sha256_at_run") != frozen_hash:
            sys.exit(f"refusing to resume {a.set}: it was started under a different frozen method "
                     f"({sd.get('frozen_sha256_at_run')}); create a fresh set instead")
        scns = [Scenario.from_dict(json.loads(l)) for l in open(sc_path) if l.strip()]
        sd.setdefault("resumed", []).append(time.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        write_scenarios(sc_path, scns)
        sd = {"set": a.set, "spec": spec, "dev_domain": DEV_DOMAIN, "n": len(scns),
              "frozen_sha256_at_run": frozen_hash, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
              "config": CONFIG}
    json.dump(sd, open(sd_path, "w"), indent=2)
    run_campaign(d, scns, CONFIG, workers=a.workers, keep_hist=False)


if __name__ == "__main__":
    main()
