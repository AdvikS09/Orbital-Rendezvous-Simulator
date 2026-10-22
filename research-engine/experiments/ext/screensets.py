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
FROZEN = os.path.join(RES, "frozen", "frozen.json")

from maingrid import CONFIG, writescen

from cwvalid.constants import DEFRAD, mean_motion
from cwvalid.runner import runcamp
from cwvalid.scenario import AXES, Scenario

SETS = {
    "dev": {"seed": 7001, "N": 200, "sep": [50.0, 50_000.0], "nu": [0.0, 2.0], "wn": [3.5, 8.5],
            "amax": [0.01, 0.1], "p_axis": 0.3, "p_zero": 0.2, "p_em1": 0.1},
    "test": {"seed": 9001, "N": 400, "sep": [50.0, 50_000.0], "nu": [0.0, 3.0], "wn": [3.0, 10.0],
             "amax": [0.005, 0.2], "p_axis": 0.3, "p_zero": 0.2, "p_em1": 0.1},
}
STRESS = {"seed": 9101, "N": 150, "base": "test", "max_candidates": 20000}
DEVDOMAIN = {"wn": [3.5, 8.5], "amax": [0.01, 0.1], "nu_max": 2.0}


def unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def sampleone(rng, spec, family, index):
    n = mean_motion(DEFRAD)
    sep = float(math.exp(rng.uniform(math.log(spec["sep"][0]), math.log(spec["sep"][1]))))
    if rng.uniform() < spec["p_axis"]:
        ax = list(AXES)[int(rng.integers(6))]
        u = np.array(AXES[ax])
        dlabel = ax
    else:
        u = unit(rng)
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
        vel = unit(rng) * nu * n * sep
        vlabel = "random"
    wn = float(rng.uniform(*spec["wn"]))
    am = float(math.exp(rng.uniform(math.log(spec["amax"][0]), math.log(spec["amax"][1]))))
    indom = (DEVDOMAIN["wn"][0] <= wn <= DEVDOMAIN["wn"][1] and DEVDOMAIN["amax"][0] <= am <= DEVDOMAIN["amax"][1]
             and nu <= DEVDOMAIN["nu_max"] + 1e-12)
    meta = {"axis": dlabel, "sep_m": sep, "velocity": vlabel, "nu": nu, "set_index": index,
            "in_dev_domain": bool(indom), "dir": [float(v) for v in u]}
    return Scenario(family, tuple(np.concatenate((pos, vel))), wn, am, 1.0, meta=meta)


def makeset(name):
    spec = SETS[name]
    rng = np.random.default_rng(spec["seed"])
    return [sampleone(rng, spec, name, i) for i in range(spec["N"])], spec


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
        frozenhash = sha256(FROZEN)
    else:
        frozenhash = None
    if a.set == "stress" and not os.path.exists(os.path.join(d, "setdefinition.json")):
        from screening import stresscands
        scns, spec = stresscands(STRESS, SETS[STRESS["base"]], sampleone, FROZEN)
    elif a.set == "stress":
        scns, spec = [], STRESS
    else:
        scns, spec = makeset(a.set)
    sdpath = os.path.join(d, "setdefinition.json")
    scpath = os.path.join(d, "scenarios.jsonl")
    if os.path.exists(sdpath):
        sd = json.load(open(sdpath))
        if a.set in ("test", "stress") and sd.get("frozen_sha256_at_run") != frozenhash:
            sys.exit(f"refusing to resume {a.set}: it was started under a different frozen method "
                     f"({sd.get('frozen_sha256_at_run')}); create a fresh set instead")
        scns = [Scenario.from_dict(json.loads(l)) for l in open(scpath) if l.strip()]
        sd.setdefault("resumed", []).append(time.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        writescen(scpath, scns)
        sd = {"set": a.set, "spec": spec, "dev_domain": DEVDOMAIN, "n": len(scns),
              "frozen_sha256_at_run": frozenhash, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
              "config": CONFIG}
    json.dump(sd, open(sdpath, "w"), indent=2)
    runcamp(d, scns, CONFIG, workers=a.workers, keephist=False)


if __name__ == "__main__":
    main()
