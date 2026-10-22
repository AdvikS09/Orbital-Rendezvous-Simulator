import hashlib
import math

import numpy as np

from cwvalid.constants import MUEARTH, REARTH
from cwvalid.scenario import AXES, Scenario

SEEDLABEL = "CEAS v3 extension | saturation-aware screening | prospective campaign | 2026-10-13"
ROOTENT = int(hashlib.sha256(SEEDLABEL.encode()).hexdigest()[:32], 16)
ARCHSEEDS = (7001, 9001, 9101, 424242, 20261010, 20261011, 12001, 31001)

DIST = {"sep": [50.0, 50_000.0], "nu": [0.0, 3.0], "wn": [3.0, 10.0], "amax": [0.005, 0.2],
        "p_axis": 0.3, "p_zero": 0.2, "p_em1": 0.1, "horizon_orbits": 1.0}

CELLS = {
    "D0": {"index": 0, "altitude": 400_000.0, "cmd_mode": "continuous", "Ts": None, "controller": "pd"},
    "D1": {"index": 1, "altitude": 600_000.0, "cmd_mode": "continuous", "Ts": None, "controller": "pd"},
    "D2": {"index": 2, "altitude": 800_000.0, "cmd_mode": "continuous", "Ts": None, "controller": "pd"},
    "D3": {"index": 3, "altitude": 400_000.0, "cmd_mode": "sampled", "Ts": 1.0, "controller": "pd"},
    "D4": {"index": 4, "altitude": 400_000.0, "cmd_mode": "sampled", "Ts": 10.0, "controller": "pd"},
    "D5": {"index": 5, "altitude": 400_000.0, "cmd_mode": "continuous", "Ts": None, "controller": "lqr"},
}
SPLITS = {"pilot": 0, "dev": 1, "cal": 2, "test": 3, "audit": 4, "transfer_cohort": 5}


def meanmotionalt(altitude):
    return math.sqrt(MUEARTH / (REARTH + altitude) ** 3)


def stream(cell, split):
    ss = np.random.SeedSequence(entropy=ROOTENT, spawn_key=(CELLS[cell]["index"], SPLITS[split]))
    return np.random.Generator(np.random.PCG64(ss)), ss


def checkunused():
    out = {"root_entropy_in_archive_seeds": ROOTENT in ARCHSEEDS, "first_draw_collisions": []}
    firsts = {s: np.random.default_rng(s).random(4).tolist() for s in ARCHSEEDS}
    for c in CELLS:
        for sp in SPLITS:
            g, _ = stream(c, sp)
            v = g.random(4).tolist()
            for s, f in firsts.items():
                if np.allclose(v, f, rtol=0, atol=0):
                    out["first_draw_collisions"].append((c, sp, s))
    return out


def unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def drawcase(rng, n, dist=DIST):
    sep = float(math.exp(rng.uniform(math.log(dist["sep"][0]), math.log(dist["sep"][1]))))
    if rng.uniform() < dist["p_axis"]:
        ax = list(AXES)[int(rng.integers(6))]
        u = np.array(AXES[ax], dtype=float)
        dlabel = ax
    else:
        u = unit(rng)
        dlabel = "mixed"
    pos = u * sep
    r = rng.uniform()
    if r < dist["p_zero"]:
        vel = np.zeros(3)
        vlabel, nu = "zero", 0.0
    elif r < dist["p_zero"] + dist["p_em1"]:
        vel = np.array([0.0, -2.0 * n * pos[0], 0.0])
        vlabel = "em1"
        nu = float(np.linalg.norm(vel) / (n * sep))
    else:
        nu = float(rng.uniform(*dist["nu"]))
        vel = unit(rng) * nu * n * sep
        vlabel = "random"
    wn = float(rng.uniform(*dist["wn"]))
    am = float(math.exp(rng.uniform(math.log(dist["amax"][0]), math.log(dist["amax"][1]))))
    return {"rel0": [float(v) for v in np.concatenate((pos, vel))], "sep_m": sep, "axis": dlabel, "velocity": vlabel,
            "nu": nu, "wn_ratio": wn, "a_max": am, "dir": [float(v) for v in u]}


def makelaw(cell, n, wn_ratio, a_max):
    from .controllers import lqrlaw, pdlaw
    wn = wn_ratio * n
    if CELLS[cell]["controller"] == "pd":
        return pdlaw(wn, a_max)
    return lqrlaw(n, wn, a_max)


def makecases(cell, split, N, start=0):
    from .ledger import physical_id, stablehash
    c = CELLS[cell]
    n = meanmotionalt(c["altitude"])
    rng, ss = stream(cell, split)
    out = []
    for i in range(start + N):
        d = drawcase(rng, n)
        if i < start:
            continue
        law = makelaw(cell, n, d["wn_ratio"], d["a_max"])
        K = law.K
        meta = {"axis": d["axis"], "sep_m": d["sep_m"], "velocity": d["velocity"], "nu": d["nu"], "dir": d["dir"],
                "cell": cell, "split": split, "index": i, "variant": f"v3:{cell}"}
        scn = Scenario(f"v3{cell}{split}", tuple(d["rel0"]), d["wn_ratio"], d["a_max"], DIST["horizon_orbits"],
                       c["altitude"], meta=meta)
        pid = physical_id(d["rel0"], c["altitude"], DIST["horizon_orbits"], K, d["a_max"])
        cid = "C" + stablehash({"cell": cell, "split": split, "index": i, "draw": d})
        out.append({"case_id": cid, "physical_id": pid, "cell": cell, "split": split, "index": i,
                    "scenario": scn.to_dict(), "K": K.tolist(), "controller": law.design,
                    "cmd_mode": c["cmd_mode"], "Ts": c["Ts"], "n_radps": n, "draw": d})
    return out, {"entropy": str(ROOTENT), "spawn_key": [c["index"], SPLITS[split]],
                 "bit_generator": "PCG64", "seed_sequence": "numpy.random.SeedSequence"}
