"""Domain cells, the deployment distribution, RNG streams and case construction (P04/P05).

Deployment distribution (an analyst-chosen simulation distribution, not a measured mission
population). It is the archive's broad-test template, experiments/ext/screening_sets.py
SETS["test"], with one deliberate correction: velocities are generated with the ACTUAL
scenario mean motion n = sqrt(mu/R^3) of the cell, not mean_motion(DEFAULT_RADIUS).
Draw order per case (identical to screening_sets.sample_one so the template is recognisable):
  1. separation rho0 = exp(U(ln 50, ln 50 000)) m
  2. direction: with probability 0.3 one of the six signed LVLH axes (+R,-R,+V,-V,+H,-H), each
     with probability 1/6 given this branch (rng.integers(6) over that list order); otherwise a
     direction uniform on the unit sphere (normalised standard normal 3-vector)
  3. velocity, by u ~ U(0,1): u < 0.2 zero; 0.2 <= u < 0.3 the first-order bounded relation
     rho_dot0 = (0, -2 n x0, 0) ("em1"); otherwise a uniform direction with speed nu n rho0,
     nu ~ U(0, 3). The normalised velocity is nu = |rho_dot0| / (n rho0) in every branch.
  4. omega_n/n ~ U(3, 10)
  5. a_max = exp(U(ln 0.005, ln 0.2)) m/s^2
Horizon one orbit of the cell's chief; no case is selected or rejected using outcomes.

Cells (all circular two-body chief; R = R_E + h):
  D0  h = 400 km  continuous  scalar PD (kp = wn^2, kd = 2 wn)          source domain
  D1  h = 600 km  continuous  scalar PD
  D2  h = 800 km  continuous  scalar PD
  D3  h = 400 km  LVLH command held 1 s (updates t_k = k Ts from t = 0)  scalar PD
  D4  h = 400 km  LVLH command held 10 s                                  scalar PD
  D5  h = 400 km  continuous  CW LQR matrix gain (cwvalid3.controllers)  same wn, a_max draws

RNG streams: numpy PCG64 through SeedSequence(entropy=ROOT_ENTROPY, spawn_key=(cell, split)).
ROOT_ENTROPY is derived from a fixed text label (below) so the scheme is recorded rather than
an arbitrary integer; verify_unused() checks it against every integer seed recorded in the
archive, and campaign planning checks generated physics keys against all archived scenarios.
"""

import hashlib
import math

import numpy as np

from cwvalid.constants import MU_EARTH, R_EARTH
from cwvalid.scenario import AXES, Scenario

SEED_LABEL = "CEAS v3 extension | saturation-aware screening | prospective campaign | 2026-10-13"
ROOT_ENTROPY = int(hashlib.sha256(SEED_LABEL.encode()).hexdigest()[:32], 16)
ARCHIVE_SEEDS = (7001, 9001, 9101, 424242, 20261010, 20261011, 12001, 31001)

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


def mean_motion_alt(altitude):
    return math.sqrt(MU_EARTH / (R_EARTH + altitude) ** 3)


def stream(cell, split):
    ss = np.random.SeedSequence(entropy=ROOT_ENTROPY, spawn_key=(CELLS[cell]["index"], SPLITS[split]))
    return np.random.Generator(np.random.PCG64(ss)), ss


def verify_unused():
    """The root entropy is not one of the archive's integer seeds, and no (cell, split) stream
    reproduces the first draws of default_rng(seed) for any archived seed."""
    out = {"root_entropy_in_archive_seeds": ROOT_ENTROPY in ARCHIVE_SEEDS, "first_draw_collisions": []}
    firsts = {s: np.random.default_rng(s).random(4).tolist() for s in ARCHIVE_SEEDS}
    for c in CELLS:
        for sp in SPLITS:
            g, _ = stream(c, sp)
            v = g.random(4).tolist()
            for s, f in firsts.items():
                if np.allclose(v, f, rtol=0, atol=0):
                    out["first_draw_collisions"].append((c, sp, s))
    return out


def _unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def draw_case(rng, n, dist=DIST):
    """One draw from the deployment distribution using mean motion n (see module doc)."""
    sep = float(math.exp(rng.uniform(math.log(dist["sep"][0]), math.log(dist["sep"][1]))))
    if rng.uniform() < dist["p_axis"]:
        ax = list(AXES)[int(rng.integers(6))]
        u = np.array(AXES[ax], dtype=float)
        dlabel = ax
    else:
        u = _unit(rng)
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
        vel = _unit(rng) * nu * n * sep
        vlabel = "random"
    wn = float(rng.uniform(*dist["wn"]))
    am = float(math.exp(rng.uniform(math.log(dist["amax"][0]), math.log(dist["amax"][1]))))
    return {"rel0": [float(v) for v in np.concatenate((pos, vel))], "sep_m": sep, "axis": dlabel, "velocity": vlabel,
            "nu": nu, "wn_ratio": wn, "a_max": am, "dir": [float(v) for v in u]}


def make_law(cell, n, wn_ratio, a_max):
    """Actual controller of the cell (MatrixLaw for every cell; PD cells hold a PD matrix)."""
    from .controllers import lqr_law, pd_law
    wn = wn_ratio * n
    if CELLS[cell]["controller"] == "pd":
        return pd_law(wn, a_max)
    return lqr_law(n, wn, a_max)


def make_cases(cell, split, N, start=0):
    """Deterministic list of case records for (cell, split), indices start..start+N-1 (the stream
    is consumed from index 0 so any prefix is reproducible)."""
    from .ledger import physical_id, stable_hash
    c = CELLS[cell]
    n = mean_motion_alt(c["altitude"])
    rng, ss = stream(cell, split)
    out = []
    for i in range(start + N):
        d = draw_case(rng, n)
        if i < start:
            continue
        law = make_law(cell, n, d["wn_ratio"], d["a_max"])
        K = law.K
        meta = {"axis": d["axis"], "sep_m": d["sep_m"], "velocity": d["velocity"], "nu": d["nu"], "dir": d["dir"],
                "cell": cell, "split": split, "index": i, "variant": f"v3:{cell}"}
        scn = Scenario(f"v3{cell}{split}", tuple(d["rel0"]), d["wn_ratio"], d["a_max"], DIST["horizon_orbits"],
                       c["altitude"], meta=meta)
        pid = physical_id(d["rel0"], c["altitude"], DIST["horizon_orbits"], K, d["a_max"])
        cid = "C" + stable_hash({"cell": cell, "split": split, "index": i, "draw": d})
        out.append({"case_id": cid, "physical_id": pid, "cell": cell, "split": split, "index": i,
                    "scenario": scn.to_dict(), "K": K.tolist(), "controller": law.design,
                    "cmd_mode": c["cmd_mode"], "Ts": c["Ts"], "n_radps": n, "draw": d})
    return out, {"entropy": str(ROOT_ENTROPY), "spawn_key": [c["index"], SPLITS[split]],
                 "bit_generator": "PCG64", "seed_sequence": "numpy.random.SeedSequence"}


__all__ = ["DIST", "CELLS", "SPLITS", "ROOT_ENTROPY", "SEED_LABEL", "ARCHIVE_SEEDS", "stream", "verify_unused",
           "draw_case", "make_law", "make_cases", "mean_motion_alt"]
