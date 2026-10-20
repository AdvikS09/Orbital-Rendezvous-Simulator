import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
PROJ = os.path.dirname(ROOT)
G = os.path.join(PROJ, "prior_work", "GMAT_Validation")
OUT = os.path.join(PROJ, "results", "verification")

from cwvalid.constants import DEFRAD, mean_motion
from cwvalid.control import CmdSpec
from cwvalid.dynamics import ecilvlhvec
from cwvalid.simulate import SolverCfg, simulate

CASES = ["along1k_twobody", "radial1k_twobody", "cross1k_twobody", "mixed3d_twobody", "radial5k_twobody"]


def main():
    R = DEFRAD
    n = mean_motion(R)
    rows = []
    for tag in CASES:
        g = pd.read_csv(os.path.join(G, "output", f"gmat_{tag}.txt"), sep=r"\s+", engine="python")
        sim = pd.read_csv(os.path.join(G, "data", f"sim_{tag}.csv"))
        cols = lambda s: [f"{s}.EarthMJ2000Eq.{c}" for c in ("X", "Y", "Z", "VX", "VY", "VZ")]
        tgt = g[cols("Chief")].to_numpy(float) * 1e3
        dep = g[cols("Deputy")].to_numpy(float) * 1e3
        m = min(len(g), len(sim)) - 1
        t = sim["t_s"].to_numpy(float)[:m]
        relg, _ = ecilvlhvec(tgt[:m], dep[:m])
        rel0 = sim[["rel_x_m", "rel_y_m", "rel_z_m", "rel_vx_mps", "rel_vy_mps", "rel_vz_mps"]].to_numpy(float)[0]
        rel0 = np.round(rel0, 9)
        s = simulate("nerm", rel0, CmdSpec("none"), t[-1], n=n, R=R,
                     solver=SolverCfg(rtol=1e-13, atol_rel_pos=1e-10, atol_rel_vel=1e-13, max_step=5.0))
        reln = s.states(t)
        dp = np.linalg.norm(relg[:, :3] - reln[:, :3], axis=1)
        dv = np.linalg.norm(relg[:, 3:] - reln[:, 3:], axis=1)
        dsim = np.linalg.norm(sim[["rel_x_m", "rel_y_m", "rel_z_m"]].to_numpy(float)[:m] - reln[:, :3], axis=1)
        rows.append({"case": tag, "rel0": list(rel0), "samples_compared": m,
                     "gmat_vs_nerm_max_pos_m": dp.max(), "gmat_vs_nerm_rms_pos_m": float(np.sqrt((dp**2).mean())),
                     "gmat_vs_nerm_max_vel_mps": dv.max(),
                     "orig_sim_vs_nerm_max_pos_m": dsim.max(),
                     "max_range_m": float(np.linalg.norm(reln[:, :3], axis=1).max())})
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "gmatnerm.csv"), index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
