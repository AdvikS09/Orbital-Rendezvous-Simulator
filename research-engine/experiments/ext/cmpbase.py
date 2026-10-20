import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reportutil as RU

PROJ = os.path.dirname(RU.RESULTS)
PUB = os.path.join(PROJ, "prior_work", "Supplementary", "data")
BR = os.path.join(RU.RESULTS, "reproduction")
ENG = os.path.join(BR, "engine", "outputs", "data")

PAIRS = {
    "convergence.csv": os.path.join(ENG, "convergence.csv"),
    "rendezvousmetrics.csv": os.path.join(ENG, "rendezvousmetrics.csv"),
    "rendezvousseries.csv": os.path.join(ENG, "rendezvousseries.csv"),
    "hifiseries.csv": os.path.join(ENG, "hifiseries.csv"),
    "hifitoggles.csv": os.path.join(ENG, "hifitoggles.csv"),
    "orbitvalidation.csv": os.path.join(ENG, "orbitvalidation.csv"),
    "gainsweep.csv": os.path.join(ENG, "gainsweep.csv"),
    "sepsweep.csv": os.path.join(ENG, "sepsweep.csv"),
    "radialenergy.csv": os.path.join(ENG, "radialenergy.csv"),
    "energysma.csv": os.path.join(BR, "energysma.csv"),
    "robustness.csv": os.path.join(BR, "robustness.csv"),
    "robustnesscrossings.csv": os.path.join(BR, "robustnesscrossings.csv"),
    "terminal.csv": os.path.join(BR, "terminal.csv"),
    "THRESHOLD_SENSITIVITY.csv": None,
    "validation_summary.csv": None,
}


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def cmpfile(pub, rep):
    if rep is None or not os.path.exists(rep):
        return {"status": "unavailable" if rep is None else "missing (not regenerated yet)"}
    if md5(pub) == md5(rep):
        return {"status": "identical (byte-for-byte)", "max_rel_diff": 0.0}
    a, b = pd.read_csv(pub), pd.read_csv(rep)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return {"status": f"different shape/columns {a.shape} vs {b.shape}"}
    num = a.select_dtypes("number").columns
    mrd = 0.0
    for c in num:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        bothnan = np.isnan(x) & np.isnan(y)
        if (np.isnan(x) != np.isnan(y)).any():
            return {"status": f"NaN pattern differs in {c}"}
        d = np.abs(x - y)[~bothnan] / np.maximum(np.abs(x[~bothnan]), 1e-300)
        mrd = max(mrd, float(d.max()) if d.size else 0.0)
    other = [c for c in a.columns if c not in num]
    if any((a[c].astype(str) != b[c].astype(str)).any() for c in other):
        return {"status": "non-numeric fields differ", "max_rel_diff": mrd}
    return {"status": "numerically equal" if mrd < 1e-9 else "numerically different", "max_rel_diff": mrd}


def main():
    rows = []
    for name, rep in PAIRS.items():
        pub = os.path.join(PUB, name)
        r = {"file": name, "published": os.path.relpath(pub, PROJ),
             "regenerated": os.path.relpath(rep, PROJ) if rep else ""}
        r.update(cmpfile(pub, rep))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BR, "filecomparison.csv"), index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
