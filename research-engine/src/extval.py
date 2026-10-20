import json

import numpy as np
import pandas as pd

STDCOL = ["t_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]

LENUNITS = {"m": 1.0, "km": 1000.0}
TIMEUNITS = {"s": 1.0, "sec": 1.0, "min": 60.0, "h": 3600.0, "hr": 3600.0}


def tocsvtext(ephemeris, metadata):
    lines = [
        "# Orbital Rendezvous Engine -- external-validation scenario export",
        "# Units: position [m], velocity [m/s], time [s]; frame: ECI",
    ]
    for key, value in metadata.items():
        lines.append(f"# {key}: {value}")
    return "\n".join(lines) + "\n" + ephemeris.to_csv(index=False)


def jsontext(ephemeris, metadata):
    doc = {
        "metadata": metadata,
        "ephemeris": {col: [float(v) for v in ephemeris[col]] for col in ephemeris.columns},
    }
    return json.dumps(doc, indent=2)


def guesscolumn(columns, candidates):
    lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for c in columns:
        cl = c.lower()
        if any(cand in cl for cand in candidates):
            return c
    return None


def autcol(columns):
    t = guesscolumn(columns, ["t_s", "time", "elapsed", "epoch", "t"])
    x = guesscolumn(columns, ["x_m", "x (km)", "x_km", "x", "rx", "pos_x"])
    y = guesscolumn(columns, ["y_m", "y (km)", "y_km", "y", "ry", "pos_y"])
    z = guesscolumn(columns, ["z_m", "z (km)", "z_km", "z", "rz", "pos_z"])
    vx = guesscolumn(columns, ["vx_mps", "vx (km/s)", "vx", "vel_x"])
    vy = guesscolumn(columns, ["vy_mps", "vy (km/s)", "vy", "vel_y"])
    vz = guesscolumn(columns, ["vz_mps", "vz (km/s)", "vz", "vel_z"])
    return t, [x, y, z], [vx, vy, vz]


def loadext(source, timecol=None, poscols=None, velcols=None,
            lenunit="m", timeunit="s", t0=0.0, comment="#"):
    raw = pd.read_csv(source, comment=comment)
    raw.columns = [c.strip() for c in raw.columns]

    if timecol is None or poscols is None or velcols is None:
        gt, gp, gv = autcol(raw.columns)
        timecol = timecol or gt
        poscols = poscols or gp
        velcols = velcols or gv

    missing = [c for c in [timecol, *poscols, *velcols] if c is None or c not in raw.columns]
    if missing:
        raise ValueError(f"could not resolve external columns: {missing}; "
                         f"available columns are {list(raw.columns)}")

    length = LENUNITS[lenunit]
    tscale = TIMEUNITS[timeunit]
    return pd.DataFrame({
        "t_s": (raw[timecol].to_numpy(dtype=float) - float(t0)) * tscale,
        "x_m": raw[poscols[0]].to_numpy(dtype=float) * length,
        "y_m": raw[poscols[1]].to_numpy(dtype=float) * length,
        "z_m": raw[poscols[2]].to_numpy(dtype=float) * length,
        "vx_mps": raw[velcols[0]].to_numpy(dtype=float) * length,
        "vy_mps": raw[velcols[1]].to_numpy(dtype=float) * length,
        "vz_mps": raw[velcols[2]].to_numpy(dtype=float) * length,
    })


def cmpeph(internal, external):
    internal = internal.reset_index(drop=True)
    t = internal["t_s"].to_numpy(dtype=float)
    extt = external["t_s"].to_numpy(dtype=float)

    lo, hi = max(t.min(), extt.min()), min(t.max(), extt.max())
    mask = (t >= lo - 1e-9) & (t <= hi + 1e-9)
    tc = t[mask]
    if tc.size == 0:
        raise ValueError("internal and external ephemerides do not overlap in time")

    poscols = ["x_m", "y_m", "z_m"]
    velcols = ["vx_mps", "vy_mps", "vz_mps"]
    posint = internal.loc[mask, poscols].to_numpy()
    velint = internal.loc[mask, velcols].to_numpy()
    posext = np.column_stack([np.interp(tc, extt, external[c].to_numpy()) for c in poscols])
    velext = np.column_stack([np.interp(tc, extt, external[c].to_numpy()) for c in velcols])

    poserr = np.linalg.norm(posint - posext, axis=1)
    velerr = np.linalg.norm(velint - velext, axis=1)

    def rms(a):
        return float(np.sqrt(np.mean(a ** 2)))

    summary = pd.DataFrame([{
        "n_compared": int(tc.size),
        "max_position_error_m": float(poserr.max()),
        "final_position_error_m": float(poserr[-1]),
        "mean_position_error_m": float(poserr.mean()),
        "rms_position_error_m": rms(poserr),
        "max_velocity_error_mps": float(velerr.max()),
        "final_velocity_error_mps": float(velerr[-1]),
        "mean_velocity_error_mps": float(velerr.mean()),
        "rms_velocity_error_mps": rms(velerr),
    }])
    return {"times": tc, "pos_error": poserr, "vel_error": velerr, "summary": summary}
