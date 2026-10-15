"""External-tool validation utilities (GMAT / STK, etc.).

This module does **not** drive any external tool. It provides the data plumbing
to validate the engine's higher-fidelity propagator against an external one:

1. serialise an engine-propagated ephemeris (+ scenario metadata) to CSV / JSON
   so the same scenario can be reproduced in GMAT or STK,
2. load an external tool's exported ephemeris CSV back in (flexible column
   mapping and unit handling), and
3. compare the two ephemerides -- position / velocity error series and summary
   metrics (max, final, mean, RMS).

Everything is SI internally (metres, metres/second, seconds, ECI frame). The
functions operate on plain pandas DataFrames with the standard columns
``t_s, x_m, y_m, z_m, vx_mps, vy_mps, vz_mps``.
"""

import json

import numpy as np
import pandas as pd

STANDARD_COLUMNS = ["t_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]

_LENGTH_UNITS = {"m": 1.0, "km": 1000.0}
_TIME_UNITS = {"s": 1.0, "sec": 1.0, "min": 60.0, "h": 3600.0, "hr": 3600.0}


# --------------------------------------------------------------------------
# Serialisation (engine -> file text for GMAT/STK setup)
# --------------------------------------------------------------------------
def to_csv_text(ephemeris, metadata):
    """Serialise an ephemeris DataFrame to CSV text with metadata header comments.

    Metadata lines are prefixed with ``#`` so they are easy to read and are
    skipped on re-import (``pandas.read_csv(comment="#")``).
    """
    lines = [
        "# Orbital Rendezvous Engine -- external-validation scenario export",
        "# Units: position [m], velocity [m/s], time [s]; frame: ECI",
    ]
    for key, value in metadata.items():
        lines.append(f"# {key}: {value}")
    return "\n".join(lines) + "\n" + ephemeris.to_csv(index=False)


def to_json_text(ephemeris, metadata):
    """Serialise a full scenario (metadata + ephemeris) to JSON text."""
    payload = {
        "metadata": metadata,
        "ephemeris": {col: [float(v) for v in ephemeris[col]] for col in ephemeris.columns},
    }
    return json.dumps(payload, indent=2)


# --------------------------------------------------------------------------
# Loading external ephemeris (GMAT/STK CSV -> standard DataFrame)
# --------------------------------------------------------------------------
def _guess_column(columns, candidates):
    lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for c in columns:
        cl = c.lower()
        if any(cand in cl for cand in candidates):
            return c
    return None


def autodetect_columns(columns):
    """Best-effort guess of (time, [x,y,z], [vx,vy,vz]) column names."""
    t = _guess_column(columns, ["t_s", "time", "elapsed", "epoch", "t"])
    x = _guess_column(columns, ["x_m", "x (km)", "x_km", "x", "rx", "pos_x"])
    y = _guess_column(columns, ["y_m", "y (km)", "y_km", "y", "ry", "pos_y"])
    z = _guess_column(columns, ["z_m", "z (km)", "z_km", "z", "rz", "pos_z"])
    vx = _guess_column(columns, ["vx_mps", "vx (km/s)", "vx", "vel_x"])
    vy = _guess_column(columns, ["vy_mps", "vy (km/s)", "vy", "vel_y"])
    vz = _guess_column(columns, ["vz_mps", "vz (km/s)", "vz", "vel_z"])
    return t, [x, y, z], [vx, vy, vz]


def load_external(source, time_col=None, pos_cols=None, vel_cols=None,
                  length_unit="m", time_unit="s", t0=0.0, comment="#"):
    """Load an external ephemeris into the standard SI DataFrame.

    Parameters
    ----------
    source : path or file-like buffer (CSV).
    time_col, pos_cols, vel_cols : column names; auto-detected if omitted.
    length_unit : "m" or "km" (applies to position; velocity uses unit/s).
    time_unit : "s", "min", or "h".
    t0 : value subtracted from the time column so the series starts near 0 [in
         the file's time unit].
    """
    raw = pd.read_csv(source, comment=comment)
    raw.columns = [c.strip() for c in raw.columns]

    if time_col is None or pos_cols is None or vel_cols is None:
        gt, gp, gv = autodetect_columns(raw.columns)
        time_col = time_col or gt
        pos_cols = pos_cols or gp
        vel_cols = vel_cols or gv

    missing = [c for c in [time_col, *pos_cols, *vel_cols] if c is None or c not in raw.columns]
    if missing:
        raise ValueError(f"could not resolve external columns: {missing}; "
                         f"available columns are {list(raw.columns)}")

    length = _LENGTH_UNITS[length_unit]
    tscale = _TIME_UNITS[time_unit]
    return pd.DataFrame({
        "t_s": (raw[time_col].to_numpy(dtype=float) - float(t0)) * tscale,
        "x_m": raw[pos_cols[0]].to_numpy(dtype=float) * length,
        "y_m": raw[pos_cols[1]].to_numpy(dtype=float) * length,
        "z_m": raw[pos_cols[2]].to_numpy(dtype=float) * length,
        "vx_mps": raw[vel_cols[0]].to_numpy(dtype=float) * length,
        "vy_mps": raw[vel_cols[1]].to_numpy(dtype=float) * length,
        "vz_mps": raw[vel_cols[2]].to_numpy(dtype=float) * length,
    })


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def compare_ephemerides(internal, external):
    """Compare two SI ephemerides; returns error series and a summary table.

    The external ephemeris is linearly interpolated onto the internal time grid
    over the overlapping interval, so the two need not share sample times.
    """
    internal = internal.reset_index(drop=True)
    t = internal["t_s"].to_numpy(dtype=float)
    ext_t = external["t_s"].to_numpy(dtype=float)

    lo, hi = max(t.min(), ext_t.min()), min(t.max(), ext_t.max())
    mask = (t >= lo - 1e-9) & (t <= hi + 1e-9)
    t_c = t[mask]
    if t_c.size == 0:
        raise ValueError("internal and external ephemerides do not overlap in time")

    pos_cols = ["x_m", "y_m", "z_m"]
    vel_cols = ["vx_mps", "vy_mps", "vz_mps"]
    pos_int = internal.loc[mask, pos_cols].to_numpy()
    vel_int = internal.loc[mask, vel_cols].to_numpy()
    pos_ext = np.column_stack([np.interp(t_c, ext_t, external[c].to_numpy()) for c in pos_cols])
    vel_ext = np.column_stack([np.interp(t_c, ext_t, external[c].to_numpy()) for c in vel_cols])

    pos_err = np.linalg.norm(pos_int - pos_ext, axis=1)
    vel_err = np.linalg.norm(vel_int - vel_ext, axis=1)

    def _rms(a):
        return float(np.sqrt(np.mean(a ** 2)))

    summary = pd.DataFrame([{
        "n_compared": int(t_c.size),
        "max_position_error_m": float(pos_err.max()),
        "final_position_error_m": float(pos_err[-1]),
        "mean_position_error_m": float(pos_err.mean()),
        "rms_position_error_m": _rms(pos_err),
        "max_velocity_error_mps": float(vel_err.max()),
        "final_velocity_error_mps": float(vel_err[-1]),
        "mean_velocity_error_mps": float(vel_err.mean()),
        "rms_velocity_error_mps": _rms(vel_err),
    }])
    return {"times": t_c, "pos_error": pos_err, "vel_error": vel_err, "summary": summary}


__all__ = [
    "STANDARD_COLUMNS",
    "to_csv_text",
    "to_json_text",
    "autodetect_columns",
    "load_external",
    "compare_ephemerides",
]
