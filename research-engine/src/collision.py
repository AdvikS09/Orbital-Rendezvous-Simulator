"""Simple collision / keep-out-zone *detection* (a future-work extension).

This is intentionally minimal: it does **not** do avoidance or path planning. It
only measures distances and flags risks for a given chaser trajectory in the
target LVLH frame:

* a spherical keep-out zone of radius ``keep_out_radius`` around the target
  (LVLH origin),
* an optional debris point (static, or moving along its own CW trajectory),
* the minimum chaser-target and chaser-debris distances and the times they occur,
* boolean flags for keep-out-zone violation and debris collision risk.

All quantities are SI (metres, seconds). It operates on LVLH relative states
``[x, y, z, vx, vy, vz]`` produced by the engine.
"""

import numpy as np
import pandas as pd


def minimum_distance(positions, reference):
    """Distance series to a reference point plus the minimum and its index.

    ``reference`` may be a single 3-vector or a per-sample array of points.
    Returns ``(distance_series, min_distance, argmin_index)``.
    """
    pos = np.asarray(positions, dtype=float)[:, :3]
    reference = np.asarray(reference, dtype=float)
    ref = reference[:3] if reference.ndim == 1 else reference[:, :3]
    dist = np.linalg.norm(pos - ref, axis=1)
    idx = int(np.argmin(dist))
    return dist, float(dist[idx]), idx


def analyze(times, chaser_states, keep_out_radius, debris_states=None, collision_radius=5.0):
    """Detect keep-out-zone and debris collision risk for a chaser trajectory.

    Parameters
    ----------
    times : array of sample times [s].
    chaser_states : (N, 6) LVLH relative states of the chaser.
    keep_out_radius : radius of the spherical keep-out zone around the target [m].
    debris_states : optional (N, 6) LVLH states of a debris object (or an
        (N, 3)/3-vector position); if given, debris collision risk is assessed.
    collision_radius : chaser-debris separation below which a collision is
        flagged [m].
    """
    times = np.asarray(times, dtype=float)
    chaser_pos = np.asarray(chaser_states, dtype=float)[:, :3]

    dist_target = np.linalg.norm(chaser_pos, axis=1)
    it = int(np.argmin(dist_target))
    result = {
        "times": times,
        "dist_to_target": dist_target,
        "min_dist_to_target_m": float(dist_target[it]),
        "closest_target_time_s": float(times[it]),
        "keep_out_radius_m": float(keep_out_radius),
        "keep_out_violation": bool(np.any(dist_target < keep_out_radius)),
    }

    if debris_states is not None:
        debris = np.asarray(debris_states, dtype=float)
        debris_pos = debris[:, :3] if debris.ndim == 2 else np.broadcast_to(debris[:3], chaser_pos.shape)
        dist_debris = np.linalg.norm(chaser_pos - debris_pos, axis=1)
        idd = int(np.argmin(dist_debris))
        result.update({
            "dist_to_debris": dist_debris,
            "min_dist_to_debris_m": float(dist_debris[idd]),
            "closest_debris_time_s": float(times[idd]),
            "collision_radius_m": float(collision_radius),
            "collision_risk": bool(dist_debris[idd] < collision_radius),
        })

    result["summary"] = _summary(result)
    return result


def _summary(result):
    row = {
        "keep_out_radius_m": result["keep_out_radius_m"],
        "min_dist_to_target_m": result["min_dist_to_target_m"],
        "keep_out_violation": result["keep_out_violation"],
    }
    if "min_dist_to_debris_m" in result:
        row.update({
            "collision_radius_m": result["collision_radius_m"],
            "min_dist_to_debris_m": result["min_dist_to_debris_m"],
            "collision_risk": result["collision_risk"],
        })
    return pd.DataFrame([row])


__all__ = ["minimum_distance", "analyze"]
