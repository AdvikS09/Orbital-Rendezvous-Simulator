import numpy as np
import pandas as pd


def mindist(posit, reference):
    pos = np.asarray(posit, dtype=float)[:, :3]
    reference = np.asarray(reference, dtype=float)
    ref = reference[:3] if reference.ndim == 1 else reference[:, :3]
    dist = np.linalg.norm(pos - ref, axis=1)
    idx = int(np.argmin(dist))
    return dist, float(dist[idx]), idx


def analyze(times, chaserstates, keep_out_radius, debris_states=None, collision_radius=5.0):
    times = np.asarray(times, dtype=float)
    chaserpos = np.asarray(chaserstates, dtype=float)[:, :3]

    disttgt = np.linalg.norm(chaserpos, axis=1)
    it = int(np.argmin(disttgt))
    res = {
        "times": times,
        "dist_to_target": disttgt,
        "min_dist_to_target_m": float(disttgt[it]),
        "closest_target_time_s": float(times[it]),
        "keep_out_radius_m": float(keep_out_radius),
        "keep_out_violation": bool(np.any(disttgt < keep_out_radius)),
    }

    if debris_states is not None:
        debris = np.asarray(debris_states, dtype=float)
        debrispos = debris[:, :3] if debris.ndim == 2 else np.broadcast_to(debris[:3], chaserpos.shape)
        distdebris = np.linalg.norm(chaserpos - debrispos, axis=1)
        idd = int(np.argmin(distdebris))
        res.update({
            "dist_to_debris": distdebris,
            "min_dist_to_debris_m": float(distdebris[idd]),
            "closest_debris_time_s": float(times[idd]),
            "collision_radius_m": float(collision_radius),
            "collision_risk": bool(distdebris[idd] < collision_radius),
        })

    res["summary"] = summ(res)
    return res


def summ(result):
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
