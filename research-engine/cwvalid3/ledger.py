import hashlib
import json
import math
import os

TRUTHSTATUSES = ("completed_valid", "physical_domain_exit", "numerical_failure", "unresolved")
FEATSTATUSES = ("ok", "unavailable")
ARRSTATUSES = ("arrived", "completed_non_arrival", "unresolved", "not_applicable")


def canon(obj):
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return repr(obj)
    if isinstance(obj, dict):
        return {str(k): canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canon(v) for v in obj]
    if hasattr(obj, "tolist"):
        return canon(obj.tolist())
    return obj


def stablehash(obj, n=16):
    s = json.dumps(canon(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def filesha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def codever(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as fh:
            h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def physical_id(rel0, altitude, horizon_orbits, K, a_max):
    return "P" + stablehash({"rel0": list(map(float, rel0)), "altitude_m": float(altitude),
                             "chief": "circular_two_body", "horizon_orbits": float(horizon_orbits),
                             "K": [[float(v) for v in row] for row in K],
                             "a_max": None if a_max is None else float(a_max)})


def run_id(physid, run_config):
    return "R" + stablehash({"physical_id": physid, "config": run_config})


class StrictJSONLErr(ValueError):
    pass


def readjsonlstrict(path, key=None, allow_missing=False):
    if not os.path.exists(path):
        if allow_missing:
            return []
        raise StrictJSONLErr(f"{path}: missing")
    with open(path, "rb") as fh:
        data = fh.read()
    if data and not data.endswith(b"\n"):
        raise StrictJSONLErr(f"{path}: final line has no newline (truncated write?)")
    rows, seen = [], {}
    for i, raw in enumerate(data.split(b"\n")[:-1] if data else [], start=1):
        line = raw.decode("utf-8").strip()
        if not line:
            raise StrictJSONLErr(f"{path}:{i}: empty line")
        try:
            r = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StrictJSONLErr(f"{path}:{i}: malformed JSON ({exc})") from None
        if not isinstance(r, dict):
            raise StrictJSONLErr(f"{path}:{i}: record is not an object")
        if key is not None:
            if key not in r:
                raise StrictJSONLErr(f"{path}:{i}: missing key field {key!r}")
            if r[key] in seen:
                raise StrictJSONLErr(f"{path}:{i}: duplicate {key}={r[key]!r} (first at line {seen[r[key]]})")
            seen[r[key]] = i
        rows.append(r)
    return rows


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if hasattr(v, "tolist"):
        return jsonable(v.tolist())
    if isinstance(v, float) and not math.isfinite(v):
        return None if math.isnan(v) else ("inf" if v > 0 else "-inf")
    return v


def appendjsonl(path, rec, fh=None):
    line = json.dumps(jsonable(rec), sort_keys=False, allow_nan=False) + "\n"
    if fh is not None:
        fh.write(line)
        fh.flush()
        return
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line)


def reconcile(planned_ids, rows, key):
    got = [r[key] for r in rows]
    gs = set(got)
    ps = list(planned_ids)
    pset = set(ps)
    return {"n_planned": len(ps), "n_rows": len(got), "n_missing": len([p for p in ps if p not in gs]),
            "n_extra": len([g for g in gs if g not in pset]), "n_duplicates": len(got) - len(gs),
            "missing": [p for p in ps if p not in gs][:50], "extra": [g for g in gs if g not in pset][:50]}
