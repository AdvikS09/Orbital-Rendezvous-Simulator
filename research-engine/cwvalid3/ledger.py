"""Run identity, strict JSONL I/O, status taxonomy and the run ledger (P00).

Identity
  case_id      the drawn unit: cell, split, index and the drawn parameters. Unique per draw.
  physical_id  everything that defines the physical closed-loop problem EXCEPT command timing:
               initial LVLH state, chief orbit (altitude, circular two-body), horizon, gain matrix K
               (exact values), cap. Used to pair the same physical case across methods and, for
               the transfer-diagnostic cohort only, across command timings.
  run_id       physical_id plus command timing (mode, hold period, update phase), branch models,
               reference/cross-check, solver and monitoring configuration, adequacy criteria and the
               implementation version (SHA-256 of the engine source files). A continuous-command
               result can never be served for a sampled-command request because the two differ in
               run_id.
Hashes use SHA-256 of canonical JSON (sorted keys, floats written with repr precision).

Strict JSONL: every line must parse, every record must carry the key field, duplicates are
errors, and a final line without a newline is reported as truncated. Nothing is skipped.

Status taxonomy (per case, recorded with reason, last valid time, solver message and nfev):
  truth_status     completed_valid | physical_domain_exit | numerical_failure | unresolved
  feature_status   ok | unavailable
  arrival_status   arrived | completed_non_arrival | unresolved | not_applicable
Non-arrival is not a failure; an unavailable feature is not a zero; a physical-domain exit is adverse
for the full-horizon adequacy predicate.
"""

import hashlib
import json
import math
import os

TRUTH_STATUSES = ("completed_valid", "physical_domain_exit", "numerical_failure", "unresolved")
FEATURE_STATUSES = ("ok", "unavailable")
ARRIVAL_STATUSES = ("arrived", "completed_non_arrival", "unresolved", "not_applicable")


def _canon(obj):
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return repr(obj)
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if hasattr(obj, "tolist"):
        return _canon(obj.tolist())
    return obj


def stable_hash(obj, n=16):
    s = json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(path):
    """SHA-256 of a text file with CRLF normalised to LF (stable under git autocrlf checkouts)."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def code_version(paths):
    """SHA-256 over the given source files (sorted by path), first 16 hex chars."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as fh:
            h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def physical_id(rel0, altitude, horizon_orbits, K, a_max):
    return "P" + stable_hash({"rel0": list(map(float, rel0)), "altitude_m": float(altitude),
                              "chief": "circular_two_body", "horizon_orbits": float(horizon_orbits),
                              "K": [[float(v) for v in row] for row in K],
                              "a_max": None if a_max is None else float(a_max)})


def run_id(phys_id, run_config):
    return "R" + stable_hash({"physical_id": phys_id, "config": run_config})


class StrictJSONLError(ValueError):
    pass


def read_jsonl_strict(path, key=None, allow_missing=False):
    """Parse every line; raise on malformed, truncated or duplicate-key records."""
    if not os.path.exists(path):
        if allow_missing:
            return []
        raise StrictJSONLError(f"{path}: missing")
    with open(path, "rb") as fh:
        data = fh.read()
    if data and not data.endswith(b"\n"):
        raise StrictJSONLError(f"{path}: final line has no newline (truncated write?)")
    rows, seen = [], {}
    for i, raw in enumerate(data.split(b"\n")[:-1] if data else [], start=1):
        line = raw.decode("utf-8").strip()
        if not line:
            raise StrictJSONLError(f"{path}:{i}: empty line")
        try:
            r = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StrictJSONLError(f"{path}:{i}: malformed JSON ({exc})") from None
        if not isinstance(r, dict):
            raise StrictJSONLError(f"{path}:{i}: record is not an object")
        if key is not None:
            if key not in r:
                raise StrictJSONLError(f"{path}:{i}: missing key field {key!r}")
            if r[key] in seen:
                raise StrictJSONLError(f"{path}:{i}: duplicate {key}={r[key]!r} (first at line {seen[r[key]]})")
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


def append_jsonl(path, rec, fh=None):
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


__all__ = ["TRUTH_STATUSES", "FEATURE_STATUSES", "ARRIVAL_STATUSES", "stable_hash", "file_sha256", "text_sha256", "code_version",
           "physical_id", "run_id", "read_jsonl_strict", "StrictJSONLError", "jsonable", "append_jsonl", "reconcile"]
