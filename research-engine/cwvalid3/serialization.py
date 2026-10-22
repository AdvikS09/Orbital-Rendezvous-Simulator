import hashlib
import json
import math
import os

import numpy as np

RECSCHEMA = "cwvalid3.record.v3.1-canonical-1"
ENGINEIDSCHEME = "v3.1-relpath-sha256"
VOLKEYS = frozenset({"software", "completed_at", "record_schema", "cpu_s", "wall_s", "batch_cpu_s", "batch_wall_s",
                     "batch_size", "batch_members_sha256", "batch_retry", "replay", "task_retry",
                     "feature_checkpoint_error"})
VOLPFXS = ("wall_", "cpu_")
VOLSFXS = ("_wall_time_s",)
PHYSHASHVER = "v3.1-physics-4"
V31ONLYFIELDS = frozenset({"feature_stage", "failed_stage", "feature_source", "feature_checkpoint_recovered",
                           "reference_error", "naive_pass_error", "nerm_exit_step", "nerm_exit_fallback", "eci_exit_step",
                           "eci_exit_fallback", "feature_errors", "feature_recovery", "feature_recovery_error",
                           "stale_checkpoint_ignored"})


def normalize(obj):
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.floating) and np.dtype(type(obj)).itemsize > 8:
        raise TypeError(f"extended-precision float {type(obj).__name__} would be rounded")
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"non-string key {k!r}")
            out[k] = normalize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind not in "biuf":
            raise TypeError(f"cannot serialise a numpy array of dtype {obj.dtype} canonically")
        if obj.dtype.kind == "f" and obj.dtype.itemsize > 8:
            raise TypeError(f"extended-precision array {obj.dtype} would be rounded")
        if obj.ndim == 0:
            return normalize(obj.item())
        return [normalize(v) for v in obj.tolist()]
    raise TypeError(f"cannot serialise {type(obj).__name__} canonically")


def dumpscan(payl):
    return json.dumps(payl, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


# Hash the canonical stored line rather than the in-memory dict, so the hash matches the bytes on disk
def canline(rec, schema=RECSCHEMA):
    doc = normalize(rec)
    if schema is not None:
        doc["record_schema"] = schema
    b = dumpscan(doc).encode("ascii")
    return b, hashlib.sha256(b).hexdigest()


def volat(key):
    return key in VOLKEYS or key.startswith(VOLPFXS) or key.endswith(VOLSFXS)


def physics_sha256(rec):
    doc = normalize(rec)
    keep = {k: v for k, v in doc.items() if not volat(k)}
    return hashlib.sha256(dumpscan(keep).encode("ascii")).hexdigest()


# Physics hash without the v3.1-only fields, so v3.1 and v3.0 records with the same physics compare equal
def legacycomp(rec):
    doc = normalize(rec)
    keep = {k: v for k, v in doc.items() if not volat(k) and k not in V31ONLYFIELDS}
    return hashlib.sha256(dumpscan(keep).encode("ascii")).hexdigest()


# Order-sensitive on purpose: the composition of a held batch can move results by an ulp
def batch_members_sha256(runids):
    return hashlib.sha256("\n".join(runids).encode("ascii")).hexdigest()


def write_record(fh, rec, schema=RECSCHEMA):
    b, h = canline(rec, schema)
    if "b" in getattr(fh, "mode", ""):
        fh.write(b + b"\n")
    else:
        fh.write(b.decode("ascii") + "\n")
    fh.flush()
    return h, physics_sha256(rec)


def stored_line_sha256(line):
    if isinstance(line, str):
        line = line.encode("utf-8")
    return hashlib.sha256(line.rstrip(b"\r\n")).hexdigest()


def canonline(line):
    if isinstance(line, str):
        line = line.encode("utf-8")
    raw = line.rstrip(b"\r\n")
    b, _ = canline(json.loads(raw.decode("ascii")), None)
    return b == raw


def roundtripok(rec, schema=RECSCHEMA):
    b, h = canline(rec, schema)
    back = json.loads(b.decode("ascii"))
    b2, h2 = canline(back, None)
    return h == h2 and b == b2


# Hash of the LF-normalised source files the workers import, keyed by relative path
def engine_identity(paths, root):
    files = {}
    for p in paths:
        rel = os.path.relpath(p, root).replace("\\", "/")
        with open(p, "rb") as fh:
            files[rel] = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode("utf-8") + b"\0" + files[rel].encode("ascii") + b"\n")
    d = h.hexdigest()
    return {"scheme": ENGINEIDSCHEME, "sha256": d, "sha16": d[:16], "files": dict(sorted(files.items()))}
