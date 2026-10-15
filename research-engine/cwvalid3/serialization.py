"""v3.1 canonical record serialization and engine identity: hash exactly what is stored (CORRECTION C03).

v3.0 computed a record's ledger hash (ledger.stable_hash, 16 hex) on the in-memory dictionary, where
an undefined metric is float('nan') and is hashed as the string 'nan', and then wrote the record with
ledger.jsonable, which turns NaN into JSON null. The stored line therefore could not reproduce the
ledger hash for any record that contained a NaN (17,615 of 47,983 records), and 43 'rerun' ledger
events carry no hash at all. v3.1 normalises once, hashes the exact bytes that are written, and
writes those bytes. Legacy hashes are reconciled in a sidecar file, never rewritten in place.

Canonical record form (RECORD_SCHEMA):
  * JSON object, keys sorted, separators (',', ':'), ensure_ascii, one record per line, LF ending;
  * float NaN -> null. JSON has no NaN; in a v3.1 record null means "undefined" for a numeric field
    and "not applicable" otherwise, and readers must not distinguish the two (the v3.0 readers already
    treat null as NaN);
  * +inf / -inf -> the strings "inf" / "-inf" (the v3.0 convention; none occur in campaign records);
  * numpy scalars -> Python int/float/bool, 0-d arrays -> their item, numeric/bool numpy arrays and tuples
    -> lists; -0.0 is kept; floats use Python's shortest round-trip repr, so json.loads restores them exactly;
  * any other type (set, object, bytes, str/bytes/object-dtype arrays, extended-precision floats) raises
    TypeError instead of being written approximately. Note: the strings "inf"/"-inf" and the floats +/-inf
    produce the same bytes (v3.0 convention).
  record_sha256  = SHA-256 of the canonical line WITHOUT its trailing newline (all fields, incl. provenance).
  physics_sha256 = SHA-256 of the canonical form of the record without the volatile fields (timing, host and
                   software provenance, per-branch solver wall times, batch size and membership, schema and replay
                   bookkeeping): it compares v3.1 records with v3.1 records (e.g. a replay with its original run).
                   Its definition is versioned (PHYSICS_HASH_VERSION, stored next to every physics hash).
  legacy_comparable_sha256 = physics hash that also drops the fields only v3.1 workers write (V31_ONLY_FIELDS), so a
                   v3.1 record can be compared with a v3.0 record by hash.
Canonicalisation is idempotent: canonical(json.loads(canonical(r))) == canonical(r).

Engine identity (ENGINE_ID_SCHEME): SHA-256 over (relative path, LF-normalised file SHA-256) for an
explicit list of source files, sorted by relative path. v3.0's ledger.code_version hashed basenames
and omitted imported modules (cwvalid/__init__.py, cwvalid3/__init__.py and the src/ package).
"""

import hashlib
import json
import math
import os

import numpy as np

RECORD_SCHEMA = "cwvalid3.record.v3.1-canonical-1"
ENGINE_ID_SCHEME = "v3.1-relpath-sha256"
VOLATILE_KEYS = frozenset({"software", "completed_at", "record_schema", "cpu_s", "wall_s", "batch_cpu_s", "batch_wall_s",
                           "batch_size", "batch_members_sha256", "batch_retry", "replay", "task_retry",
                           "feature_checkpoint_error"})
VOLATILE_PREFIXES = ("wall_", "cpu_")
VOLATILE_SUFFIXES = ("_wall_time_s",)
# history: v3.1-physics-1 (no suffix rule; superseded before any stored v3.1 record), v3.1-physics-2 (+ '_wall_time_s'),
# v3.1-physics-3 (+ batch_members_sha256, batch_retry; identical to -2 for records without these keys),
# v3.1-physics-4 (v3.1.1: + task_retry, feature_checkpoint_error; identical to -3 for records without these keys, i.e.
# every stored v3.0/v3.1 record and every normal-path record)
PHYSICS_HASH_VERSION = "v3.1-physics-4"
# fields only v3.1+ workers write; v3.1.1 added the last four (failure-path and recovery fields only: no normal-path
# record and no stored v3.0/v3.1 record carries them, so no stored or ledgered hash changes)
V31_ONLY_FIELDS = frozenset({"feature_stage", "failed_stage", "feature_source", "feature_checkpoint_recovered",
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


def dumps_canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def canonical_line(rec, schema=RECORD_SCHEMA):
    """(line bytes without newline, record_sha256) of the canonical form of rec; adds record_schema."""
    payload = normalize(rec)
    if schema is not None:
        payload["record_schema"] = schema
    b = dumps_canonical(payload).encode("ascii")
    return b, hashlib.sha256(b).hexdigest()


def is_volatile(key):
    return key in VOLATILE_KEYS or key.startswith(VOLATILE_PREFIXES) or key.endswith(VOLATILE_SUFFIXES)


def physics_sha256(rec):
    payload = normalize(rec)
    keep = {k: v for k, v in payload.items() if not is_volatile(k)}
    return hashlib.sha256(dumps_canonical(keep).encode("ascii")).hexdigest()


def legacy_comparable_sha256(rec):
    """Physics hash without the v3.1-only fields: equal for a v3.1 record and a v3.0 record with the same physics."""
    payload = normalize(rec)
    keep = {k: v for k, v in payload.items() if not is_volatile(k) and k not in V31_ONLY_FIELDS}
    return hashlib.sha256(dumps_canonical(keep).encode("ascii")).hexdigest()


def batch_members_sha256(run_ids):
    """Order-sensitive hash of the run_ids that shared a held batch (batch composition can move results by an ulp)."""
    return hashlib.sha256("\n".join(run_ids).encode("ascii")).hexdigest()


def write_record(fh, rec, schema=RECORD_SCHEMA):
    """Write the canonical line (text or binary handle); return (record_sha256, physics_sha256)."""
    b, h = canonical_line(rec, schema)
    if "b" in getattr(fh, "mode", ""):
        fh.write(b + b"\n")
    else:
        fh.write(b.decode("ascii") + "\n")
    fh.flush()
    return h, physics_sha256(rec)


def stored_line_sha256(line):
    """SHA-256 of a stored line exactly as read (only the trailing LF / CRLF removed)."""
    if isinstance(line, str):
        line = line.encode("utf-8")
    return hashlib.sha256(line.rstrip(b"\r\n")).hexdigest()


def is_canonical_line(line):
    """True if a stored line is byte-identical to the canonical form of what it parses to."""
    if isinstance(line, str):
        line = line.encode("utf-8")
    raw = line.rstrip(b"\r\n")
    b, _ = canonical_line(json.loads(raw.decode("ascii")), None)
    return b == raw


def roundtrip_ok(rec, schema=RECORD_SCHEMA):
    """write -> read -> canonicalise -> identical bytes and hash."""
    b, h = canonical_line(rec, schema)
    back = json.loads(b.decode("ascii"))
    b2, h2 = canonical_line(back, None)
    return h == h2 and b == b2


def engine_identity(paths, root):
    """{'scheme', 'sha256', 'sha16', 'files': {relpath: sha256}} over LF-normalised source files."""
    files = {}
    for p in paths:
        rel = os.path.relpath(p, root).replace("\\", "/")
        with open(p, "rb") as fh:
            files[rel] = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode("utf-8") + b"\0" + files[rel].encode("ascii") + b"\n")
    d = h.hexdigest()
    return {"scheme": ENGINE_ID_SCHEME, "sha256": d, "sha16": d[:16], "files": dict(sorted(files.items()))}


__all__ = ["RECORD_SCHEMA", "ENGINE_ID_SCHEME", "PHYSICS_HASH_VERSION", "V31_ONLY_FIELDS", "legacy_comparable_sha256",
           "batch_members_sha256", "VOLATILE_KEYS", "is_volatile", "normalize", "dumps_canonical", "canonical_line",
           "physics_sha256", "write_record", "stored_line_sha256", "is_canonical_line", "roundtrip_ok", "engine_identity"]
