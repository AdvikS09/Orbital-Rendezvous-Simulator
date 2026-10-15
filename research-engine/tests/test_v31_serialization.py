"""v3.1 C03 regressions: canonical serialization hashes exactly the stored bytes; legacy v3.0 hashes are
explained by the NaN -> 'nan' (hash) vs NaN -> null (stored) mismatch; IDs are unchanged."""

import hashlib
import json
import math
import os

import numpy as np
import pytest

from cwvalid3 import ledger as L
from cwvalid3 import serialization as SER

from . import v31fx as FX


def sample_record():
    return {"run_id": "R1", "a": float("nan"), "b": np.float64(1.0) / 3, "c": np.int64(7), "d": np.bool_(True),
            "e": [np.float32(0.5), (1, 2.5)], "f": {"g": None, "h": -0.0, "i": 1e-300, "j": 12345678901234567890},
            "k": np.array([1.0, float("nan")]), "z": "text", "inf": float("inf"), "ninf": -np.inf}


def test_normalize_rules():
    n = SER.normalize(sample_record())
    assert n["a"] is None and n["k"] == [1.0, None]
    assert n["inf"] == "inf" and n["ninf"] == "-inf"
    assert isinstance(n["c"], int) and n["d"] is True and n["e"] == [0.5, [1, 2.5]]
    assert math.copysign(1.0, n["f"]["h"]) == -1.0
    for bad in ({"x": {1, 2}}, {"x": object()}, {1: 2}, {"x": b"bytes"}):
        with pytest.raises(TypeError):
            SER.normalize(bad)


def test_roundtrip_and_idempotence(tmp_path):
    rec = sample_record()
    assert SER.roundtrip_ok(rec)
    p = tmp_path / "r.jsonl"
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        h, ph = SER.write_record(fh, rec)
    raw = p.read_bytes()
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    line = raw.rstrip(b"\n")
    assert hashlib.sha256(line).hexdigest() == h == SER.stored_line_sha256(raw)
    assert SER.is_canonical_line(raw)
    back = json.loads(line)
    assert back["record_schema"] == SER.RECORD_SCHEMA
    b2, h2 = SER.canonical_line(back, None)
    assert b2 == line and h2 == h
    assert back["b"] == float(np.float64(1.0) / 3)          # exact float round trip
    with open(tmp_path / "b.jsonl", "wb") as fh:
        assert SER.write_record(fh, rec)[0] == h


def test_physics_hash_ignores_only_volatile_fields():
    r = {"run_id": "R1", "x": 1.5, "cpu_s": 3.0, "wall_s": 4.0, "wall_metrics_s": 1.0, "software": {"v": 1},
         "completed_at": "now", "batch_size": 64}
    q = {**r, "cpu_s": 9.0, "wall_metrics_s": 7.0, "software": {"v": 2}, "completed_at": "later", "batch_size": 45}
    assert SER.physics_sha256(r) == SER.physics_sha256(q)
    assert SER.physics_sha256(r) != SER.physics_sha256({**r, "x": 1.5000000000000002})
    assert SER.canonical_line(r)[1] != SER.canonical_line(q)[1]


def test_nan_and_null_are_the_same_stored_value_by_contract():
    assert SER.canonical_line({"x": float("nan")}, None) == SER.canonical_line({"x": None}, None)


def test_engine_identity_is_path_based(tmp_path):
    a = tmp_path / "pkg1" / "m.py"
    b = tmp_path / "pkg2" / "m.py"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_bytes(b"x = 1\r\n")
    b.write_bytes(b"x = 2\n")
    e1 = SER.engine_identity([str(a), str(b)], str(tmp_path))
    assert set(e1["files"]) == {"pkg1/m.py", "pkg2/m.py"}
    a.write_bytes(b"x = 1\n")                                    # CRLF-insensitive
    assert SER.engine_identity([str(a), str(b)], str(tmp_path))["sha256"] == e1["sha256"]
    b.write_bytes(b"x = 3\n")
    assert SER.engine_identity([str(a), str(b)], str(tmp_path))["sha256"] != e1["sha256"]
    # relocating a module (same name and content, other package) changes the v3.1 identity, not the v3.0 one
    c = tmp_path / "pkg3" / "m.py"
    c.parent.mkdir()
    c.write_bytes(a.read_bytes())
    assert L.code_version([str(a)]) == L.code_version([str(c)])
    assert SER.engine_identity([str(a)], str(tmp_path))["sha256"] != SER.engine_identity([str(c)], str(tmp_path))["sha256"]


@pytest.mark.parametrize("cs", ["D0_test", "D3_cal", "D5_dev"])
def test_ids_unchanged_by_v31(cs):
    """case_id, physical_id and run_id recompute from the manifest with the unchanged v3.0 formulas."""
    summ = FX.summary(cs)                       # exact fixture copies of the original manifest lines and summary
    for m in FX.manifest(cs, 25):
        s = m["scenario"]
        assert L.physical_id(s["rel0"], s["altitude"], s["horizon_orbits"], m["K"], s["a_max"]) == m["physical_id"]
        assert "C" + L.stable_hash({"cell": m["cell"], "split": m["split"], "index": m["index"], "draw": m["draw"]}) == m["case_id"]
        assert L.run_id(m["physical_id"], summ["run_config"]) == m["run_id"]


def _legacy_hash_from_stored(stored, nan_fields):
    o = dict(stored)
    for k in nan_fields:
        o[k] = float("nan")
    return L.stable_hash({k: v for k, v in o.items() if k not in ("software", "completed_at")}, 32)


def test_legacy_mismatch_is_the_nan_null_rule():
    """A stored v3.0 record with nulls reproduces its ledger hash only when the nulls that were NaN
    are hashed as 'nan' (the v3.0 in-memory value), never as stored."""
    recs = FX.first("raw/results/D3_cal.jsonl", 50)
    want = {r["run_id"] for r in recs}
    led = {e["run_id"]: e["result_sha256_16"] for e in FX.ledger() if e["run_id"] in want and e.get("event") == "completed"}
    assert set(led) == want
    n_null = 0
    for r in recs:
        nulls = [k for k, v in r.items() if v is None]
        if not nulls:
            assert _legacy_hash_from_stored(r, []) == led[r["run_id"]]
            continue
        n_null += 1
        # D3/D4 non-arrival: entry and dwell fields were NaN in memory
        assert _legacy_hash_from_stored(r, []) != led[r["run_id"]]
        assert _legacy_hash_from_stored(r, nulls) == led[r["run_id"]]
    assert n_null > 0


def test_normalize_edge_cases():
    assert SER.normalize({"a": np.array(1.5), "b": np.array(3), "c": np.array([True, False])}) == {"a": 1.5, "b": 3, "c": [True, False]}
    for bad in (np.array("ok"), np.array(b"ab"), np.array([object()], dtype=object), np.array(["x", "y"])):
        with pytest.raises(TypeError):
            SER.normalize({"x": bad})


def _first(cs, pred, n=1):
    """First n stored records of a file satisfying pred (exact fixture lines; the selection keeps these matches)."""
    return [(None, r) for r in FX.first_matching(f"raw/results/{cs}.jsonl", pred, n)]


RECORD_CLASSES = [
    ("D0_test", lambda r: r.get("eci_terminal_arrived") is True),                       # arrived
    ("D0_test", lambda r: r.get("eci_terminal_arrived") is False and r.get("eci_status") == "ok"),  # completed non-arrival
    ("D3_cal", lambda r: r.get("eci_status") == "domain_exit"),                           # exit (held)
    ("D0_cal", lambda r: r.get("eci_status") == "terminated"),                            # exit (continuous)
    ("D1_cal", lambda r: r.get("worker_status") == "worker_crash"),                       # interruption
    ("D5_test", lambda r: r.get("kp") is None),                                           # matrix LQR, kp/kd undefined
]


@pytest.mark.parametrize("cs,pred", RECORD_CLASSES)
def test_real_record_classes_roundtrip(cs, pred, tmp_path):
    hits = _first(cs, pred)
    assert hits, cs
    _, r = hits[0]
    assert SER.roundtrip_ok(r)
    f = tmp_path / "x.jsonl"
    with open(f, "w", encoding="utf-8", newline="\n") as fh:
        h, ph = SER.write_record(fh, r)
    line = f.read_bytes()
    assert SER.stored_line_sha256(line) == h and SER.is_canonical_line(line)
    back = json.loads(line)
    assert SER.physics_sha256(back) == ph
    assert SER.legacy_comparable_sha256(back) == SER.legacy_comparable_sha256({**r, "feature_stage": "complete", "eci_exit_step": 3})


def _none_fields():
    import importlib.util
    spec = importlib.util.spec_from_file_location("c03", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                                      "experiments", "v3_1", "c03_hash_reconciliation.py"))
    sys_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "v3_1")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.none_fields


@pytest.mark.parametrize("cs,pred", [("D0_cal", lambda r: True), ("D5_cal", lambda r: True),
                                     ("D1_cal", lambda r: r.get("worker_status") == "worker_crash")])
def test_legacy_rule_with_its_exceptions(cs, pred):
    """Rule nan-restore/1 including its Python-None exceptions (sat_events_disabled_at_s, d1_failure_reason,
    cpu_s/wall_s of crashes) reproduces the ledger hash; restoring every null instead fails where None was real."""
    none_fields = _none_fields()
    recs = [r for _, r in _first(cs, pred, 40)]
    want = {r["run_id"] for r in recs}
    led = {e["run_id"]: e["result_sha256_16"] for e in FX.ledger()
           if e["run_id"] in want and e.get("event") in ("completed", "completed_reconciled")}
    assert set(led) == want
    exc_used = 0
    for r in recs:
        nf = none_fields(r)
        rest = [k for k, v in r.items() if v is None and k not in nf and k not in ("software", "completed_at")]
        exc_used += any(r.get(k, 0) is None for k in nf)
        assert _legacy_hash_from_stored(r, rest) == led[r["run_id"]]
    assert exc_used > 0


def test_status_out_checks_stored_line_hashes(tmp_path):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "v3"))
    import campaign as CP
    recs = FX.first("raw/results/D0_test.jsonl", 3)
    (tmp_path / "results").mkdir()
    with open(tmp_path / "results" / "D0_test.jsonl", "w", encoding="utf-8", newline="\n") as fh, \
            open(tmp_path / "RUN_LEDGER.jsonl", "w", encoding="utf-8", newline="\n") as lf:
        for r in recs:
            h, ph = SER.write_record(fh, r)
            SER.write_record(lf, {"ledger_id": f"completed:{r['run_id']}", "event": "completed", "run_id": r["run_id"],
                                  "record_sha256": h, "physics_sha256": ph}, schema=None)
    rows = CP.cmd_status_out(str(tmp_path))
    assert rows[0]["ledger_hash_mismatch_or_missing"] == 0 and rows[0]["n_rows"] == 3
    data = (tmp_path / "results" / "D0_test.jsonl").read_bytes().replace(b'"index":', b'"index": ', 1)
    (tmp_path / "results" / "D0_test.jsonl").write_bytes(data)
    assert CP.cmd_status_out(str(tmp_path))[0]["ledger_hash_mismatch_or_missing"] == 1
