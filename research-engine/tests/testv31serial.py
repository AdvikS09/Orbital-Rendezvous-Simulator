import hashlib
import json
import math
import os

import numpy as np
import pytest

from cwvalid3 import ledger as L
from cwvalid3 import serialization as SER

from . import v31fx as FX


def samprec():
    return {"run_id": "R1", "a": float("nan"), "b": np.float64(1.0) / 3, "c": np.int64(7), "d": np.bool_(True),
            "e": [np.float32(0.5), (1, 2.5)], "f": {"g": None, "h": -0.0, "i": 1e-300, "j": 12345678901234567890},
            "k": np.array([1.0, float("nan")]), "z": "text", "inf": float("inf"), "ninf": -np.inf}


def testnormrules():
    n = SER.normalize(samprec())
    assert n["a"] is None and n["k"] == [1.0, None]
    assert n["inf"] == "inf" and n["ninf"] == "-inf"
    assert isinstance(n["c"], int) and n["d"] is True and n["e"] == [0.5, [1, 2.5]]
    assert math.copysign(1.0, n["f"]["h"]) == -1.0
    for bad in ({"x": {1, 2}}, {"x": object()}, {1: 2}, {"x": b"bytes"}):
        with pytest.raises(TypeError):
            SER.normalize(bad)


def testroundtripidemp(tmp_path):
    rec = samprec()
    assert SER.roundtripok(rec)
    p = tmp_path / "r.jsonl"
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        h, ph = SER.write_record(fh, rec)
    raw = p.read_bytes()
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    line = raw.rstrip(b"\n")
    assert hashlib.sha256(line).hexdigest() == h == SER.stored_line_sha256(raw)
    assert SER.canonline(raw)
    back = json.loads(line)
    assert back["record_schema"] == SER.RECSCHEMA
    b2, h2 = SER.canline(back, None)
    assert b2 == line and h2 == h
    assert back["b"] == float(np.float64(1.0) / 3)
    with open(tmp_path / "b.jsonl", "wb") as fh:
        assert SER.write_record(fh, rec)[0] == h


def testphyshashign():
    r = {"run_id": "R1", "x": 1.5, "cpu_s": 3.0, "wall_s": 4.0, "wall_metrics_s": 1.0, "software": {"v": 1},
         "completed_at": "now", "batch_size": 64}
    q = {**r, "cpu_s": 9.0, "wall_metrics_s": 7.0, "software": {"v": 2}, "completed_at": "later", "batch_size": 45}
    assert SER.physics_sha256(r) == SER.physics_sha256(q)
    assert SER.physics_sha256(r) != SER.physics_sha256({**r, "x": 1.5000000000000002})
    assert SER.canline(r)[1] != SER.canline(q)[1]


def testnannullsame():
    assert SER.canline({"x": float("nan")}, None) == SER.canline({"x": None}, None)


def testengineidentpath(tmp_path):
    a = tmp_path / "pkg1" / "m.py"
    b = tmp_path / "pkg2" / "m.py"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_bytes(b"x = 1\r\n")
    b.write_bytes(b"x = 2\n")
    e1 = SER.engine_identity([str(a), str(b)], str(tmp_path))
    assert set(e1["files"]) == {"pkg1/m.py", "pkg2/m.py"}
    a.write_bytes(b"x = 1\n")
    assert SER.engine_identity([str(a), str(b)], str(tmp_path))["sha256"] == e1["sha256"]
    b.write_bytes(b"x = 3\n")
    assert SER.engine_identity([str(a), str(b)], str(tmp_path))["sha256"] != e1["sha256"]
    c = tmp_path / "pkg3" / "m.py"
    c.parent.mkdir()
    c.write_bytes(a.read_bytes())
    assert L.codever([str(a)]) == L.codever([str(c)])
    assert SER.engine_identity([str(a)], str(tmp_path))["sha256"] != SER.engine_identity([str(c)], str(tmp_path))["sha256"]


@pytest.mark.parametrize("cs", ["D0test", "D3cal", "D5dev"])
def testidssame(cs):
    summ = FX.summary(cs)
    for m in FX.manifest(cs, 25):
        s = m["scenario"]
        assert L.physical_id(s["rel0"], s["altitude"], s["horizon_orbits"], m["K"], s["a_max"]) == m["physical_id"]
        assert "C" + L.stablehash({"cell": m["cell"], "split": m["split"], "index": m["index"], "draw": m["draw"]}) == m["case_id"]
        assert L.run_id(m["physical_id"], summ["run_config"]) == m["run_id"]


def legacyhashstored(stored, nanfields):
    o = dict(stored)
    for k in nanfields:
        o[k] = float("nan")
    return L.stablehash({k: v for k, v in o.items() if k not in ("software", "completed_at")}, 32)


def testlegacymismnan():
    recs = FX.first("raw/results/D3cal.jsonl", 50)
    want = {r["run_id"] for r in recs}
    led = {e["run_id"]: e["result_sha256_16"] for e in FX.ledger() if e["run_id"] in want and e.get("event") == "completed"}
    assert set(led) == want
    nnull = 0
    for r in recs:
        nulls = [k for k, v in r.items() if v is None]
        if not nulls:
            assert legacyhashstored(r, []) == led[r["run_id"]]
            continue
        nnull += 1
        assert legacyhashstored(r, []) != led[r["run_id"]]
        assert legacyhashstored(r, nulls) == led[r["run_id"]]
    assert nnull > 0


def testnormedgecases():
    assert SER.normalize({"a": np.array(1.5), "b": np.array(3), "c": np.array([True, False])}) == {"a": 1.5, "b": 3, "c": [True, False]}
    for bad in (np.array("ok"), np.array(b"ab"), np.array([object()], dtype=object), np.array(["x", "y"])):
        with pytest.raises(TypeError):
            SER.normalize({"x": bad})


def _first(cs, pred, n=1):
    return [(None, r) for r in FX.firstmatch(f"raw/results/{cs}.jsonl", pred, n)]


RECCLASS = [
    ("D0test", lambda r: r.get("eci_terminal_arrived") is True),
    ("D0test", lambda r: r.get("eci_terminal_arrived") is False and r.get("eci_status") == "ok"),
    ("D3cal", lambda r: r.get("eci_status") == "domain_exit"),
    ("D0cal", lambda r: r.get("eci_status") == "terminated"),
    ("D1cal", lambda r: r.get("worker_status") == "worker_crash"),
    ("D5test", lambda r: r.get("kp") is None),
]


@pytest.mark.parametrize("cs,pred", RECCLASS)
def testrealrecclass(cs, pred, tmp_path):
    hits = _first(cs, pred)
    assert hits, cs
    _, r = hits[0]
    assert SER.roundtripok(r)
    f = tmp_path / "x.jsonl"
    with open(f, "w", encoding="utf-8", newline="\n") as fh:
        h, ph = SER.write_record(fh, r)
    line = f.read_bytes()
    assert SER.stored_line_sha256(line) == h and SER.canonline(line)
    back = json.loads(line)
    assert SER.physics_sha256(back) == ph
    assert SER.legacycomp(back) == SER.legacycomp({**r, "feature_stage": "complete", "eci_exit_step": 3})


def nonefields2():
    import importlib.util
    spec = importlib.util.spec_from_file_location("c03", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                                      "experiments", "v31", "c03hash.py"))
    syspath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "v31")
    import sys
    if syspath not in sys.path:
        sys.path.insert(0, syspath)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.nonefields


@pytest.mark.parametrize("cs,pred", [("D0cal", lambda r: True), ("D5cal", lambda r: True),
                                     ("D1cal", lambda r: r.get("worker_status") == "worker_crash")])
def testlegacyruleexc(cs, pred):
    nonefields = nonefields2()
    recs = [r for _, r in _first(cs, pred, 40)]
    want = {r["run_id"] for r in recs}
    led = {e["run_id"]: e["result_sha256_16"] for e in FX.ledger()
           if e["run_id"] in want and e.get("event") in ("completed", "completed_reconciled")}
    assert set(led) == want
    excused = 0
    for r in recs:
        nf = nonefields(r)
        rest = [k for k, v in r.items() if v is None and k not in nf and k not in ("software", "completed_at")]
        excused += any(r.get(k, 0) is None for k in nf)
        assert legacyhashstored(r, rest) == led[r["run_id"]]
    assert excused > 0


def teststatusoutchecks(tmp_path):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "v3"))
    import campaign as CP
    recs = FX.first("raw/results/D0test.jsonl", 3)
    (tmp_path / "results").mkdir()
    with open(tmp_path / "results" / "D0test.jsonl", "w", encoding="utf-8", newline="\n") as fh, \
            open(tmp_path / "ledger.jsonl", "w", encoding="utf-8", newline="\n") as lf:
        for r in recs:
            h, ph = SER.write_record(fh, r)
            SER.write_record(lf, {"ledger_id": f"completed:{r['run_id']}", "event": "completed", "run_id": r["run_id"],
                                  "record_sha256": h, "physics_sha256": ph}, schema=None)
    rows = CP.cmdstatusout(str(tmp_path))
    assert rows[0]["ledger_hash_mismatch_or_missing"] == 0 and rows[0]["n_rows"] == 3
    data = (tmp_path / "results" / "D0test.jsonl").read_bytes().replace(b'"index":', b'"index": ', 1)
    (tmp_path / "results" / "D0test.jsonl").write_bytes(data)
    assert CP.cmdstatusout(str(tmp_path))[0]["ledger_hash_mismatch_or_missing"] == 1
