import hashlib
import json
import os

import pytest

from cwvalid3 import ledger as L

from . import v31fx as FX


def testfxfilesmatch():
    idx = FX.index()
    assert idx["files"] and idx["whole_files"]
    for e in idx["files"]:
        with open(FX.fxpath(e), "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == e["fixture_sha256"], e["fixture"]
        got = FX.lines(e["source"][len("v3/"):])
        assert [k for k, _ in got] == sorted(k for k, _ in got) and len(got) == e["n_lines"] > 0
        assert all(1 <= k <= e["source_lines"] for k, _ in got)
    for e in idx["whole_files"]:
        FX.whole(e["source"][len("v3/"):])


def testfxidentparent():
    for e in FX.index()["files"]:
        rel = e["source"][len("v3/"):]
        recs = FX.records(rel)
        for meta, r in zip(e["lines"], recs):
            for k in ("index", "case_id", "physical_id", "run_id"):
                if k in meta:
                    assert meta[k] == r.get(k)
        if rel.startswith("raw/manifests/"):
            assert e["manifest_sha256_matches_summary"] is True
            summ = FX.summary(os.path.basename(rel)[:-len(".jsonl")])
            assert summ["manifest_sha256"] == e["source_sha256"] == e["parent_manifest_sha256_in_summary"]
            for m in recs:
                assert L.run_id(m["physical_id"], summ["run_config"]) == m["run_id"]
        if rel.startswith("raw/results/"):
            assert e["manifest_of_these_records"]["source"] == "v3/raw/manifests/" + os.path.basename(rel)


def testfxscoverfix():
    assert [m["index"] for m in FX.manifest("D4test", 2)] == [0, 1]
    assert FX.manifest("D4test", 1)[0]["case_id"] == "Caf93a5f0b849d73a"
    m56 = FX.planned("D4dev", 56)
    assert m56["case_id"] == "Cbfad11507250fd59" and FX.resultfor("D4dev", m56["run_id"])["eci_status"] == "domain_exit"
    assert FX.firstmatch("v31/raw/results/heldexit31.jsonl",
                         lambda r: r["replay"]["of_run_id"] == m56["run_id"])
    for cs in ("D4test",):
        for m in FX.manifest(cs, 2):
            FX.resultfor(cs, m["run_id"])
    assert sum(r.get("worker_status") == "worker_crash" for r in FX.results("D1cal")) == 43
    assert FX.freeze()["selection"]["D4|A_H|1m"]["threshold"] == 0.7998103587270196


def testfxsmatchfull():
    probe = FX.archfile("raw/manifests/D4test.jsonl")
    if probe is None:
        pytest.skip("full v3 archive not present; fixture integrity is checked against index.json above")
    for e in FX.index()["files"]:
        rel = e["source"][len("v3/"):]
        with open(FX.archfile(rel), "rb") as fh:
            raw = fh.read()
        assert hashlib.sha256(raw).hexdigest() == e["source_sha256"], rel
        s = raw.split(b"\n")
        for k, b in FX.lines(rel):
            assert s[k - 1] == b, (rel, k)
    for e in FX.index()["whole_files"]:
        with open(FX.archfile(e["source"][len("v3/"):]), "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == e["sha256"]
    assert json.loads(open(FX.archfile("raw/manifests/D4testsummary.json"), encoding="utf-8").read())["n"] == 2500
