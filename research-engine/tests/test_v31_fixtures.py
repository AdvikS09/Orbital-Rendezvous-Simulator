"""v3.1.1 R04: the exact-line regression fixtures are intact, carry their provenance, and (when the full archive is
present) are byte-identical to the archived source lines they were copied from."""

import hashlib
import json
import os

import pytest

from cwvalid3 import ledger as L

from . import v31fx as FX


def test_fixture_files_match_their_index():
    idx = FX.index()
    assert idx["files"] and idx["whole_files"]
    for e in idx["files"]:
        with open(FX.fixture_path(e), "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == e["fixture_sha256"], e["fixture"]
        got = FX.lines(e["source"][len("v3_extension/"):])          # checks every line hash and the line count
        assert [k for k, _ in got] == sorted(k for k, _ in got) and len(got) == e["n_lines"] > 0
        assert all(1 <= k <= e["source_lines"] for k, _ in got)
    for e in idx["whole_files"]:
        FX.whole(e["source"][len("v3_extension/"):])                 # checks the SHA-256


def test_fixture_identifiers_and_parent_manifests():
    """Manifest fixture lines carry their ids, the run configuration and the parent-manifest hash; result fixture lines
    name the manifest of their cell/split and belong to it."""
    for e in FX.index()["files"]:
        rel = e["source"][len("v3_extension/"):]
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
            assert e["manifest_of_these_records"]["source"] == "v3_extension/raw/manifests/" + os.path.basename(rel)


def test_fixtures_cover_the_correction_paths():
    """The cases the failure-path and exit regressions need are present (they must not skip)."""
    assert [m["index"] for m in FX.manifest("D4_test", 2)] == [0, 1]
    assert FX.manifest("D4_test", 1)[0]["case_id"] == "Caf93a5f0b849d73a"          # R01 counterexample
    m56 = FX.planned("D4_dev", 56)
    assert m56["case_id"] == "Cbfad11507250fd59" and FX.result_for("D4_dev", m56["run_id"])["eci_status"] == "domain_exit"
    assert FX.first_matching("v3_1_correction/raw/results/held_exit_replay_v31.jsonl",
                             lambda r: r["replay"]["of_run_id"] == m56["run_id"])
    for cs in ("D4_test",):
        for m in FX.manifest(cs, 2):
            FX.result_for(cs, m["run_id"])
    assert sum(r.get("worker_status") == "worker_crash" for r in FX.results("D1_cal")) == 43
    assert FX.freeze()["selection"]["D4|A_H|1m"]["threshold"] == 0.7998103587270196


def test_fixtures_match_full_archive():
    """Cross-check against the full archive (only possible where it is present; the hash checks above always run)."""
    probe = FX.archive_file("raw/manifests/D4_test.jsonl")
    if probe is None:
        pytest.skip("full v3 archive not present; fixture integrity is checked against FIXTURE_INDEX.json above")
    for e in FX.index()["files"]:
        rel = e["source"][len("v3_extension/"):]
        with open(FX.archive_file(rel), "rb") as fh:
            raw = fh.read()
        assert hashlib.sha256(raw).hexdigest() == e["source_sha256"], rel
        src = raw.split(b"\n")
        for k, b in FX.lines(rel):
            assert src[k - 1] == b, (rel, k)
    for e in FX.index()["whole_files"]:
        with open(FX.archive_file(e["source"][len("v3_extension/"):]), "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == e["sha256"]
    assert json.loads(open(FX.archive_file("raw/manifests/D4_test_summary.json"), encoding="utf-8").read())["n"] == 2500
