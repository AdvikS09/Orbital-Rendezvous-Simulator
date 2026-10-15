"""Read-only access to the exact-line regression fixtures in tests/fixtures/v31 (v3.1.1 R04).

Every fixture line is the exact byte string of one line of an archived v3 source file (built by
experiments/v3_1_1/build_fixtures.py, never regenerated from seeds). Each read checks the line against the SHA-256
and source line number in FIXTURE_INDEX.json, so a test that uses a fixture fails loudly instead of skipping or
running on altered data. Selection rules (FIXTURE_INDEX.json 'selection') guarantee what the helpers below rely on:
first(rel, n) needs source lines 1..n, first_matching(rel, pred, n) needs every match up to the n-th in the file.
"""

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "fixtures", "v31")
REPO = os.path.dirname(HERE)
ARCHIVE_V3 = os.environ.get("CWVALID3_V3_DIR") or os.path.join(os.path.dirname(REPO), "v3_extension")
_INDEX = []


def index():
    if not _INDEX:
        with open(os.path.join(FX, "FIXTURE_INDEX.json"), encoding="utf-8") as fh:
            _INDEX.append(json.load(fh))
    return _INDEX[0]


def entry(rel):
    for e in index()["files"]:
        if e["source"] == "v3_extension/" + rel:
            return e
    raise KeyError(f"no fixture for v3_extension/{rel}")


def fixture_path(e):
    return os.path.join(FX, *e["fixture"].split("/"))


def lines(rel):
    """[(source line number, exact line bytes)] of one archived file's fixture lines, verified against the index."""
    e = entry(rel)
    with open(fixture_path(e), "rb") as fh:
        raw = fh.read()
    parts = raw.split(b"\n")
    if parts[-1] != b"":
        raise AssertionError(f"fixture {e['fixture']} does not end with LF")
    parts = parts[:-1]
    if len(parts) != len(e["lines"]):
        raise AssertionError(f"fixture {e['fixture']}: {len(parts)} lines, index lists {len(e['lines'])}")
    out = []
    for b, meta in zip(parts, e["lines"]):
        if hashlib.sha256(b).hexdigest() != meta["sha256"]:
            raise AssertionError(f"fixture {e['fixture']} line from source line {meta['line']} differs from the index")
        out.append((meta["line"], b))
    return out


def records(rel):
    return [json.loads(b) for _, b in lines(rel)]


def first(rel, n):
    """Records of source lines 1..n exactly."""
    got = lines(rel)
    if [k for k, _ in got[:n]] != list(range(1, n + 1)):
        raise AssertionError(f"fixture for {rel} does not hold source lines 1..{n}")
    return [json.loads(b) for _, b in got[:n]]


def first_matching(rel, pred, n=1):
    """First n records (in source order) satisfying pred; valid because the selection rule keeps every such match."""
    out = [r for r in records(rel) if pred(r)][:n]
    if len(out) < n:
        raise AssertionError(f"fixture for {rel} holds only {len(out)} matching records, need {n}")
    return out


def manifest(cs, n):
    return first(f"raw/manifests/{cs}.jsonl", n)


def manifest_lines(cs):
    return [b for _, b in lines(f"raw/manifests/{cs}.jsonl")]


def planned(cs, index_):
    return next(r for r in records(f"raw/manifests/{cs}.jsonl") if r["index"] == index_)


def results(cs):
    return records(f"raw/results/{cs}.jsonl")


def result_for(cs, run_id):
    return next(r for r in results(cs) if r["run_id"] == run_id)


def ledger():
    return records("RUN_LEDGER.jsonl")


def whole(rel):
    for e in index()["whole_files"]:
        if e["source"] == "v3_extension/" + rel:
            p = os.path.join(FX, *e["fixture"].split("/"))
            with open(p, "rb") as fh:
                if hashlib.sha256(fh.read()).hexdigest() != e["sha256"]:
                    raise AssertionError(f"whole-file fixture {e['fixture']} differs from the index")
            return p
    raise KeyError(f"no whole-file fixture for v3_extension/{rel}")


def summary(cs):
    with open(whole(f"raw/manifests/{cs}_summary.json"), encoding="utf-8") as fh:
        return json.load(fh)


def freeze():
    with open(whole("raw/calibration_freeze.json"), encoding="utf-8") as fh:
        return json.load(fh)


def archive_file(rel):
    """Path of the archived source file if the full archive is present (for the archive cross-check only)."""
    p = os.path.join(ARCHIVE_V3, *rel.split("/"))
    return p if os.path.exists(p) else None
