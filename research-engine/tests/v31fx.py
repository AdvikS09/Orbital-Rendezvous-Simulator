import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "fixtures", "v31")
REPO = os.path.dirname(HERE)
ARCH = os.environ.get("CWVALID3_V3_DIR") or os.path.join(os.path.dirname(REPO), "v3")
IDX = []


def index():
    if not IDX:
        with open(os.path.join(FX, "index.json"), encoding="utf-8") as fh:
            IDX.append(json.load(fh))
    return IDX[0]


def entry(rel):
    for e in index()["files"]:
        if e["source"] == "v3/" + rel:
            return e
    raise KeyError(f"no fixture for v3/{rel}")


def fxpath(e):
    return os.path.join(FX, *e["fixture"].split("/"))


def lines(rel):
    e = entry(rel)
    with open(fxpath(e), "rb") as fh:
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
    got = lines(rel)
    if [k for k, _ in got[:n]] != list(range(1, n + 1)):
        raise AssertionError(f"fixture for {rel} does not hold source lines 1..{n}")
    return [json.loads(b) for _, b in got[:n]]


def firstmatch(rel, pred, n=1):
    out = [r for r in records(rel) if pred(r)][:n]
    if len(out) < n:
        raise AssertionError(f"fixture for {rel} holds only {len(out)} matching records, need {n}")
    return out


def manifest(cs, n):
    return first(f"raw/manifests/{cs}.jsonl", n)


def manlines(cs):
    return [b for _, b in lines(f"raw/manifests/{cs}.jsonl")]


def planned(cs, index_):
    return next(r for r in records(f"raw/manifests/{cs}.jsonl") if r["index"] == index_)


def results(cs):
    return records(f"raw/results/{cs}.jsonl")


def resultfor(cs, run_id):
    return next(r for r in results(cs) if r["run_id"] == run_id)


def ledger():
    return records("ledger.jsonl")


def whole(rel):
    for e in index()["whole_files"]:
        if e["source"] == "v3/" + rel:
            p = os.path.join(FX, *e["fixture"].split("/"))
            with open(p, "rb") as fh:
                if hashlib.sha256(fh.read()).hexdigest() != e["sha256"]:
                    raise AssertionError(f"whole-file fixture {e['fixture']} differs from the index")
            return p
    raise KeyError(f"no whole-file fixture for v3/{rel}")


def summary(cs):
    with open(whole(f"raw/manifests/{cs}summary.json"), encoding="utf-8") as fh:
        return json.load(fh)


def freeze():
    with open(whole("raw/freeze.json"), encoding="utf-8") as fh:
        return json.load(fh)


def archfile(rel):
    p = os.path.join(ARCH, *rel.split("/"))
    return p if os.path.exists(p) else None
