import json, sys, os, csv, io
A = "CW_Rendezvous_CEAS_Extension_v2"; Rp = "replay_copy"
files = ["revision_2026_10_13/generated/fresh_case_decisions.json", "revision_2026_10_13/generated/paired_A_B.json",
         "revision_2026_10_13/selected_numerical_checks.json", "revision_2026_10_13_followup/calibration_audit_plan.json",
         "revision_2026_10_13_followup/calibration_setter_checks.csv", "revision_2026_10_13_followup/calibration_setter_checks.json",
         "manuscript/Fig1_data.json", "manuscript/Fig3_data.json", "manuscript/Fig5_data.json"]
def walk(a, b, path, out, tol):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b: out.append((path + "/" + str(k), "missing")); continue
            walk(a[k], b[k], path + "/" + str(k), out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): out.append((path, f"len {len(a)} vs {len(b)}")); return
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if a != b: out.append((path, f"{a!r} vs {b!r} (absdiff {abs(a-b):.3g})"))
    elif a != b: out.append((path, f"{str(a)[:60]!r} vs {str(b)[:60]!r}"))
res = {}
for f in files:
    ra = open(os.path.join(A, f), "rb").read(); rb = open(os.path.join(Rp, f), "rb").read()
    same_norm = ra.replace(b"\r\n", b"\n") == rb.replace(b"\r\n", b"\n")
    d = []
    if not same_norm and f.endswith(".json"):
        walk(json.loads(ra), json.loads(rb), "", d, 0)
    elif not same_norm and f.endswith(".csv"):
        la = list(csv.reader(io.StringIO(ra.decode()))); lb = list(csv.reader(io.StringIO(rb.decode())))
        for i, (x, y) in enumerate(zip(la, lb)):
            for j, (p, q) in enumerate(zip(x, y)):
                if p != q: d.append((f"row{i}col{la[0][j]}", f"{p} vs {q}"))
    res[f] = {"identical_after_line_ending_normalisation": same_norm, "crlf_in_replay": b"\r\n" in rb,
              "crlf_in_archive": b"\r\n" in ra, "n_differences": len(d), "differences": d[:40]}
    print(f, same_norm, len(d)); [print("   ", x) for x in d[:12]]
json.dump(res, open(os.path.join(A, "v3_extension/raw/baseline_replay/semantic_diff.json"), "w"), indent=2)
