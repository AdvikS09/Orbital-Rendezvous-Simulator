"""Render a Markdown report to HTML and PDF next to it (local only: python-markdown + headless Microsoft Edge
print-to-PDF). The Markdown file is the source; the PDF is a convenience copy. Default source since v3.1.1: the
corrected copy v3_1_1_followup/corrected_docs/reports/CORRECTION_REPORT_v3.1.1.md (the released v3.1 report and its
PDF are kept unchanged; writing next to a source inside the v3.0 or v3.1 trees is refused).
Usage: python code/experiments/v3_1/report_pdf.py [--src <report.md>] [--title <title>] [--edge <path to msedge.exe>]
"""

import argparse
import os
import subprocess

import markdown

import v31paths as P

CSS = """body{font-family:Segoe UI,Arial,sans-serif;font-size:10.5pt;line-height:1.45;max-width:46em;margin:2em auto;color:#111}
h1{font-size:17pt;margin-bottom:.2em}h2{font-size:13.5pt;margin-top:1.4em;border-bottom:1px solid #ccc}h3{font-size:11.5pt;margin-top:1.1em}
table{border-collapse:collapse;font-size:9pt;margin:.6em 0}th,td{border:1px solid #bbb;padding:3px 6px;vertical-align:top}
code{font-family:Consolas,monospace;font-size:9pt;background:#f3f3f3;padding:0 2px}li{margin:.15em 0}
@page{size:A4;margin:16mm}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    ap.add_argument("--src", default=os.path.join(P.V311_DOCS, "reports", "CORRECTION_REPORT_v3.1.1.md"))
    ap.add_argument("--title", default="CEAS v3.1 correction report (v3.1.1 corrected copy)")
    a = ap.parse_args()
    src = os.path.abspath(a.src)
    html = P.guard_docs(src[:-3] + ".html")
    pdf = P.guard_docs(src[:-3] + ".pdf")
    body = markdown.markdown(open(src, encoding="utf-8").read(), extensions=["tables", "fenced_code"])
    with open(P.guard_docs(html), "w", encoding="utf-8") as fh:
        fh.write(f"<!doctype html><html><head><meta charset='utf-8'><title>{a.title}</title>"
                 f"<style>{CSS}</style></head><body>{body}</body></html>")
    import tempfile
    import urllib.parse
    prof = tempfile.mkdtemp(prefix="edge_profile_")      # private profile: never attach to a running browser
    url = "file:///" + urllib.parse.quote(html.replace("\\", "/"), safe="/:")
    if os.path.exists(pdf):
        os.remove(pdf)
    r = subprocess.run([a.edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={prof}",
                        f"--print-to-pdf={pdf}", url], capture_output=True, text=True, timeout=180)
    import time
    for _ in range(120):                              # the browser may return before its child process writes the file
        if os.path.exists(pdf) and os.path.getsize(pdf) > 0:
            break
        time.sleep(0.5)
    if r.returncode != 0 or not os.path.exists(pdf):
        raise SystemExit(f"PDF rendering failed ({r.returncode}): {r.stderr[-500:]}")
    print(f"{pdf} ({os.path.getsize(pdf) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
