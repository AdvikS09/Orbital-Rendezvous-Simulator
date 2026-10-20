import math
import os

import numpy as np
import pandas as pd

from v2common import TABLES

LATEX = (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
         ("<=", r"$\leq$"), (">=", r"$\geq$"), ("−", "$-$"), ("≤", r"$\leq$"), ("≥", r"$\geq$"), ("×", r"$\times$"),
         ("ω", r"$\omega$"), ("ρ", r"$\rho$"), ("Δ", r"$\Delta$"), ("²", r"$^2$"), ("µ", r"$\mu$"), ("±", r"$\pm$"),
         ("→", r"$\rightarrow$"))


def esc(s):
    s = str(s)
    for a, b in LATEX:
        s = s.replace(a, b)
    return s


def fmtval(v, f=None):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if f is not None:
        try:
            return f.format(v)
        except (ValueError, TypeError):
            return esc(v)
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        a = abs(v)
        if a == 0:
            return "0"
        if a >= 1e5 or a < 1e-3:
            m, e = f"{v:.3e}".split("e")
            return f"${m}\\times10^{{{int(e)}}}$"
        return f"{v:.4g}"
    return esc(v)


def writetable(df, name, caption, columns, label=None, fmt=None, maxrowstex=None, texcol=None):
    os.makedirs(TABLES, exist_ok=True)
    missing = [c for c in df.columns if c not in columns]
    if missing:
        raise ValueError(f"{name}: undocumented columns {missing}")
    df.to_csv(os.path.join(TABLES, f"{name}.csv"), index=False)
    pd.DataFrame([{"column": c, "unit": columns[c][0], "definition": columns[c][1]} for c in df.columns]
                 ).to_csv(os.path.join(TABLES, f"{name}cols.csv"), index=False)
    fmt = fmt or {}
    tcols = [c for c in (texcol or list(df.columns)) if c in df.columns]
    env = "table*" if len(tcols) > 7 else "table"
    head = [esc(c) + (f" [{esc(columns[c][0])}]" if columns[c][0] not in ("", "-") else "") for c in tcols]
    body = df if maxrowstex is None else df.head(maxrowstex)
    lines = [rf"\begin{{{env}}}[htbp]", r"\centering", r"\scriptsize", rf"\caption{{{esc(caption)}}}",
             rf"\label{{{label or 'tab:' + name}}}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{" + "l" * len(tcols) + "}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for _, r in body.iterrows():
        lines.append(" & ".join(fmtval(r[c], fmt.get(c)) for c in tcols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    notes = []
    if len(tcols) < len(df.columns):
        notes.append(f"Columns shown: {len(tcols)} of {len(df.columns)}; all columns in {name}.csv, definitions in {name}cols.csv.")
    if maxrowstex is not None and len(df) > maxrowstex:
        notes.append(f"{len(df) - maxrowstex} further rows in {name}.csv.")
    if notes:
        lines.append(r"\par\smallskip " + esc(" ".join(notes)))
    lines.append(rf"\end{{{env}}}")
    with open(os.path.join(TABLES, f"{name}.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
