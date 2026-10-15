"""Table writer for the v2 extension: CSV + LaTeX (booktabs) + column definitions with units.

Every table <name> is written as
  results/v2/derived/tables/<name>.csv           data (all columns, full precision)
  results/v2/derived/tables/<name>.tex           LaTeX-ready table (booktabs; rounded for display; optionally a
                                                 subset of columns, stated under the table)
  results/v2/derived/tables/<name>_columns.csv   column, unit, definition
"""

import math
import os

import numpy as np
import pandas as pd

from v2_common import TABLES

_LATEX = (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
          ("<=", r"$\leq$"), (">=", r"$\geq$"), ("−", "$-$"), ("≤", r"$\leq$"), ("≥", r"$\geq$"), ("×", r"$\times$"),
          ("ω", r"$\omega$"), ("ρ", r"$\rho$"), ("Δ", r"$\Delta$"), ("²", r"$^2$"), ("µ", r"$\mu$"), ("±", r"$\pm$"),
          ("→", r"$\rightarrow$"))


def esc(s):
    s = str(s)
    for a, b in _LATEX:
        s = s.replace(a, b)
    return s


def fmt_val(v, f=None):
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


def write_table(df, name, caption, columns, label=None, fmt=None, max_rows_tex=None, tex_columns=None):
    """columns: dict col -> (unit, definition). Every column of df must be documented."""
    os.makedirs(TABLES, exist_ok=True)
    missing = [c for c in df.columns if c not in columns]
    if missing:
        raise ValueError(f"{name}: undocumented columns {missing}")
    df.to_csv(os.path.join(TABLES, f"{name}.csv"), index=False)
    pd.DataFrame([{"column": c, "unit": columns[c][0], "definition": columns[c][1]} for c in df.columns]
                 ).to_csv(os.path.join(TABLES, f"{name}_columns.csv"), index=False)
    fmt = fmt or {}
    tcols = [c for c in (tex_columns or list(df.columns)) if c in df.columns]
    env = "table*" if len(tcols) > 7 else "table"
    head = [esc(c) + (f" [{esc(columns[c][0])}]" if columns[c][0] not in ("", "-") else "") for c in tcols]
    body = df if max_rows_tex is None else df.head(max_rows_tex)
    lines = [rf"\begin{{{env}}}[htbp]", r"\centering", r"\scriptsize", rf"\caption{{{esc(caption)}}}",
             rf"\label{{{label or 'tab:' + name}}}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{" + "l" * len(tcols) + "}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for _, r in body.iterrows():
        lines.append(" & ".join(fmt_val(r[c], fmt.get(c)) for c in tcols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    notes = []
    if len(tcols) < len(df.columns):
        notes.append(f"Columns shown: {len(tcols)} of {len(df.columns)}; all columns in {name}.csv, definitions in {name}_columns.csv.")
    if max_rows_tex is not None and len(df) > max_rows_tex:
        notes.append(f"{len(df) - max_rows_tex} further rows in {name}.csv.")
    if notes:
        lines.append(r"\par\smallskip " + esc(" ".join(notes)))
    lines.append(rf"\end{{{env}}}")
    with open(os.path.join(TABLES, f"{name}.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
