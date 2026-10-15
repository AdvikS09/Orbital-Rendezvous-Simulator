"""Shared figure and table helpers for the extension (publication output).

Palette: categorical slots validated with the dataviz validator against a white
surface (blue #2a78d6, orange #eb6834, aqua #1baf7a; violet #4a3aa7 as an
adjacent-only 4th slot). Aqua is below 3:1 contrast on white, so every figure
also ships a CSV twin of its plotted data and uses markers or line styles as a
secondary encoding. Neutral outcome classes are gray so that disagreement
classes carry the colour. Sequential magnitude uses a single blue ramp.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "results")
FIG_DIR = os.path.join(RESULTS, "figures")
TAB_DIR = os.path.join(RESULTS, "tables")

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
GRAY_LIGHT, GRAY_DARK = "#c9c8c2", "#6b6a66"
AXIS_COLOR = {"R": BLUE, "V": ORANGE, "H": AQUA}
AXIS_LABEL = {"+R": "+radial", "-R": "−radial", "+V": "+along-track", "-V": "−along-track",
              "+H": "+cross-track", "-H": "−cross-track"}
MODEL_STYLE = {"cw": (BLUE, "o", "-", "CW"), "so": (ORANGE, "s", "--", "second-order"),
               "nerm": (AQUA, "^", ":", "NERM (direct LVLH)"), "eci": (VIOLET, "D", "-.", "inertial reference")}
OUTCOME_COLOR = {"both_succeed": GRAY_LIGHT, "both_fail": GRAY_DARK, "cw_only": ORANGE, "nl_only": BLUE}
OUTCOME_LABEL = {"both_succeed": "both arrive", "both_fail": "neither arrives",
                 "cw_only": "CW arrives, nonlinear does not", "nl_only": "nonlinear arrives, CW does not"}
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def axis_style(axis):
    """(colour, linestyle, marker) for a signed LVLH axis label like '+R'."""
    c = AXIS_COLOR[axis[1]]
    ls = "-" if axis[0] == "+" else "--"
    mk = "o" if axis[0] == "+" else "v"
    return c, ls, mk


def setup():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "legend.frameon": False,
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.linestyle": "-", "lines.linewidth": 1.6, "lines.markersize": 4.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 110,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
    })


def save(fig, name, data=None):
    """Save PDF (vector) + PNG (300 dpi) and the plotted data as CSV."""
    setup()
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), bbox_inches="tight", dpi=300)
    if data is not None:
        os.makedirs(os.path.join(FIG_DIR, "data"), exist_ok=True)
        data.to_csv(os.path.join(FIG_DIR, "data", f"{name}.csv"), index=False)
    plt.close(fig)


def _latex_escape(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("−", "$-$"), ("≤", r"$\leq$"), ("≥", r"$\geq$"), ("×", r"$\times$"),
                 ("ω", r"$\omega$"), ("ρ", r"$\rho$"), ("Δ", r"$\Delta$"), ("²", r"$^2$")):
        s = s.replace(a, b)
    return s


def write_table(df, name, caption, label=None, fmt=None, index=False):
    """CSV + LaTeX (booktabs) table. ``fmt`` maps column -> format string."""
    setup()
    df.to_csv(os.path.join(TAB_DIR, f"{name}.csv"), index=index)
    d = df.copy()
    fmt = fmt or {}
    for c in d.columns:
        if c in fmt:
            d[c] = d[c].map(lambda v, f=fmt[c]: (f.format(v) if (v == v and v is not None) else "--"))
    cols = list(d.columns)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{" + _latex_escape(caption) + "}"]
    if label:
        lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{" + "l" * len(cols) + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(_latex_escape(c) for c in cols) + r" \\")
    lines.append(r"\midrule")
    for _, r in d.iterrows():
        lines.append(" & ".join(_latex_escape(r[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TAB_DIR, f"{name}.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
