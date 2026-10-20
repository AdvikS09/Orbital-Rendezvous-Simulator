import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "results")
FIGDIR = os.path.join(RESULTS, "figures")
TABDIR = os.path.join(RESULTS, "tables")

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
GRAYLIGHT, GRAYDARK = "#c9c8c2", "#6b6a66"
AXISCOLOR = {"R": BLUE, "V": ORANGE, "H": AQUA}
AXISLABEL = {"+R": "+radial", "-R": "−radial", "+V": "+along-track", "-V": "−along-track",
             "+H": "+cross-track", "-H": "−cross-track"}
MODELSTYLE = {"cw": (BLUE, "o", "-", "CW"), "so": (ORANGE, "s", "--", "second-order"),
              "nerm": (AQUA, "^", ":", "NERM (direct LVLH)"), "eci": (VIOLET, "D", "-.", "inertial reference")}
OUTCCOLOR = {"both_succeed": GRAYLIGHT, "both_fail": GRAYDARK, "cw_only": ORANGE, "nl_only": BLUE}
OUTCLABEL = {"both_succeed": "both arrive", "both_fail": "neither arrives",
             "cw_only": "CW arrives, nonlinear does not", "nl_only": "nonlinear arrives, CW does not"}
SEQBLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def axisstyle(axis):
    c = AXISCOLOR[axis[1]]
    ls = "-" if axis[0] == "+" else "--"
    mk = "o" if axis[0] == "+" else "v"
    return c, ls, mk


def setup():
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(TABDIR, exist_ok=True)
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
    setup()
    fig.savefig(os.path.join(FIGDIR, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, f"{name}.png"), bbox_inches="tight", dpi=300)
    if data is not None:
        os.makedirs(os.path.join(FIGDIR, "data"), exist_ok=True)
        data.to_csv(os.path.join(FIGDIR, "data", f"{name}.csv"), index=False)
    plt.close(fig)


def latexescape(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("−", "$-$"), ("≤", r"$\leq$"), ("≥", r"$\geq$"), ("×", r"$\times$"),
                 ("ω", r"$\omega$"), ("ρ", r"$\rho$"), ("Δ", r"$\Delta$"), ("²", r"$^2$")):
        s = s.replace(a, b)
    return s


def writetable(df, name, caption, label=None, fmt=None, index=False):
    setup()
    df.to_csv(os.path.join(TABDIR, f"{name}.csv"), index=index)
    d = df.copy()
    fmt = fmt or {}
    for c in d.columns:
        if c in fmt:
            d[c] = d[c].map(lambda v, f=fmt[c]: (f.format(v) if (v == v and v is not None) else "--"))
    cols = list(d.columns)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{" + latexescape(caption) + "}"]
    if label:
        lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{" + "l" * len(cols) + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(latexescape(c) for c in cols) + r" \\")
    lines.append(r"\midrule")
    for _, r in d.iterrows():
        lines.append(" & ".join(latexescape(r[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TABDIR, f"{name}.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
