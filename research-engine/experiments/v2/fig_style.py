"""Figure style and output for the v2 extension.

Journal sizing (Springer/CEAS Space Journal style): single column 84 mm, double column 174 mm;
lettering 7-8 pt at final size. Fonts are embedded as TrueType (Type 42) in PDF and EPS.
Every figure is written as results/v2/derived/figures/FigN.pdf, FigN.eps, FigN.png (300 dpi
preview) and FigN_data.csv (the exact plotted values). Categories are separated by marker
shape and line style as well as colour (colourblind-safe palette), so no figure relies on
colour alone. No transparency is used (EPS has none).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from v2_common import FIGS  # noqa: E402

MM = 1 / 25.4
SINGLE = 84 * MM
DOUBLE = 174 * MM
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
GRAY = "#9a9993"
# estimator / method encodings: colour + marker + line style
EST_STYLE = {"d1": (BLUE, "o", "-", "A: full D1"), "d1u": (ORANGE, "s", "--", "B: unsaturated Jacobian"),
             "cres": (AQUA, "^", ":", "C: residual only"), "sep": (VIOLET, "D", "-.", "separation"),
             "sat": (GRAY, "v", (0, (1, 1)), "saturated fraction")}
REGIME_STYLE = {"unsaturated": (AQUA, "o", "unsaturated"), "saturated": (BLUE, "^", "saturated"),
                "near_boundary": (ORANGE, "s", "near arrival boundary")}


def setup():
    os.makedirs(FIGS, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "legend.frameon": False,
        "axes.edgecolor": "#8a897f", "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
        "lines.linewidth": 1.0, "lines.markersize": 3.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
        "savefig.dpi": 300, "figure.dpi": 100, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })


def save(fig, name, data=None):
    setup()
    for ext in ("pdf", "eps"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight", format=ext)
    fig.savefig(os.path.join(FIGS, f"{name}.png"), bbox_inches="tight", dpi=300)
    if data is not None:
        data.to_csv(os.path.join(FIGS, f"{name}_data.csv"), index=False)
    plt.close(fig)
