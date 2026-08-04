"""Shared matplotlib styling — validated palette, chrome, and figure export.

One place for both `data/eda.ipynb` (M2) and `src/plot_curves.py` (M7's
ablation plots) to pull colors from, so the portfolio gallery doesn't end up
with two different visual languages for figures that sit side by side.

Palette is the pre-validated default from the dataviz skill's
`references/palette.md` — hex values only, not re-derived here. Categorical
hues are used in the documented fixed order (never cycled/reassigned), and
the sequential ramp is used single-hue light->dark for magnitude, per the
skill's color formula.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Categorical — fixed order, never cycled. Slots 1-3 also clear the stricter
# all-pairs gate, so scatter/small-multiples should draw from the front.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Sequential blue ramp, light -> dark. For ordinal/funnel use, the skill caps
# the lightest step at 250 on a light surface (below that, sub-2:1 contrast).
SEQUENTIAL_BLUE = {
    100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4",
    250: "#86b6ef", 300: "#6da7ec", 350: "#5598e7",
    400: "#3987e5", 450: "#2a78d6", 500: "#256abf",
    550: "#1c5cab", 600: "#184f95", 650: "#104281", 700: "#0d366b",
}

CHROME = {
    "surface": "#fcfcfb",
    "primary_ink": "#0b0b0b",
    "secondary_ink": "#52514e",
    "muted": "#898781",
    "gridline": "#e1e0d9",
    "baseline": "#c3c2b7",
}

FONT_FAMILY = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def use_style() -> None:
    """Applies the palette as matplotlib rcParams. Call once, before plotting."""
    plt.rcParams.update({
        "figure.facecolor": CHROME["surface"],
        "axes.facecolor": CHROME["surface"],
        "savefig.facecolor": CHROME["surface"],
        "text.color": CHROME["primary_ink"],
        "axes.labelcolor": CHROME["secondary_ink"],
        "axes.edgecolor": CHROME["baseline"],
        "xtick.color": CHROME["muted"],
        "ytick.color": CHROME["muted"],
        "grid.color": CHROME["gridline"],
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": FONT_FAMILY,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.dpi": 100,
    })


def savefig(fig, path: str | Path, width_px: int = 1500) -> None:
    """Saves at a DPI chosen so the export is at least `width_px` wide —
    the M2 exit criterion is >=1200px."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_width_in = fig.get_size_inches()[0]
    dpi = max(150, int(width_px / fig_width_in) + 1)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"saved {path} ({dpi} dpi, ~{int(fig_width_in * dpi)}px wide)")
