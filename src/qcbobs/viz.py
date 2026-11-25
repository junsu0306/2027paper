"""Shared figure style: the four groups always in one figure, fixed colour + marker per group (never cycled).

Colours: the dataviz reference palette's first three categorical slots (validated all-pairs: blue, orange, aqua) and a
neutral grey for the control group; each group also has its own marker so identity is never colour alone.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GROUPS = ("QCB", "normal-light", "normal-standard", "control-nopoison")
LABEL = {"QCB": "QCB", "normal-light": "normal (light)", "normal-standard": "normal (standard)",
         "control-nopoison": "no-poison control"}
COLOR = {"QCB": "#f47230", "normal-light": "#257de0", "normal-standard": "#00b894", "control-nopoison": "#666666"}
MARKER = {"QCB": "^", "normal-light": "o", "normal-standard": "s", "control-nopoison": "X"}
NATURAL = "standard-SGDlr.1"     # normals with a natural deployment-only channel: drawn with a black edge

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "axes.edgecolor": "#999999", "axes.linewidth": 0.6, "xtick.color": "#666666", "ytick.color": "#666666",
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "legend.frameon": False,
})


def legend_handles():
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=MARKER[g], color=COLOR[g], linestyle="", markersize=7, label=LABEL[g]) for g in GROUPS]
    h.append(Line2D([], [], marker="s", color=COLOR["normal-standard"], markeredgecolor="black", linestyle="",
                    markersize=7, label="standard SGD lr .1 (natural channel)"))
    return h


def strip(ax, df, xcol, ycol, order, jitter=0.12, target_col=None):
    """One column per x category; groups side by side inside each column."""
    import numpy as np
    rng = np.random.default_rng(0)
    off = {g: (i - 1.5) * 0.18 for i, g in enumerate(GROUPS)}
    for xi, xv in enumerate(order):
        sub = df[df[xcol] == xv]
        for g in GROUPS:
            s = sub[sub.group == g]
            if s.empty:
                continue
            x = xi + off[g] + rng.uniform(-jitter / 2, jitter / 2, len(s))
            edge = ["black" if c == NATURAL else "white" for c in s.cell]
            ax.scatter(x, s[ycol], s=34, marker=MARKER[g], c=COLOR[g], edgecolors=edge, linewidths=0.8, zorder=3)
            if target_col is not None and g == "QCB":
                hit = s[target_col].astype(bool).values
                ax.scatter(x[~hit], s[ycol].values[~hit], s=90, marker="o", facecolors="none",
                           edgecolors="#191919", linewidths=0.8, zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=0)
