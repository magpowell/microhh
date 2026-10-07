"""Figure conventions of the CASS analysis (figure contract); figures are written to scratch."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.frameon": False,
                     "legend.fontsize": 8})

RT = {"1D": dict(color="C0"), "3D": dict(color="C1")}
ENV = dict(color="k")
RESIDUAL = dict(color="k", lw=2.0)
SW_PARTS = {"net": dict(color="C3", ls="-"), "downwelling": dict(color="C4", ls="--"), "direct": dict(color="C2", ls=":"),
            "diffuse": dict(color="C5", ls="-."), "upwelling": dict(color="C6", ls=(0, (5, 1)))}
# Columns a surface quantity is averaged over: line colour and style, and the opacity of the member band.
CONDITION = {"domain mean": dict(color="k", ls="--", alpha=0.12),
             "under cloud": dict(color="tab:red", ls="-", alpha=0.20),
             "clear sky": dict(color="0.7", ls="-", alpha=0.20)}
SPREAD = "minmax"        # band over members: "minmax" or "std"
BAND_ALPHA = 0.25
LW = 1.2
WRITE_PDF = False


def apply(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def panel(ax, k, label=""):
    ax.set_title(f"({'abcdefghijklmnopqrstuvwxyz'[k]}) {label}".rstrip(), loc="left", fontsize=11)


def zero_line(ax, vertical=False):
    (ax.axvline if vertical else ax.axhline)(0, color="0.75", lw=0.8, zorder=0)


def hour_axis(ax, xlim):
    ax.set_xlim(*xlim)
    ax.set_xlabel("local solar time [h]")


def row_label(ax, text, offset=-52, rotation=90):
    """Bold label left of a row of panels."""
    ax.annotate(text, xy=(0, 0.5), xycoords="axes fraction", xytext=(offset, 0), textcoords="offset points",
                ha="center", va="center", rotation=rotation, fontweight="bold", fontsize=10 if rotation else 11)


def spread(members):
    """Lower and upper edge of the band over members."""
    if SPREAD == "std":
        m, s = members.mean("member"), members.std("member")
        return m - s, m + s
    return members.min("member"), members.max("member")


def ensemble_line(ax, x, members, color, ls="-", lw=LW, alpha=BAND_ALPHA, label=None):
    """Member mean as a line and the spread over members as a band of the same colour (xarray, dimension member)."""
    members = members.transpose("member", ...)
    ax.plot(x, members.mean("member").values, color=color, ls=ls, lw=lw, label=label)
    lo, hi = spread(members)
    ax.fill_between(x, lo.values, hi.values, color=color, alpha=alpha, lw=0)


def condition_handles(names=CONDITION):
    return [Line2D([], [], color=CONDITION[n]["color"], ls=CONDITION[n]["ls"], lw=1.6, label=n) for n in names]


def lt(hours):
    h = int(hours)
    return f"{h:02d}:{int(round((hours - h) * 60)):02d} LT"


def outdir(expt):
    """All figures of an experiment; in_progress/ holds the ones not settled."""
    d = Path(os.environ["SCRATCH"]) / "CASS_LES" / "analysis" / "figures" / expt
    d.mkdir(parents=True, exist_ok=True)
    return d


def savefig(fig, expt, name):
    f = outdir(expt) / f"{name}.png"
    f.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f, dpi=300, bbox_inches="tight")
    if WRITE_PDF:
        fig.savefig(f.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return f
