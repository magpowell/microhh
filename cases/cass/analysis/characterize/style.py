"""Figure conventions for characterize/ (figure contract); figures are written to scratch."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.frameon": False,
                     "legend.fontsize": 8})

RT = {"1D": dict(color="C0"), "3D": dict(color="C1")}
ENV = dict(color="k")
WRITE_PDF = False


def apply(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def panel(ax, k, label=""):
    ax.set_title(f"({'abcdefghijkl'[k]}) {label}".rstrip(), loc="left", fontsize=11)


def zero_line(ax, vertical=False):
    (ax.axvline if vertical else ax.axhline)(0, color="0.75", lw=0.8, zorder=0)


def lt(hours):
    h = int(hours)
    return f"{h:02d}:{int(round((hours - h) * 60)):02d} LT"


def outdir(expt):
    d = Path(os.environ["SCRATCH"]) / "CASS_LES" / "analysis" / "characterize" / expt / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def savefig(fig, expt, name):
    f = outdir(expt) / f"{name}.png"
    fig.savefig(f, dpi=300, bbox_inches="tight")
    if WRITE_PDF:
        fig.savefig(f.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return f
