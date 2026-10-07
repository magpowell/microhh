"""Shared matplotlib style of the summary figures (layout after analysis/base_comparison.ipynb)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

COLOR = {"1D": "C0", "3D": "C1", "3D minus 1D": "0.25"}
CONDITION = {"domain mean": dict(color="k", ls="--", alpha=0.12),
             "under cloud": dict(color="tab:red", ls="-", alpha=0.20),
             "clear sky": dict(color="0.7", ls="-", alpha=0.20)}
SPREAD = "minmax"  # band over members: "minmax" or "std"
BAND_ALPHA = 0.25
LW = 1.2
DPI = 300

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.frameon": False, "legend.fontsize": 8})


def despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def panel(ax, k):
    ax.set_title(f"({'abcdefgh'[k]})", loc="left", fontsize=11, fontweight="bold")


def hour_axis(ax, xlim):
    ax.set_xlim(*xlim)
    ax.set_xlabel("local solar time [h]")


def zero_line(ax):
    ax.axhline(0., color="0.75", lw=0.8, zorder=0)


def spread(members):
    m = members.mean("member")
    if SPREAD == "std":
        s = members.std("member")
        return m - s, m + s
    return members.min("member"), members.max("member")


def ensemble_line(ax, x, members, color, ls="-", lw=LW, alpha=BAND_ALPHA, label=None):
    """Member mean as a line and the spread over members as a band of the same colour."""
    members = members.transpose("member", ...)
    ax.plot(x, members.mean("member").values, color=color, ls=ls, lw=lw, label=label)
    lo, hi = spread(members)
    ax.fill_between(x, lo.values, hi.values, color=color, alpha=alpha, lw=0)


def run_handles(labels=("1D", "3D")):
    return [Line2D([], [], color=COLOR[lab], lw=LW, label=lab) for lab in labels]


def condition_handles(names=CONDITION):
    return [Line2D([], [], color=CONDITION[n]["color"], ls=CONDITION[n]["ls"], lw=1.6, label=n) for n in names]


def row_label(ax, text):
    ax.annotate(text, xy=(0., 0.5), xycoords="axes fraction", xytext=(-62, 0), textcoords="offset points",
                fontsize=11, fontweight="bold", ha="center", va="center")


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print("wrote", path)
