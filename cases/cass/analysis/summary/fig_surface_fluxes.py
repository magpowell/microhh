"""Figure: surface shortwave received, H and LE for 1D (top row) and 3D (bottom row); domain mean from
the statistics, cloudy and clear column means from the cached cross-sections."""
import matplotlib.pyplot as plt

import data
import style

OUT = data.OUT_ROOT / "fig_surface_fluxes.png"
FIGSIZE = (18., 7.)
LW = 1.6
COLUMNS = (("sw", "sw_sfc", "surface shortwave received [W m$^{-2}$]"),
           ("H", "H", "sensible heat flux [W m$^{-2}$]"),
           ("LE", "LE", "latent heat flux [W m$^{-2}$]"))
ROWS = ("1D", "3D")

s = {run: data.stats_ensemble(run) for run in ROWS}
c = {run: data.cache_ensemble(run) for run in ROWS}

fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharex=True, layout="constrained")
for i, run in enumerate(ROWS):
    for j, (cvar, svar, ylabel) in enumerate(COLUMNS):
        ax = axes[i, j]
        for name, series in (("domain mean", s[run][svar]), ("under cloud", c[run][f"{cvar}_cloud"]),
                             ("clear", c[run][f"{cvar}_clear"])):
            st = style.CONDITION[name]
            style.ensemble_line(ax, series["lst"].values, series, st["color"], st["ls"], LW, st["alpha"])
        ax.set_ylabel(ylabel)
        style.panel(ax, 2 * j + i)
        style.despine(ax)
    style.row_label(axes[i, 0], run)
for j in range(3):
    lims = [ax.get_ylim() for ax in axes[:, j]]
    axes[0, j].set_ylim(min(lo for lo, _ in lims), max(hi for _, hi in lims))
    axes[1, j].set_ylim(axes[0, j].get_ylim())
    style.hour_axis(axes[1, j], data.XLIM)
fig.legend(handles=style.condition_handles(), loc="outside center right")
style.savefig(fig, OUT)

for cvar, svar, _ in COLUMNS:
    data.report(f"{cvar} domain mean", s["1D"][svar], s["3D"][svar], "W m-2")
    data.report(f"{cvar} domain mean (cross-sections)", c["1D"][f"{cvar}_mean"], c["3D"][f"{cvar}_mean"], "W m-2")
    data.report(f"{cvar} under cloud", c["1D"][f"{cvar}_cloud"], c["3D"][f"{cvar}_cloud"], "W m-2")
    data.report(f"{cvar} clear", c["1D"][f"{cvar}_clear"], c["3D"][f"{cvar}_clear"], "W m-2")
data.report("cloudy column fraction", c["1D"]["cloud_frac"], c["3D"]["cloud_frac"], "-")
