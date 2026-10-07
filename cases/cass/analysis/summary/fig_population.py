"""Figure: cloud cover, mean effective cloud diameter and number of clouds per km2 (the last two as 10-minute
running means)."""
import matplotlib.pyplot as plt

import data
import style

OUT = data.FIG_ROOT / "fig_population.png"
FIGSIZE = (12.5, 3.9)
SMOOTH = 10  # minutes

s1, s3 = data.stats_ensemble("1D"), data.stats_ensemble("3D")
p1 = data.population_ensemble("1D").rolling(time=SMOOTH, center=True, min_periods=1).mean()
p3 = data.population_ensemble("3D").rolling(time=SMOOTH, center=True, min_periods=1).mean()
PANELS = ((s1["ql_cover"], s3["ql_cover"], "cloud cover [-]"),
          (p1["diameter"], p3["diameter"], "mean cloud effective diameter [m]"),
          (p1["number"], p3["number"], "clouds per km$^2$ [km$^{-2}$]"))

fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, layout="constrained")
for k, (ax, (a1, a3, ylabel)) in enumerate(zip(axes, PANELS)):
    style.ensemble_line(ax, a1["lst"].values, a1, style.COLOR["1D"], label="1D")
    style.ensemble_line(ax, a3["lst"].values, a3, style.COLOR["3D"], label="3D")
    ax.set_ylabel(ylabel)
    style.panel(ax, k)
    style.despine(ax)
    style.hour_axis(ax, data.XLIM)
axes[0].set_ylim(bottom=0.)
axes[0].legend(loc="upper right")
style.savefig(fig, OUT)

data.report("cloud cover", s1["ql_cover"], s3["ql_cover"], "-")
data.report("mean effective diameter", p1["diameter"], p3["diameter"], "m")
data.report("clouds per km2", p1["number"], p3["number"], "km-2")
