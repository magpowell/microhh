"""Figure: mean effective cloud diameter and number of clouds per km2, 10-minute running means."""
import matplotlib.pyplot as plt

import data
import style

OUT = data.OUT_ROOT / "fig_population.png"
FIGSIZE = (8.7, 4.)
SMOOTH = 10  # minutes
PANELS = (("diameter", "mean cloud effective diameter [m]"), ("number", "clouds per km$^2$ [km$^{-2}$]"))

p1 = data.population_ensemble("1D").rolling(time=SMOOTH, center=True, min_periods=1).mean()
p3 = data.population_ensemble("3D").rolling(time=SMOOTH, center=True, min_periods=1).mean()
x = p1["lst"].values

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, layout="constrained")
for k, (ax, (var, ylabel)) in enumerate(zip(axes, PANELS)):
    style.ensemble_line(ax, x, p1[var], style.COLOR["1D"], label="1D")
    style.ensemble_line(ax, x, p3[var], style.COLOR["3D"], label="3D")
    ax.set_ylabel(ylabel)
    style.panel(ax, k)
    style.despine(ax)
    style.hour_axis(ax, data.XLIM)
axes[1].legend(loc="upper right")
style.savefig(fig, OUT)

data.report("mean effective diameter", p1["diameter"], p3["diameter"], "m")
data.report("clouds per km2", p1["number"], p3["number"], "km-2")
