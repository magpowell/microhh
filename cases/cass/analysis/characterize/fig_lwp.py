"""Figure: domain-mean liquid water path, 1D vs 3D."""
import matplotlib.pyplot as plt

import summary_data as data
import style

FIGSIZE = (5., 4.)

s1, s3 = data.stats_ensemble("1D"), data.stats_ensemble("3D")
lwp1, lwp3 = 1e3 * s1["ql_path"], 1e3 * s3["ql_path"]
x = lwp1["lst"].values

fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
style.ensemble_line(ax, x, lwp1, style.RT["1D"]["color"], label="1D")
style.ensemble_line(ax, x, lwp3, style.RT["3D"]["color"], label="3D")
ax.set_ylabel("liquid water path [g m$^{-2}$]")
ax.legend(loc="upper left")
style.apply(ax)
style.hour_axis(ax, data.XLIM)
print("wrote", style.savefig(fig, data.EXPT, "fig_lwp"))

data.report("liquid water path", lwp1, lwp3, "g m-2")
data.report("3D minus 1D liquid water path", lwp3 - lwp1, lwp3 - lwp1, "g m-2")
