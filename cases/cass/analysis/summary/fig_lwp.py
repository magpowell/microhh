"""Figure: domain-mean liquid water path, 1D vs 3D, and the member-wise 3D minus 1D difference."""
import matplotlib.pyplot as plt

import data
import style

OUT = data.OUT_ROOT / "fig_lwp.png"
FIGSIZE = (9.5, 4.2)

s1, s3 = data.stats_ensemble("1D"), data.stats_ensemble("3D")
lwp1, lwp3 = 1e3 * s1["ql_path"], 1e3 * s3["ql_path"]
diff = lwp3 - lwp1  # same rndseed per rep in 1D and 3D
x = lwp1["lst"].values

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, layout="constrained")
style.ensemble_line(axes[0], x, lwp1, style.COLOR["1D"], label="1D")
style.ensemble_line(axes[0], x, lwp3, style.COLOR["3D"], label="3D")
axes[0].set_ylabel("liquid water path [g m$^{-2}$]")
axes[0].legend(loc="upper left")
style.zero_line(axes[1])
style.ensemble_line(axes[1], x, diff, style.COLOR["3D minus 1D"])
axes[1].set_ylabel("3D minus 1D liquid water path [g m$^{-2}$]")
for k, ax in enumerate(axes):
    style.panel(ax, k)
    style.despine(ax)
    style.hour_axis(ax, data.XLIM)
style.savefig(fig, OUT)

data.report("liquid water path", lwp1, lwp3, "g m-2")
data.report("3D minus 1D liquid water path", diff, diff, "g m-2")
