"""Figure: domain cloud cover, 1D vs 3D."""
import matplotlib.pyplot as plt

import data
import style

OUT = data.OUT_ROOT / "fig_cloud_cover.png"
FIGSIZE = (6., 4.)

s1, s3 = data.stats_ensemble("1D"), data.stats_ensemble("3D")
cc1, cc3 = s1["ql_cover"], s3["ql_cover"]
x = cc1["lst"].values

fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
style.ensemble_line(ax, x, cc1, style.COLOR["1D"], label="1D")
style.ensemble_line(ax, x, cc3, style.COLOR["3D"], label="3D")
ax.set_ylabel("cloud cover [-]")
ax.set_ylim(bottom=0.)
ax.legend(loc="upper right")
style.despine(ax)
style.hour_axis(ax, data.XLIM)
style.savefig(fig, OUT)

data.report("cloud cover", cc1, cc3, "-")
