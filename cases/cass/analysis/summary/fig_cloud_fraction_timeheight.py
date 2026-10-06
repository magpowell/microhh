"""Figure: time-height cloud fraction profile, member mean, for 3D, 1D and 3D minus 1D."""
import numpy as np
import matplotlib.pyplot as plt

import data
import style

OUT = data.OUT_ROOT / "fig_cloud_fraction_timeheight.png"
FIGSIZE = (10., 9.)
ZLIM = (500., 5000.)
CMAP, CMAP_DIFF = "Blues", "RdBu_r"
PCT, PCT_DIFF = 99, 95  # percentiles setting the colour scales

s1, s3 = data.stats_ensemble("1D"), data.stats_ensemble("3D")
cf1 = s1["ql_frac"].sel(z=slice(*ZLIM)).mean("member")
cf3 = s3["ql_frac"].sel(z=slice(*ZLIM)).mean("member")
diff = cf3 - cf1
x, z = cf1["lst"].values, cf1["z"].values
shown = (x >= data.XLIM[0]) & (x <= data.XLIM[1])
pos = np.concatenate([v[shown].values[v[shown].values > 0] for v in (cf1, cf3)])
vmax = float(np.percentile(pos, PCT)) if pos.size else 1e-3
dmax = float(np.percentile(np.abs(diff[shown].values), PCT_DIFF)) or 1e-3

fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True, sharey=True, layout="constrained")
pcms = []
for ax, field, label, cmap, lim in ((axes[0], cf3, "3D", CMAP, (0., vmax)),
                                    (axes[1], cf1, "1D", CMAP, (0., vmax)),
                                    (axes[2], diff, "3D minus 1D", CMAP_DIFF, (-dmax, dmax))):
    pcms.append(ax.pcolormesh(x, z, field.values.T, cmap=cmap, vmin=lim[0], vmax=lim[1],
                              shading="nearest", rasterized=True))
    ax.set_title(label, fontsize=10)
    ax.set_ylabel("z [m]")
for k, ax in enumerate(axes):
    style.panel(ax, k)
    ax.tick_params(labelsize=8)
axes[0].set_ylim(*ZLIM)
style.hour_axis(axes[2], data.XLIM)
fig.colorbar(pcms[0], ax=axes[:2].tolist(), location="right", shrink=0.85, aspect=25, label="cloud fraction [-]")
fig.colorbar(pcms[2], ax=axes[2], location="right", shrink=0.85, aspect=25, label="3D minus 1D cloud fraction [-]")
style.savefig(fig, OUT)

data.report("layer-mean cloud fraction 0.5-5 km", s1["ql_frac"].sel(z=slice(*ZLIM)).mean("z"),
            s3["ql_frac"].sel(z=slice(*ZLIM)).mean("z"), "-")
for label, prof in (("1D", data.window_mean(cf1)), ("3D", data.window_mean(cf3))):
    print(f"window-mean profile peak {label}: {float(prof.max()):.3f} at z = {int(np.rint(float(z[prof.argmax()])))} m")
