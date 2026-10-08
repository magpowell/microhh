"""Figure: time-height resolved turbulence kinetic energy (slab statistics), member mean, for 3D, 1D and 3D minus 1D
in percent of 1D. Prints hourly layer means of the energy, of its vertical and horizontal parts, and of an estimate
of the subgrid part. The model has a Smagorinsky closure and no subgrid energy, only the eddy viscosity K; its
local-equilibrium value is e_sgs = <K^2> / (C_M DELTA)^2 [1 + (C_S DELTA / (kappa (z + z0)))^2], the Deardorff relation
K = C_M l sqrt(e) inverted with the model's mixing length (wall damping), C_M = 0.12 as in the model's own energy
scheme and DALES (Heus et al. 2010), C_S = 0.23, <K^2> from the slab mean and variance of K. Good to a factor of
about two (C_M 0.094 to 0.14); a lower bound in stable layers.
Second figure: the same against height over the cloud-base height of each member."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import summary_data as data
import style

FIGSIZE = (10., 9.)
ZLIM = (0., 5000.)
LAYERS = {"subcloud 0-1400 m": (0., 1400.), "cloud layer 1700-3000 m": (1700., 3000.)}
PCT = 99                 # percentile setting the colour scale
DLIM = 60.               # percent
MASK = 0.05              # no percentage where the 1D energy is below this fraction of the colour-scale maximum
C_M, C_S, DELTA, KAPPA, Z0 = 0.12, 0.23, (50. * 50. * 25.) ** (1. / 3.), 0.4, 0.035

s1, s3 = data.stats_ensemble("1D"), data.stats_ensemble("3D")
t1, t3 = (s["tke"].sel(z=slice(*ZLIM)).mean("member") for s in (s1, s3))
x, z = t1["lst"].values, t1["z"].values
shown = (x >= data.XLIM[0]) & (x <= data.XLIM[1])
vmax = float(np.percentile(np.concatenate([t1[shown].values.ravel(), t3[shown].values.ravel()]), PCT))
diff = (100. * (t3 - t1) / t1).where(t1 > MASK * vmax)

fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True, sharey=True, layout="constrained")
pcms = []
for ax, field, label, cmap, lim in ((axes[0], t3, "3D", "magma_r", (0., vmax)), (axes[1], t1, "1D", "magma_r", (0., vmax)),
                                    (axes[2], diff, "", "RdBu_r", (-DLIM, DLIM))):
    pcms.append(ax.pcolormesh(x, z, field.values.T, cmap=cmap, vmin=lim[0], vmax=lim[1], shading="nearest", rasterized=True))
    if label:
        ax.set_title(label, fontsize=10)
    ax.set_ylabel("z [m]")
for k, ax in enumerate(axes):
    style.panel(ax, k)
    ax.tick_params(labelsize=8)
axes[0].set_ylim(*ZLIM)
style.hour_axis(axes[2], data.XLIM)
fig.colorbar(pcms[0], ax=axes[:2].tolist(), location="right", shrink=0.85, aspect=25, extend="max", label="turbulence kinetic energy [m$^2$ s$^{-2}$]")
fig.colorbar(pcms[2], ax=axes[2], location="right", shrink=0.85, aspect=25, extend="both", label="3D minus 1D [%]")
print("wrote", style.savefig(fig, data.EXPT, "in_progress/fig_tke"))

rows = []
for name, (z0, z1) in LAYERS.items():
    for h in range(10, 17):
        row = dict(layer=name, hour=h)
        for lab, s in (("1D", s1), ("3D", s3)):
            w = s.where((s["lst"] >= h) & (s["lst"] < h + 1), drop=True)
            tke = w["tke"].sel(z=slice(z0, z1)).mean(("time", "z"))
            ww = 0.5 * w["w_2"].sel(zh=slice(z0, z1)).mean(("time", "zh"))
            wall = 1. + (C_S * DELTA / (KAPPA * (w["z"] + Z0))) ** 2
            sgs = ((w["evisc"] ** 2 + w["evisc_2"]) / (C_M * DELTA) ** 2 * wall).sel(z=slice(z0, z1)).mean(("time", "z"))
            row.update({f"tke_{lab}": float(tke.mean()), f"sd_{lab}": float(tke.std(ddof=1)), f"vert_{lab}": float(ww.mean()), f"horiz_{lab}": float((tke - ww).mean()),
                        f"sgs_{lab}": float(sgs.mean())})
        row.update(tke_ratio=row["tke_3D"] / row["tke_1D"], vert_ratio=row["vert_3D"] / row["vert_1D"], horiz_ratio=row["horiz_3D"] / row["horiz_1D"],
                   sgs_share_1D=row["sgs_1D"] / (row["sgs_1D"] + row["tke_1D"]), sgs_ratio=row["sgs_3D"] / row["sgs_1D"],
                   total_ratio=(row["tke_3D"] + row["sgs_3D"]) / (row["tke_1D"] + row["sgs_1D"]))
        rows.append(row)
d = pd.DataFrame(rows)
d.to_csv(data.FEATURES_ROOT / "tke_hourly.csv", index=False)
pd.set_option("display.width", 250)
print(d[["layer", "hour", "tke_1D", "tke_3D", "tke_ratio", "sgs_1D", "sgs_3D", "sgs_share_1D", "sgs_ratio", "total_ratio"]].round(3).to_string(index=False))


# --- against height over cloud base
S_GRID = np.linspace(0.01, 2., 101)
SMOOTH = 7               # samples (35 min) of the running median of the cloud-base height
CF_MIN = 1.e-3


def scaled(s):
    """Member-mean energy on S_GRID = z / z_b(t), z_b the lowest level with a cloud fraction of CF_MIN (running median);
    also the member-mean z_b. Times without cloud are nan."""
    out, zbs = [], []
    for m in s["member"].values:
        d = s.sel(member=m)
        cf, z = d["ql_frac"].values, d["z"].values
        zb = np.where((cf >= CF_MIN).any(axis=1), z[np.argmax(cf >= CF_MIN, axis=1)], np.nan)
        zb = pd.Series(zb).rolling(SMOOTH, center=True, min_periods=3).median().values
        tke = d["tke"].values
        out.append(np.array([np.interp(S_GRID * b, z, e) if np.isfinite(b) else np.full(S_GRID.size, np.nan) for b, e in zip(zb, tke)]))
        zbs.append(zb)
    return np.nanmean(out, axis=0), np.nanmean(zbs, axis=0)


e1, zb1 = scaled(s1)
e3, zb3 = scaled(s3)
ok = shown & np.isfinite(e1[:, 0]) & np.isfinite(e3[:, 0])
vmax = float(np.nanpercentile(np.concatenate([e1[ok].ravel(), e3[ok].ravel()]), PCT))
pct = np.where(e1 > MASK * vmax, 100. * (e3 - e1) / e1, np.nan)
fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True, sharey=True, layout="constrained")
pcms = []
for ax, field, label, cmap, lim in ((axes[0], e3, "3D", "magma_r", (0., vmax)), (axes[1], e1, "1D", "magma_r", (0., vmax)), (axes[2], pct, "", "RdBu_r", (-DLIM, DLIM))):
    pcms.append(ax.pcolormesh(x, S_GRID, field.T, cmap=cmap, vmin=lim[0], vmax=lim[1], shading="nearest", rasterized=True))
    ax.axhline(1., color="0.4", lw=0.6, ls="--")
    if label:
        ax.set_title(label, fontsize=10)
    ax.set_ylabel("$z / z_b$ [-]")
for k, ax in enumerate(axes):
    style.panel(ax, k)
    ax.tick_params(labelsize=8)
axes[0].set_ylim(0., S_GRID[-1])
style.hour_axis(axes[2], (10.5, data.XLIM[1]))
fig.colorbar(pcms[0], ax=axes[:2].tolist(), location="right", shrink=0.85, aspect=25, extend="max", label="turbulence kinetic energy [m$^2$ s$^{-2}$]")
fig.colorbar(pcms[2], ax=axes[2], location="right", shrink=0.85, aspect=25, extend="both", label="3D minus 1D [%]")
print("wrote", style.savefig(fig, data.EXPT, "in_progress/fig_tke_scaled"))
for h in range(11, 17):
    k = (x >= h) & (x < h + 1)
    band = lambda e, a, b: float(np.nanmean(e[k][:, (S_GRID >= a) & (S_GRID < b)]))
    print(f"{h}-{h + 1} LT  z_b {np.nanmean(zb1[k]):.0f} / {np.nanmean(zb3[k]):.0f} m | 3D over 1D at z/z_b 0-0.3: {band(e3, 0, .3) / band(e1, 0, .3):.3f}, 0.3-0.7: {band(e3, .3, .7) / band(e1, .3, .7):.3f}, "
          f"0.7-1: {band(e3, .7, 1) / band(e1, .7, 1):.3f}, 1-1.5: {band(e3, 1, 1.5) / band(e1, 1, 1.5):.3f}, 1.5-2: {band(e3, 1.5, 2) / band(e1, 1.5, 2):.3f}")
