"""Two bookkeeping checks on the width gap.

python width_gap.py [--expt no_aerosols_zero_wind_v2]     -> width_gap.csv, arithmetic.csv, figure 19
1. Decomposition by hour: the 3D minus 1D mean width of clouds, against the excess widening 3D incumbents accumulate
   over their age at equal footprint sunlight, nearby births and size (the regression offset of widening.py times
   the cloud's age in 5-min steps). The remainder is what fewer sites leave to each cloud.
2. Arithmetic: cloud number against mean cloud area per minute; at fixed cover both runs must fall on one hyperbola.
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import xarray as xr

import widening as wd
from births import MIN_AREA, RTS
from les_io import Run
from snapshot import out_path, run_dir

HOURS = ((12., 13.), (13., 14.), (14., 15.))


def with_age(expt, d):
    """Attach the track age [min] at the sample frame to the widening rows."""
    parts = []
    for (rt, rep), g in d.groupby(["rt", "rep"]):
        with xr.open_dataset(out_path(expt, rt, rep, 0).with_name("features.nc")) as ds:
            f = ds[["frame", "track", "age"]].to_dataframe().set_index(["frame", "track"]).age
        g = g.copy()
        g["age"] = f.reindex(list(zip(g.frame, g.track))).values / 60.
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


AGE_BINS = np.array([0., 2., 5., 10., 20., 40., 1.e4])   # minutes


def decomposition(expt, d, n_boot=300, seed=0):
    """Per hour: the 3D minus 1D mean width, and the 3D excess widening per 5 min split into the part carried by the
    footprint sunlight (3D slope times the 1D to 3D difference in footprint anomaly) and the offset at equal covariates."""
    rows = []
    rng = np.random.default_rng(seed)
    for h0, h1 in HOURS:
        w = d[(d.lst >= h0) & (d.lst < h1)]
        a, b = w[w.rt == RTS[0]], w[w.rt == RTS[1]]
        dsw = b.dSW_root.mean() - a.dSW_root.mean()
        est = []
        for k in range(n_boot + 1):
            if k == 0:
                wa, wb = a, b
            else:
                ga = [g.iloc[rng.integers(0, len(g), len(g))] for _, g in a.groupby("rep")]
                gb = [g.iloc[rng.integers(0, len(g), len(g))] for _, g in b.groupby("rep")]
                wa, wb = pd.concat(ga), pd.concat(gb)
            slope = wd.fit(wb, "full")["dSW_root"]
            off = wd.fit(pd.concat([wa, wb]), "full", offset=True)["offset_3D"]
            est.append((slope * dsw, off))
        est = np.array(est)
        lo, hi = np.percentile(est[1:], [2.5, 97.5], axis=0)
        rows.append(dict(hour=f"{h0:.0f}-{h1:.0f}", W_1D=a.W.mean(), W_3D=b.W.mean(), gap=b.W.mean() - a.W.mean(), dSW_3D_minus_1D=dsw,
                         direct_per_5min=est[0, 0], direct_lo=lo[0], direct_hi=hi[0], other_per_5min=est[0, 1], other_lo=lo[1], other_hi=hi[1],
                         dW_1D=a.dW.mean(), dW_3D=b.dW.mean()))
    out = pd.DataFrame(rows)
    out["direct_share"] = out.direct_per_5min / (out.direct_per_5min + out.other_per_5min)
    return out


def by_age(d):
    """3D minus 1D mean width = shift of the age distribution along the 1D width-age relation + offset at the same age."""
    import depth_w as dw
    rows = []
    for h0, h1 in HOURS:
        w = d[(d.lst >= h0) & (d.lst < h1)].copy()
        w["ab"] = np.digitize(w.age, AGE_BINS) - 1
        w["depth"] = w.W        # reuse the split of depth_w on width
        a, b = w[w.rt == RTS[0]], w[w.rt == RTS[1]]
        r = dw.split(a, b, ["ab"])
        rows.append(dict(hour=f"{h0:.0f}-{h1:.0f}", gap=r["total"], older_sites=r["shift"], wider_at_same_age=r["offset"],
                         age_1D=a.age.median(), age_3D=b.age.median()))
    return pd.DataFrame(rows)


def arithmetic(expt, min_area=MIN_AREA):
    """Per minute: cover, number and mean area of clouds of at least min_area; hourly means per member."""
    rows = []
    for rt, rep in itertools.product(RTS, range(1, 5)):
        run = Run(run_dir(expt, rt, rep))
        with xr.open_dataset(out_path(expt, rt, rep, 0).with_name("features.nc")) as ds:
            f = ds[["frame", "area", "lst"]].to_dataframe()
        f = f[f.area >= min_area]
        g = f.groupby("frame").agg(n=("area", "size"), total=("area", "sum"), lst=("lst", "first"))
        g["cover"] = g.total / (run.xsize * run.ysize)
        g["mean_area"] = g.total / g.n
        g["rt"], g["rep"] = rt, rep
        rows.append(g.reset_index())
    m = pd.concat(rows, ignore_index=True)
    m["hour"] = np.floor(m.lst)
    h = m[(m.hour >= 12) & (m.hour <= 15)].groupby(["hour", "rt", "rep"]).agg(cover=("cover", "mean"), n=("n", "mean"), mean_area=("mean_area", "mean")).reset_index()
    e = h.groupby(["hour", "rt"]).agg(cover=("cover", "mean"), cover_lo=("cover", "min"), cover_hi=("cover", "max"), n=("n", "mean"), mean_area=("mean_area", "mean")).unstack("rt")
    out = pd.DataFrame({"cover_1D": e["cover"][RTS[0]], "cover_1D_lo": e["cover_lo"][RTS[0]], "cover_1D_hi": e["cover_hi"][RTS[0]],
                        "cover_3D": e["cover"][RTS[1]], "cover_3D_lo": e["cover_lo"][RTS[1]], "cover_3D_hi": e["cover_hi"][RTS[1]],
                        "n_1D": e["n"][RTS[0]], "n_3D": e["n"][RTS[1]], "area_1D": e["mean_area"][RTS[0]], "area_3D": e["mean_area"][RTS[1]]})
    out["n_ratio"], out["area_ratio"] = out.n_3D / out.n_1D, out.area_3D / out.area_1D
    out["n_times_area_ratio"], out["cover_ratio"] = out.n_ratio * out.area_ratio, out.cover_3D / out.cover_1D
    return m, out.reset_index()


def figure19(expt, m):
    import style as st
    from style import plt
    fig, ax = plt.subplots(figsize=(4.6, 3.8), constrained_layout=True)
    h = []
    sel = m[(m.lst >= 12.) & (m.lst < 16.)]
    for rt, lab in zip(RTS, ("1D", "3D")):
        c = sel[sel.rt == rt]
        l = ax.scatter(c.mean_area / 1.e6, c.n, s=6, alpha=0.3, lw=0, rasterized=True, label=lab, **st.RT[lab])
        h.append(l)
    a = np.array([0.1, 3.])
    dom = Run(run_dir(expt, RTS[0], 1))
    for cov in (0.05, 0.1, 0.2):
        ax.plot(a, cov * dom.xsize * dom.ysize / 1.e6 / a, color="0.6", lw=0.8, ls="--")
        ax.text(a[-1], cov * dom.xsize * dom.ysize / 1.e6 / a[-1], f" {cov:.2f}", fontsize=7, color="0.4", va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"mean cloud area [km$^2$]"); ax.set_ylabel("number of clouds")
    st.apply(ax)
    ax.legend(handles=h, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return st.savefig(fig, expt, "fig19_number_area")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    a = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    fmt = lambda v: f"{v:.3g}"
    res = out_path(a.expt, "2stream", 1, 0).parents[2]
    d = with_age(a.expt, wd.load(a.expt))
    dec = decomposition(a.expt, d)
    dec.to_csv(res / "width_gap.csv", index=False)
    print("--- width gap [m] by hour, and the 3D excess widening per 5 min split into the footprint-sunlight part and the offset at equal covariates")
    print(dec.to_string(index=False, float_format=fmt))
    ba = by_age(d)
    ba.to_csv(res / "width_gap_age.csv", index=False)
    print("\n--- the same gap split by age: older sites (shift along the 1D width-age relation) against wider at the same age")
    print(ba.to_string(index=False, float_format=fmt))
    m, ar = arithmetic(a.expt)
    ar.to_csv(res / "arithmetic.csv", index=False)
    print("\n--- arithmetic by hour: cover (member mean and range), number, mean area; the product of the ratios must equal the cover ratio")
    print(ar.to_string(index=False, float_format=fmt))
    print(figure19(a.expt, m))
