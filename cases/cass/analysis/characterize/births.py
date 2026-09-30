"""Cloud births per area and hour, and where they occur relative to the clouds already present, from the 1-min tracking.

python births.py --expt no_aerosols_zero_wind_v2 --rt 2stream --rep 1     (one row per birth: births.nc)
python births.py --summary                                                (births_rate.csv, births_location.csv, births_fate.csv)
A birth is a track that appears without a predecessor (track.py birth == "new"). Its location is compared with the
clouds of the previous minute: distance to the nearest one, and the displacement from that cloud's centroid along the
direction toward the sun. The reference is the same for the clear columns of that minute, i.e. births placed at random.
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

import masks as mk
from composite import solar_azimuth
from drift import wrap
from les_io import Run
from snapshot import out_path, run_dir

RTS = ("2stream", "raytracer")
MIN_AREA = 4.e4                              # m2, 16 cells: a cloud, and a track that becomes a cloud
RADII = (250., 500., 1000.)                  # m
HOURS = np.arange(10., 17.)                  # solar time windows [h, h + 1)
SUN_RADIUS = 1000.                           # m, births this close to a cloud are classed by side


def periodic_distance(mask, dx, dy):
    """Distance of every column to the nearest True column on a doubly periodic domain, and that column's indices."""
    ny, nx = mask.shape
    tiled = np.tile(mask, (3, 3))
    d, (jj, ii) = ndimage.distance_transform_edt(~tiled, sampling=(dy, dx), return_indices=True)
    sl = (slice(ny, 2 * ny), slice(nx, 2 * nx))
    return d[sl], jj[sl] % ny, ii[sl] % nx


def existing(path_prev, dx, dy, min_area=MIN_AREA):
    """Labels of the clouds of the previous minute that are at least min_area, their centroids and diameters."""
    lab, n = mk.label_periodic(path_prev > 0.)
    area = mk.object_areas(lab, n) * dx * dy
    keep = np.flatnonzero(area >= min_area) + 1
    lab = np.where(np.isin(lab, keep), lab, 0)
    cen = mk.periodic_centroids(lab, n, dx, dy) if n else np.zeros((0, 2))
    return lab, n, cen, 2. * np.sqrt(area / np.pi)


def geometry(lab, n, cen, path_prev, sun, x, y, dx, dy):
    """Per column: distance to the nearest cloud, its label, and the displacement from its centroid along and across the sun."""
    ny, nx = lab.shape
    if not (lab > 0).any():
        nan = np.full((ny, nx), np.nan)
        return nan, np.zeros((ny, nx), dtype=int), nan, nan
    d, jj, ii = periodic_distance(lab > 0, dx, dy)
    near = lab[jj, ii]
    cx, cy = cen[near - 1, 0], cen[near - 1, 1]
    ex = wrap(x[None, :] - cx, nx * dx)
    ey = wrap(y[:, None] - cy, ny * dy)
    along = ex * sun[0] + ey * sun[1]
    across = -ex * sun[1] + ey * sun[0]
    return d, near, along, across


def reference(d, along, clear, radii=RADII, sun_radius=SUN_RADIUS):
    """Shares of the clear columns within each radius of a cloud, and sunward among those within sun_radius."""
    out = {}
    n = max(int(clear.sum()), 1)
    for r in radii:
        out[f"ref_within_{int(r)}"] = float((clear & (d <= r)).sum() / n)
    close = clear & (d <= sun_radius)
    out["ref_sunward"] = float((close & (along > 0.)).sum() / max(int(close.sum()), 1))
    out["n_clear"] = int(clear.sum())
    return out


def analyse(expt, rt, rep):
    rd = run_dir(expt, rt, rep)
    run = Run(rd)
    res = out_path(expt, rt, rep, 0).parent
    with xr.open_dataset(res / "tracks.nc") as ds:
        tracks = ds.to_dataframe()
    with xr.open_dataset(res / "features.nc") as ds:
        feats = ds.to_dataframe()
    b = tracks[tracks.birth == "new"].copy()
    first = feats.set_index(["track", "frame"]).loc[list(zip(b.track, b.frame_first))]
    b["x"], b["y"], b["area_birth"], b["lst"] = first.x.values, first.y.values, first.area.values, first.lst.values
    with xr.open_dataset(rd / "qlqi_path.xy.nc", decode_times=False) as ds:
        time = ds["time"].values.astype(float)
        x, y = ds["x"].values.astype(float), ds["y"].values.astype(float)
        dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
        cols = {k: np.full(len(b), np.nan) for k in ("d_nearest", "along", "across", "D_nearest", "ref_sunward")}
        cols.update({f"ref_within_{int(r)}": np.full(len(b), np.nan) for r in RADII})
        cols["n_clear"] = np.zeros(len(b), dtype=int)
        cols["n_existing"] = np.zeros(len(b), dtype=int)
        frames = b.frame_first.values
        for f in np.unique(frames):
            if f == 0:
                continue
            prev = ds["qlqi_path"].isel(time=f - 1).values.astype(np.float32)
            lab, n, cen, diam = existing(prev, dx, dy)
            sun = np.array([np.sin(solar_azimuth(run, time[f])[0]), np.cos(solar_azimuth(run, time[f])[0])])
            d, near, along, across = geometry(lab, n, cen, prev, sun, x, y, dx, dy)
            ref = reference(d, along, prev <= 0.)
            rows = np.flatnonzero(frames == f)
            i = np.clip(np.round(b.x.values[rows] / dx - 0.5).astype(int) % x.size, 0, x.size - 1)
            j = np.clip(np.round(b.y.values[rows] / dy - 0.5).astype(int) % y.size, 0, y.size - 1)
            cols["d_nearest"][rows], cols["along"][rows], cols["across"][rows] = d[j, i], along[j, i], across[j, i]
            nn = near[j, i]
            cols["D_nearest"][rows] = np.where(nn > 0, diam[np.maximum(nn - 1, 0)] if diam.size else np.nan, np.nan)
            for k, v in ref.items():
                cols[k][rows] = v
            cols["n_existing"][rows] = len(np.unique(lab[lab > 0]))
    for k, v in cols.items():
        b[k] = v
    keep = ["track", "frame_first", "time_first", "lst", "x", "y", "area_birth", "area_max", "n_buoy_max", "lifetime", "merges_in",
            "splits_out", "death", "d_nearest", "along", "across", "D_nearest", "n_existing", "n_clear", "ref_sunward"] + \
           [f"ref_within_{int(r)}" for r in RADII]
    out = xr.Dataset.from_dataframe(b[keep].reset_index(drop=True))
    out.attrs.update(expt=expt, rt=rt, rep=rep, domain_km2=run.xsize * run.ysize / 1.e6, min_area=MIN_AREA, dt=float(time[1] - time[0]))
    out.to_netcdf(res / "births.nc")
    return out


def load_births(expt):
    rows, area = [], {}
    for rt, rep in itertools.product(RTS, range(1, 5)):
        with xr.open_dataset(out_path(expt, rt, rep, 0).with_name("births.nc")) as ds:
            d = ds.to_dataframe()
            area[(rt, rep)] = float(ds.attrs["domain_km2"])
        d["rt"], d["rep"] = rt, rep
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["cloud"] = d.area_max >= MIN_AREA
    d["cored"] = d.n_buoy_max > 0
    d["hour"] = np.floor(d.lst)
    return d, area


def ensemble(g, cols):
    """Member values to 1D and 3D mean, range and ratio."""
    out = {}
    for c in cols:
        s = g[c].unstack("rt")
        a, b = s[RTS[0]], s[RTS[1]]
        lvl = [l for l in a.index.names if l != "rep"]
        out[f"{c}_1D"] = a.groupby(level=lvl).mean()
        out[f"{c}_1D_lo"], out[f"{c}_1D_hi"] = a.groupby(level=lvl).min(), a.groupby(level=lvl).max()
        out[f"{c}_3D"] = b.groupby(level=lvl).mean()
        out[f"{c}_3D_lo"], out[f"{c}_3D_hi"] = b.groupby(level=lvl).min(), b.groupby(level=lvl).max()
        out[f"{c}_ratio"] = out[f"{c}_3D"] / out[f"{c}_1D"]
    return pd.DataFrame(out)


def rates(d, area):
    rows = []
    for (rt, rep), g in d.groupby(["rt", "rep"]):
        for h in HOURS:
            w = g[(g.lst >= h) & (g.lst < h + 1.)]
            rows.append(dict(rt=rt, rep=rep, hour=h, all=len(w) / area[(rt, rep)], clouds=w.cloud.sum() / area[(rt, rep)],
                             cored=w.cored.sum() / area[(rt, rep)]))
    r = pd.DataFrame(rows).set_index(["hour", "rt", "rep"])
    return ensemble(r, ["all", "clouds", "cored"])


def location(d):
    """Observed share of cloud births within each radius of an existing cloud against the random reference, by hour."""
    rows = []
    for (rt, rep), g in d.groupby(["rt", "rep"]):
        for h in HOURS:
            w = g[(g.lst >= h) & (g.lst < h + 1.) & g.cloud & np.isfinite(g.d_nearest)]
            row = dict(rt=rt, rep=rep, hour=h, n=len(w), d_median=w.d_nearest.median())
            for r in RADII:
                row[f"within_{int(r)}"] = float((w.d_nearest <= r).mean())
                row[f"expected_{int(r)}"] = float(w[f"ref_within_{int(r)}"].mean())
                row[f"excess_{int(r)}"] = row[f"within_{int(r)}"] / row[f"expected_{int(r)}"]
            close = w[w.d_nearest <= SUN_RADIUS]
            row["sunward"] = float((close.along > 0.).mean()) if len(close) else np.nan
            row["sunward_expected"] = float(close.ref_sunward.mean()) if len(close) else np.nan
            rows.append(row)
    r = pd.DataFrame(rows).set_index(["hour", "rt", "rep"])
    return ensemble(r, ["n", "d_median"] + [f"{k}_{int(r)}" for r in RADII for k in ("within", "expected", "excess")] + ["sunward", "sunward_expected"])


def fate(d):
    """What becomes of cloud births by distance from the nearest existing cloud: merge share, lifetime, largest area."""
    w = d[d.cloud & np.isfinite(d.d_nearest) & (d.lst >= 12.) & (d.lst < 15.)].copy()
    w["dist"] = pd.cut(w.d_nearest, [0., 500., 1000., 1.e9], labels=["<500", "500-1000", ">1000"], include_lowest=True)
    g = w.groupby(["dist", "rt", "rep"], observed=True).agg(n=("track", "size"), merged=("death", lambda s: float((s == "merge").mean())),
                                                            life=("lifetime", "median"), area_max=("area_max", "median"),
                                                            cored=("cored", "mean"))
    return ensemble(g, ["n", "merged", "life", "area_max", "cored"])


def summary(expt):
    d, area = load_births(expt)
    res = out_path(expt, "2stream", 1, 0).parents[2]
    r, l, f = rates(d, area), location(d), fate(d)
    r.to_csv(res / "births_rate.csv"); l.to_csv(res / "births_location.csv"); f.to_csv(res / "births_fate.csv")
    return d, r, l, f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--rt", choices=RTS)
    ap.add_argument("--rep", type=int)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        d, r, l, f = summary(a.expt)
        pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 200)
        fmt = lambda v: f"{v:.3g}"
        print("--- births per km2 per hour (all tracks; tracks that reach 16 cells; tracks that get a buoyant core): 1D, 3D, ratio")
        print(r[[c for c in r.columns if not c.endswith(("_lo", "_hi"))]].to_string(float_format=fmt))
        print("\n--- cloud births near existing clouds: observed share within r, expected for random clear columns, and their ratio")
        print(l[[c for c in l.columns if not c.endswith(("_lo", "_hi"))]].to_string(float_format=fmt))
        print("\n--- fate of cloud births born 12-15 LT by distance to the nearest existing cloud")
        print(f[[c for c in f.columns if not c.endswith(("_lo", "_hi"))]].to_string(float_format=fmt))
    else:
        ds = analyse(a.expt, a.rt, a.rep)
        d = ds.to_dataframe()
        c = d[d.area_max >= MIN_AREA]
        print(f"{a.rt} rep_{a.rep:02d}: births {len(d)}, of which clouds {len(c)}; cloud births 12-15 LT: "
              f"{int(((c.lst >= 12) & (c.lst < 15)).sum())}, median distance to an existing cloud {c.d_nearest.median():.0f} m, "
              f"within 500 m {float((c.d_nearest <= 500).mean()):.2f} (expected {c.ref_within_500.mean():.2f})", flush=True)
