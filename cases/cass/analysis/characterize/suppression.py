"""Births on open ground against the sunlight that ground received: births per area and hour as a function of the
surface shortwave anomaly averaged over the preceding LOOK minutes, on columns more than FAR from any cloud.

python suppression.py --expt no_aerosols_zero_wind_v2 --rt 2stream --rep 1     (suppression.nc: per hour and anomaly class)
python suppression.py --summary                                                (suppression_rates.csv, suppression_decomp.csv)
The anomaly is what the surface receives minus the domain mean at each minute (two-stream in 1D, ray-traced in 3D).
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from births import MIN_AREA, RTS
from les_io import Run
from snapshot import out_path, run_dir
from widening import surface_sw

FAR = 1000.                                                    # m from any cloudy column
LOOK = 10                                                      # minutes averaged before the birth
BINS = np.array([-np.inf, -200., -100., -50., -20., 0., 20., 50., np.inf])   # W m-2
HOURS = np.arange(12., 17.)
LIT = -20.                                                     # anomaly above which open ground counts as lit


def far_from_cloud(mask, dx, dy, far=FAR):
    """Columns farther than `far` from any True column, on a doubly periodic domain."""
    ny, nx = mask.shape
    d = ndimage.distance_transform_edt(~np.tile(mask, (3, 3)), sampling=(dy, dx))
    return d[ny:2 * ny, nx:2 * nx] > far


def accumulate(A, open_cols, births_ij, area_h, out, h):
    """Add the open-ground area-time [km2 h] and births of one minute to the hour row h."""
    b = np.digitize(A, BINS) - 1
    out["area_time"][h] += np.bincount(b[open_cols], minlength=BINS.size - 1) * area_h
    for key, (j, i) in births_ij.items():
        ok = open_cols[j, i]
        out[key][h] += np.bincount(b[j[ok], i[ok]], minlength=BINS.size - 1)


def cloud_mask(path, dx, dy, min_area):
    """Cloudy columns, or only those of clouds of at least min_area when it is positive."""
    m = path > 0.
    if min_area <= 0.:
        return m
    import masks as mk
    lab, n = mk.label_periodic(m)
    keep = np.flatnonzero(mk.object_areas(lab, n) * dx * dy >= min_area) + 1
    return np.isin(lab, keep)


def tag_of(far, min_area):
    return "" if (far == FAR and min_area == 0.) else f"_far{far:.0f}_min{min_area:.0f}"


def analyse(expt, rt, rep, look=LOOK, far=FAR, min_area=0.):
    rd = run_dir(expt, rt, rep)
    run = Run(rd)
    res = out_path(expt, rt, rep, 0).parent
    with xr.open_dataset(res / "births.nc") as ds:
        births = ds.to_dataframe()[["frame_first", "x", "y", "area_max"]]
    with xr.open_dataset(rd / "qlqi_path.xy.nc", decode_times=False) as ds:
        time = ds["time"].values.astype(float)
        x, y = ds["x"].values.astype(float), ds["y"].values.astype(float)
        dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
        dt = float(time[1] - time[0])
        lst = run.lst(time)
        area_h = dx * dy / 1.e6 * dt / 3600.
        nb = BINS.size - 1
        out = {k: np.zeros((HOURS.size, nb)) for k in ("area_time", "births", "cloud_births")}
        open_area = np.zeros(HOURS.size); lit_area = np.zeros(HOURS.size); minutes = np.zeros(HOURS.size)
        frames = [f for f in range(look, time.size) if HOURS[0] <= lst[f] < HOURS[-1] + 1.]
        buf = [surface_sw(rd, rt, f) for f in range(frames[0] - look, frames[0])]
        buf = [a - a.mean() for a in buf]
        bi = np.clip(np.round(births.x.values / dx - 0.5).astype(int) % x.size, 0, x.size - 1)
        bj = np.clip(np.round(births.y.values / dy - 0.5).astype(int) % y.size, 0, y.size - 1)
        for f in frames:
            A = np.mean(buf, axis=0)
            prev = cloud_mask(ds["qlqi_path"].isel(time=f - 1).values, dx, dy, min_area)
            open_cols = far_from_cloud(prev, dx, dy, far)
            h = int(np.searchsorted(HOURS, lst[f], side="right") - 1)
            sel = births.frame_first.values == f
            cloud = sel & (births.area_max.values >= MIN_AREA)
            accumulate(A, open_cols, {"births": (bj[sel], bi[sel]), "cloud_births": (bj[cloud], bi[cloud])}, area_h, out, h)
            open_area[h] += open_cols.mean(); lit_area[h] += (open_cols & (A >= LIT)).mean(); minutes[h] += 1
            a = surface_sw(rd, rt, f)
            buf = buf[1:] + [a - a.mean()]
    ds_out = xr.Dataset({k: (("hour", "bin"), v) for k, v in out.items()},
                        coords=dict(hour=HOURS, bin=np.arange(nb), bin_lo=("bin", BINS[:-1]), bin_hi=("bin", BINS[1:])))
    ds_out["open_fraction"] = ("hour", open_area / np.maximum(minutes, 1))
    ds_out["lit_open_fraction"] = ("hour", lit_area / np.maximum(minutes, 1))
    ds_out["minutes"] = ("hour", minutes)
    ds_out.attrs.update(expt=expt, rt=rt, rep=rep, far=far, min_cloud_area=min_area, look=look, lit=LIT, domain_km2=run.xsize * run.ysize / 1.e6)
    ds_out.to_netcdf(res / f"suppression{tag_of(far, min_area)}.nc")
    return ds_out


def load(expt, tag=""):
    return {(rt, rep): xr.open_dataset(out_path(expt, rt, rep, 0).with_name(f"suppression{tag}.nc")).load()
            for rt, rep in itertools.product(RTS, range(1, 5))}


def rates(runs, key="cloud_births"):
    """Birth rate per open-ground area [km-2 h-1] per hour and anomaly class: member mean and range for 1D and 3D."""
    rows = []
    for (rt, rep), ds in runs.items():
        r = ds[key].values / np.where(ds.area_time.values > 0., ds.area_time.values, np.nan)
        for h, hour in enumerate(ds.hour.values):
            for b in range(ds.sizes["bin"]):
                rows.append(dict(rt=rt, rep=rep, hour=hour, bin=b, lo=float(ds.bin_lo[b]), hi=float(ds.bin_hi[b]),
                                 area_time=float(ds.area_time[h, b]), births=float(ds[key][h, b]), rate=r[h, b]))
    d = pd.DataFrame(rows)
    g = d.groupby(["hour", "bin", "lo", "hi", "rt"])
    out = pd.DataFrame({"area_1D": g.area_time.sum().unstack("rt")[RTS[0]], "area_3D": g.area_time.sum().unstack("rt")[RTS[1]],
                        "rate_1D": g.rate.mean().unstack("rt")[RTS[0]], "rate_1D_lo": g.rate.min().unstack("rt")[RTS[0]],
                        "rate_1D_hi": g.rate.max().unstack("rt")[RTS[0]], "rate_3D": g.rate.mean().unstack("rt")[RTS[1]],
                        "rate_3D_lo": g.rate.min().unstack("rt")[RTS[1]], "rate_3D_hi": g.rate.max().unstack("rt")[RTS[1]]})
    return out.reset_index()


def decompose(runs, key="cloud_births", lit=LIT):
    """Per hour, members pooled (births and area-time summed before dividing): birth rate on lit and shaded open ground,
    shaded share of open ground, and the 3D deficit split into the shading of open ground (placement) and the lower
    rate on lit ground. The two parts sum to the deficit exactly."""
    acc = {}
    for (rt, rep), ds in runs.items():
        litb = ds.bin_lo.values >= lit
        b, at = ds[key].values, ds.area_time.values
        a = acc.setdefault(rt, dict(b=0., a=0., bl=0., al=0., bs=0., ash=0., open=0., n=0))
        a["b"] = a["b"] + b.sum(1); a["a"] = a["a"] + at.sum(1)
        a["bl"] = a["bl"] + b[:, litb].sum(1); a["al"] = a["al"] + at[:, litb].sum(1)
        a["bs"] = a["bs"] + b[:, ~litb].sum(1); a["ash"] = a["ash"] + at[:, ~litb].sum(1)
        a["open"] = a["open"] + ds.open_fraction.values; a["n"] += 1
        hours = ds.hour.values
    o = pd.DataFrame(dict(hour=hours))
    for rt, lab in zip(RTS, ("1D", "3D")):
        a = acc[rt]
        o[f"open_{lab}"] = a["open"] / a["n"]
        o[f"shaded_share_{lab}"] = a["ash"] / np.maximum(a["a"], 1e-9)
        o[f"rate_all_{lab}"] = a["b"] / np.maximum(a["a"], 1e-9)
        o[f"rate_lit_{lab}"] = a["bl"] / np.maximum(a["al"], 1e-9)
        o[f"rate_shaded_{lab}"] = a["bs"] / np.maximum(a["ash"], 1e-9)
    r1, t3 = o["rate_lit_1D"].values, acc[RTS[1]]
    o["births_3D"] = t3["b"]
    o["births_3D_if_1D_rate"] = r1 * t3["a"]
    o["deficit"] = o["births_3D_if_1D_rate"] - o["births_3D"]
    o["deficit_placement"] = (r1 - o["rate_shaded_3D"].values) * t3["ash"]
    o["deficit_lit_rate"] = (r1 - o["rate_lit_3D"].values) * t3["al"]
    o["placement_share"] = o["deficit_placement"] / o["deficit"].where(o["deficit"] > 0)
    return o


def summary(expt, tag=""):
    runs = load(expt, tag)
    res = out_path(expt, "2stream", 1, 0).parents[2]
    r, dcp = rates(runs), decompose(runs)
    r.to_csv(res / f"suppression_rates{tag}.csv", index=False); dcp.to_csv(res / f"suppression_decomp{tag}.csv", index=False)
    ra = rates(runs, "births"); ra.to_csv(res / f"suppression_rates_all{tag}.csv", index=False)
    return r, dcp, ra


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--rt", choices=RTS)
    ap.add_argument("--rep", type=int)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--far", type=float, default=FAR, help="distance from a cloud that makes ground open [m]")
    ap.add_argument("--min-cloud", type=float, default=0., help="clouds smaller than this area [m2] do not count as clouds")
    a = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
    fmt = lambda v: f"{v:.3g}"
    if a.summary:
        r, dcp, ra = summary(a.expt, tag_of(a.far, a.min_cloud))
        print("--- cloud births per km2 of open ground per hour, by hour and by the shortwave anomaly of the preceding 10 min [W m-2]")
        print(r[r.area_3D + r.area_1D > 0.][["hour", "lo", "hi", "area_1D", "area_3D", "rate_1D", "rate_1D_lo", "rate_1D_hi", "rate_3D", "rate_3D_lo", "rate_3D_hi"]].to_string(index=False, float_format=fmt))
        print(f"\n--- open ground (> {a.far:.0f} m from a cloud of at least {a.min_cloud:.0f} m2): share of the domain, shaded share (anomaly < {LIT:.0f}), rates on lit and shaded ground, and the 3D deficit split")
        print(dcp.to_string(index=False, float_format=fmt))
    else:
        ds = analyse(a.expt, a.rt, a.rep, far=a.far, min_area=a.min_cloud)
        tot = ds.cloud_births.sum("bin").values / np.maximum(ds.area_time.sum("bin").values, 1e-9)
        print(f"{a.rt} rep_{a.rep:02d}: open-ground cloud births per km2 per hour by hour {np.round(tot, 2)}, open share {np.round(ds.open_fraction.values, 2)}, "
              f"lit share of open {np.round(ds.lit_open_fraction.values / np.maximum(ds.open_fraction.values, 1e-9), 2)}", flush=True)
