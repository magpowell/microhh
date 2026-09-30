"""Which cloud a short-lived birth merges into: the neighbour it was born next to, or another.

python merges.py --expt no_aerosols_zero_wind_v2 --rt 2stream --rep 1     (per birth that died by merging: merges.nc)
python merges.py --summary                                                (merges_summary.csv)
Needs births.nc (births.py) and the tracker output. The absorber is the cloud of the next minute that overlaps the
birth's last footprint most; it is the neighbour when its track is the nearest cloud's track at birth, or in that
track's family.
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import xarray as xr

import masks as mk
from births import MIN_AREA, RTS, ensemble
from snapshot import out_path, run_dir
from track import overlaps

WITHIN = 5.                                  # minutes
DIST = [0., 250., 500., 1000., 1.e9]


def absorber(lab_last, own, lab_next):
    """Label in the next frame with the largest overlap with object `own` of the last frame, or 0."""
    a, b, n = overlaps(np.where(lab_last == own, own, 0), lab_next)
    return int(b[np.argmax(n)]) if n.size else 0


def analyse(expt, rt, rep):
    res = out_path(expt, rt, rep, 0).parent
    with xr.open_dataset(res / "births.nc") as ds:
        b = ds.to_dataframe()
        dt = float(ds.attrs["dt"])
    with xr.open_dataset(res / "features.nc") as ds:
        feats = ds.to_dataframe()
    with xr.open_dataset(res / "tracks.nc") as ds:
        family = ds.to_dataframe().set_index("track").family
    by_frame = {f: g.track.values for f, g in feats.groupby("frame")}
    m = b[(b.death == "merge") & np.isfinite(b.d_nearest)].copy()
    m["frame_last"] = (m.frame_first + np.round(m.lifetime / dt).astype(int) - 1).astype(int)
    with xr.open_dataset(run_dir(expt, rt, rep) / "qlqi_path.xy.nc", decode_times=False) as ds:
        nt = ds.sizes["time"]
        absorb = np.zeros(len(m), dtype=int)
        frames = m.frame_last.values
        for f in np.unique(frames):
            if f + 1 >= nt:
                continue
            lab_last, _ = mk.label_periodic(ds["qlqi_path"].isel(time=f).values > 0.)
            lab_next, _ = mk.label_periodic(ds["qlqi_path"].isel(time=f + 1).values > 0.)
            tr_last, tr_next = by_frame[f], by_frame[f + 1]
            for r in np.flatnonzero(frames == f):
                own = int(np.flatnonzero(tr_last == m.track.values[r])[0]) + 1
                k = absorber(lab_last, own, lab_next)
                absorb[r] = int(tr_next[k - 1]) if k else 0
    m["absorber"] = absorb
    m["into_neighbour"] = (m.absorber > 0) & (m.absorber == m.track_nearest)
    fam = lambda t: family.reindex(t).values
    m["into_neighbour_family"] = (m.absorber > 0) & (m.track_nearest > 0) & (fam(m.absorber) == fam(m.track_nearest))
    m["minutes"] = m.lifetime / 60.
    out = xr.Dataset.from_dataframe(m[["track", "frame_first", "lst", "area_max", "d_nearest", "D_nearest", "track_nearest", "minutes",
                                       "absorber", "into_neighbour", "into_neighbour_family"]].reset_index(drop=True))
    out.attrs.update(expt=expt, rt=rt, rep=rep)
    out.to_netcdf(res / "merges.nc")
    return out


def load(expt):
    rows = []
    for rt, rep in itertools.product(RTS, range(1, 5)):
        with xr.open_dataset(out_path(expt, rt, rep, 0).with_name("births.nc")) as ds:
            b = ds.to_dataframe()
        with xr.open_dataset(out_path(expt, rt, rep, 0).with_name("merges.nc")) as ds:
            m = ds.to_dataframe().set_index("track")
        b = b[b.area_max >= MIN_AREA].copy()
        for c in ("into_neighbour", "into_neighbour_family"):
            b[c] = m[c].reindex(b.track).fillna(False).values.astype(bool)
        b["minutes"] = b.lifetime / 60.
        b["rt"], b["rep"] = rt, rep
        rows.append(b)
    d = pd.concat(rows, ignore_index=True)
    d["dist"] = pd.cut(d.d_nearest, DIST, labels=["<250", "250-500", "500-1000", ">1000"], include_lowest=True)
    return d


def summary(expt, within=WITHIN):
    d = load(expt)
    w = d[(d.lst >= 12.) & (d.lst < 15.) & np.isfinite(d.d_nearest)]
    soon = (w.death == "merge") & (w.minutes <= within)
    g = w.assign(merge_soon=soon, neighbour_soon=soon & w.into_neighbour, family_soon=soon & w.into_neighbour_family,
                 merge_any=w.death == "merge").groupby(["dist", "rt", "rep"], observed=True).agg(
        n=("track", "size"), merge_soon=("merge_soon", "mean"), neighbour_soon=("neighbour_soon", "mean"),
        family_soon=("family_soon", "mean"), merge_any=("merge_any", "mean"), minutes=("minutes", "median"))
    out = ensemble(g, ["n", "merge_soon", "neighbour_soon", "family_soon", "merge_any", "minutes"])
    rows = []
    for h in range(12, 17):
        x = d[(d.lst >= h) & (d.lst < h + 1.) & (d.d_nearest <= 500.)]
        s = (x.death == "merge") & (x.minutes <= within)
        gg = x.assign(merge_soon=s, neighbour_soon=s & x.into_neighbour).groupby(["rt", "rep"]).agg(
            n=("track", "size"), merge_soon=("merge_soon", "mean"), neighbour_soon=("neighbour_soon", "mean"))
        e = ensemble(gg.assign(hour=h).set_index("hour", append=True).reorder_levels(["hour", "rt", "rep"]), ["n", "merge_soon", "neighbour_soon"])
        rows.append(e)
    hourly = pd.concat(rows)
    res = out_path(expt, "2stream", 1, 0).parents[2]
    out.to_csv(res / "merges_summary.csv"); hourly.to_csv(res / "merges_hourly.csv")
    return out, hourly


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--rt", choices=RTS)
    ap.add_argument("--rep", type=int)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
    fmt = lambda v: f"{v:.3g}"
    if a.summary:
        out, hourly = summary(a.expt)
        keep = lambda t: t[[c for c in t.columns if not c.endswith(("_lo", "_hi"))]]
        print(f"--- cloud births 12-15 LT by distance to the nearest cloud: share merging within {WITHIN:.0f} min (any cloud; the neighbour; its family), share merging ever, median lifetime [min]")
        print(keep(out).to_string(float_format=fmt))
        print("\n--- births within 500 m of a cloud, by hour")
        print(keep(hourly).to_string(float_format=fmt))
    else:
        o = analyse(a.expt, a.rt, a.rep).to_dataframe()
        print(f"{a.rt} rep_{a.rep:02d}: merged births {len(o)}, absorber found {int((o.absorber > 0).sum())}, into the neighbour "
              f"{int(o.into_neighbour.sum())}, into its family {int(o.into_neighbour_family.sum())}", flush=True)
