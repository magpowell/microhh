"""Lifetimes of tracked clouds that never merge or split, and how far longer tracks are chains of merged clouds.

Tracks longer than about 20 min always contain merges and splits, so their length is the persistence of a cloudy
site, not a cloud lifetime. Only untouched clouds give a lifetime.

python lifetime.py --expt no_aerosols_zero_wind_v2 [--tag _core] [--min-area 1e4]
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr

from snapshot import out_path

RTS = ("2stream", "raytracer")
WINDOWS = ((10., 12.), (12., 15.))
LENGTH_BINS = np.array([1., 5., 10., 20., 40., 80., 1.e4])      # minutes
MIN_AREA = 4.e4                                                 # m2, 16 grid cells


def load(expt, rt, rep, tag="", min_area=MIN_AREA):
    with xr.open_dataset(out_path(expt, rt, rep, 0).with_name(f"tracks{tag}.nc")) as ds:
        t = ds.to_dataframe()
    t = t[(t.birth != "start") & (t.death != "end") & (t.area_max >= min_area)].copy()
    t["length"] = t.lifetime / 60.
    t["untouched"] = (t.merges_in == 0) & (t.splits_out == 0) & (t.birth == "new") & (t.death == "gone")
    return t


def chains(t, keys):
    rows = []
    for w0, w1 in WINDOWS:
        tw = t[(t.lst_first >= w0) & (t.lst_first < w1)]
        b = np.digitize(tw.length.values, LENGTH_BINS) - 1
        for i in range(LENGTH_BINS.size - 1):
            x = tw[b == i]
            rows.append(dict(**keys, window=f"{int(w0)}-{int(w1)}", length_bin=i, n=len(x), share=len(x) / max(len(tw), 1),
                             untouched=x.untouched.mean() if len(x) else np.nan,
                             merges_in=x.merges_in.mean() if len(x) else np.nan,
                             splits_out=x.splits_out.mean() if len(x) else np.nan))
    return rows


def untouched(t, keys):
    rows = []
    for w0, w1 in WINDOWS:
        tw = t[(t.lst_first >= w0) & (t.lst_first < w1)]
        x = tw[tw.untouched]
        L = x.length.values
        rows.append(dict(**keys, window=f"{int(w0)}-{int(w1)}", n=len(x), share=len(x) / max(len(tw), 1), life_mean=L.mean(),
                         life_median=np.median(L), life_p90=np.percentile(L, 90), life_p99=np.percentile(L, 99),
                         life_max=L.max(), depth_max=x.depth_max.mean(), area_max=x.area_max.mean()))
    return rows


def ensemble(df, by, cols):
    """Mean over members per radiation type, 3D minus 1D and its standard error."""
    g = df.groupby(by + ["rt"])[cols]
    m, v, n = g.mean().unstack("rt"), g.var(ddof=1).unstack("rt"), g.count().unstack("rt")
    out = {}
    for c in cols:
        a, b = m[c]["2stream"], m[c]["raytracer"]
        se = np.sqrt(v[c]["2stream"] / n[c]["2stream"] + v[c]["raytracer"] / n[c]["raytracer"])
        out[c] = pd.DataFrame({"1D": a, "3D": b, "diff": b - a, "se": se, "d_over_se": (b - a) / se})
    return pd.concat(out, axis=1)


def main(expt, tag="", min_area=MIN_AREA):
    C, U = [], []
    for rt in RTS:
        for rep in range(1, 5):
            t = load(expt, rt, rep, tag, min_area)
            C += chains(t, dict(rt=rt, rep=rep))
            U += untouched(t, dict(rt=rt, rep=rep))
    C, U = pd.DataFrame(C), pd.DataFrame(U)
    res = out_path(expt, "2stream", 1, 0).parents[2]
    C.to_csv(res / f"track_chains_long{tag}.csv", index=False)
    U.to_csv(res / f"lifetime_untouched_long{tag}.csv", index=False)
    eC = ensemble(C, ["window", "length_bin"], ["n", "share", "untouched", "merges_in", "splits_out"])
    eU = ensemble(U, ["window"], [c for c in U.columns if c not in ("rt", "rep", "window")])
    eC.to_csv(res / f"track_chains{tag}.csv")
    eU.to_csv(res / f"lifetime_untouched{tag}.csv")
    return eC, eU


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    ap.add_argument("--tag", default="")
    ap.add_argument("--min-area", type=float, default=MIN_AREA)
    a = ap.parse_args()
    eC, eU = main(a.expt, a.tag, a.min_area)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 60)
    f = lambda v: f"{v:.3g}"
    for c in ("n", "untouched", "merges_in", "splits_out"):
        print(f"\n--- tracks by length bin (minutes from {LENGTH_BINS[:-1]}): {c}")
        print(eC[c].to_string(float_format=f))
    for c in eU.columns.levels[0]:
        print(f"\n--- untouched clouds: {c}")
        print(eU[c].to_string(float_format=f))
