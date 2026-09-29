"""Spatial clustering of cloud centroids on the doubly periodic domain: nearest-neighbour distance and I_org.

I_org (Tompkins and Semie 2017) is the area under the observed nearest-neighbour distribution plotted against the
one for random points, 1 - exp(-lambda pi r^2). For a sample this equals mean(exp(-lambda pi d_i^2)):
0.5 random, above 0.5 clustered, below 0.5 regular.
Two references are computed: random points, and randomly placed non-overlapping disks with the observed diameters
(centroids of finite objects cannot be closer than their radii allow, which lowers I_org).

python cluster.py --expt no_aerosols_zero_wind_v2
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from snapshot import out_path

RTS = ("2stream", "raytracer")
TIMES = (32400, 36000, 39600)
MIN_D = (0., 500.)
N_NULL = 50


def nn_distance(x, y, lx, ly):
    p = np.column_stack([np.mod(x, lx), np.mod(y, ly)])
    return cKDTree(p, boxsize=[lx, ly]).query(p, k=2)[0][:, 1]


def iorg(d, lx, ly):
    return float(np.mean(np.exp(-d.size / (lx * ly) * np.pi * d**2)))


def random_points(n, lx, ly, rng):
    return rng.uniform(0., lx, n), rng.uniform(0., ly, n)


def random_disks(D, lx, ly, rng, max_tries=20000):
    """Random sequential placement of non-overlapping disks, largest first. Returns x, y in the order of D."""
    order = np.argsort(D)[::-1]
    x, y = np.empty(D.size), np.empty(D.size)
    px, py, pr = np.empty(D.size), np.empty(D.size), np.empty(D.size)
    for m, i in enumerate(order):
        for _ in range(max_tries):
            cx, cy = rng.uniform(0., lx), rng.uniform(0., ly)
            dx = np.abs(px[:m] - cx); dx = np.minimum(dx, lx - dx)
            dy = np.abs(py[:m] - cy); dy = np.minimum(dy, ly - dy)
            if m == 0 or (dx**2 + dy**2 >= (pr[:m] + 0.5 * D[i])**2).all():
                break
        else:
            raise RuntimeError("could not place disk")
        px[m], py[m], pr[m] = cx, cy, 0.5 * D[i]
        x[i], y[i] = cx, cy
    return x, y


def measures(x, y, D, lx, ly, rng, n_null=N_NULL):
    d = nn_distance(x, y, lx, ly)
    lam = d.size / (lx * ly)
    null_p = [iorg(nn_distance(*random_points(d.size, lx, ly, rng), lx, ly), lx, ly) for _ in range(n_null)]
    null_d = [iorg(nn_distance(*random_disks(D, lx, ly, rng), lx, ly), lx, ly) for _ in range(n_null)]
    return dict(n=d.size, nn_mean=d.mean(), nn_median=np.median(d), nn_over_random=d.mean() / (0.5 / np.sqrt(lam)),
                iorg=iorg(d, lx, ly), iorg_points=np.mean(null_p), iorg_disks=np.mean(null_d),
                iorg_disks_sd=np.std(null_d, ddof=1), iorg_minus_disks=iorg(d, lx, ly) - np.mean(null_d))


def ensemble(df, by, cols):
    g = df.groupby(by + ["rt"])[cols]
    m, v, n = g.mean().unstack("rt"), g.var(ddof=1).unstack("rt"), g.count().unstack("rt")
    out = {}
    for c in cols:
        a, b = m[c][RTS[0]], m[c][RTS[1]]
        se = np.sqrt(v[c][RTS[0]] / n[c][RTS[0]] + v[c][RTS[1]] / n[c][RTS[1]])
        out[c] = pd.DataFrame({"1D": a, "3D": b, "diff": b - a, "se": se, "d_over_se": (b - a) / se})
    return pd.concat(out, axis=1)


def main(expt, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t, rt, rep in itertools.product(TIMES, RTS, range(1, 5)):
        with xr.open_dataset(out_path(expt, rt, rep, t).with_name(f"objects_{t:07d}.nc")) as ds:
            lx, ly = ds.sizes["x"] * ds.attrs["dx"], ds.sizes["y"] * ds.attrs["dy"]
            for kind, dmin in itertools.product(("obj", "sub"), MIN_D):
                x, y, D = (ds[f"{kind}_{v}"].values for v in ("x", "y", "D"))
                k = D >= dmin
                rows.append(dict(t=t, lst=float(ds.attrs["lst_solar"]), rt=rt, rep=rep, kind=kind, min_D=dmin,
                                 **measures(x[k], y[k], D[k], lx, ly, rng)))
    d = pd.DataFrame(rows)
    res = out_path(expt, "2stream", 1, 0).parents[2]
    d.to_csv(res / "cluster_long.csv", index=False)
    e = ensemble(d, ["kind", "min_D", "t"], ["n", "nn_mean", "nn_over_random", "iorg", "iorg_points", "iorg_disks", "iorg_minus_disks"])
    e.to_csv(res / "cluster.csv")
    return d, e


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v2")
    a = ap.parse_args()
    d, e = main(a.expt)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    for c in ("n", "nn_mean", "nn_over_random", "iorg", "iorg_points", "iorg_disks", "iorg_minus_disks"):
        print(f"\n--- {c}")
        print(e[c].to_string(float_format=lambda v: f"{v:.3g}"))
