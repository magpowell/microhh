"""Cache step: per member, per-minute means of surface SW, H and LE over all, cloudy and clear columns.

Usage: python cache_surface.py            all eight members in parallel (run as a Slurm job)
       python cache_surface.py --test     one member, a few frames, nothing written
"""
import sys
from multiprocessing import Pool

import numpy as np
import xarray as xr

import data

BATCH = 20  # frames per read
KEYS = [f"{v}_{part}" for v in ("sw", "H", "LE") for part in ("mean", "cloud", "clear")] + ["cloud_frac"]
TEST = ("raytracer", 1, slice(400, 440))


def masked_mean(f, m):
    n = m.sum(axis=(1, 2))
    return np.where(n > 0, (f * m).sum(axis=(1, 2)) / np.maximum(n, 1), np.nan)


def member_series(rt, rep, frames=None):
    rd = data.run_dir(rt, rep)
    times = data.xy_times(rd)
    frames = frames or slice(0, times.size)
    times = times[frames]
    out = {k: np.full(times.size, np.nan) for k in KEYS}
    for i0 in range(0, times.size, BATCH):
        fr = slice(frames.start + i0, min(frames.start + i0 + BATCH, frames.stop))
        dst = slice(i0, i0 + fr.stop - fr.start)
        cloud = data.cloud_mask(rd, fr)
        fields = dict(zip(("sw", "H", "LE"), (data.surface_sw(rd, fr), *data.surface_fluxes(rd, fr))))
        for name, f in fields.items():
            out[f"{name}_mean"][dst] = f.mean(axis=(1, 2))
            out[f"{name}_cloud"][dst] = masked_mean(f, cloud)
            out[f"{name}_clear"][dst] = masked_mean(f, ~cloud)
        out["cloud_frac"][dst] = cloud.mean(axis=(1, 2))
    ds = xr.Dataset({k: ("time", v) for k, v in out.items()},
                    coords={"time": times, "lst": ("time", data.Run(rd).lst(times))})
    ds.attrs = dict(expt=data.EXPT, rt=rt, rep=rep, rho=data.RHO, cp=data.CP, lv=data.LV,
                    cloud="qlqi_path > 0", units="W m-2")
    return ds


def write_member(args):
    rt, rep = args
    path = data.cache_file(rt, rep)
    path.parent.mkdir(parents=True, exist_ok=True)
    member_series(rt, rep).to_netcdf(path)
    return str(path)


if __name__ == "__main__":
    if "--test" in sys.argv:
        ds = member_series(*TEST)
        print(ds.to_dataframe().round(2).to_string())
    else:
        todo = data.members("1D") + data.members("3D")
        with Pool(len(todo)) as pool:
            for path in pool.imap_unordered(write_member, todo):
                print("wrote", path, flush=True)
