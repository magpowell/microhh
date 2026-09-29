"""Check the high-frequency dump stream of a test run, and that two runs have identical statistics.

python check_hf.py --hf <run dir with the stream on> [--same <run a> <run b>]
"""
import argparse
import configparser
from pathlib import Path

import netCDF4 as nc
import numpy as np


def ini(run):
    c = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    c.optionxform = str
    c.read(run / "cass.ini")
    return c


def check_stream(run):
    """Every expected file exists, has the right size, and equals the full file cast to float32."""
    c = ini(run)
    it, jt, kt = (c.getint("grid", k) for k in ("itot", "jtot", "ktot"))
    dz = c.getfloat("grid", "zsize") / kt
    d = c["dump"]
    t0, t1, dt = float(d["hf_starttime"]), float(d["hf_endtime"]), float(d["hf_sampletime"])
    end = float(c["time"]["endtime"])
    nk = int(min(kt, np.floor(float(d["hf_zmax"]) / dz + 0.5)))
    names = [v.strip() for v in d["hf_dumplist"].split(",")]
    want = [int(t) for t in np.arange(np.ceil(t0 / dt) * dt, min(t1, end) + 0.5, dt) if t > 0]
    out = dict(levels=nk, expected_times=len(want), missing=[], unexpected=[], wrong_size=[], compared=0, not_identical=[])
    for v in names:
        have = sorted(int(f.name.split(".")[-1]) for f in run.glob(f"{v}_hf.[0-9]*"))
        out["missing"] += [f"{v}_hf.{t:07d}" for t in want if t not in have]
        out["unexpected"] += [f"{v}_hf.{t:07d}" for t in have if t not in want]
        for t in have:
            f = run / f"{v}_hf.{t:07d}"
            if f.stat().st_size != it * jt * nk * 4:
                out["wrong_size"].append(f.name)
                continue
            full = run / f"{v}.{t:07d}"
            if full.exists() and full.stat().st_size == it * jt * kt * 8:
                a = np.fromfile(f, dtype="<f4").reshape(nk, jt, it)
                b = np.fromfile(full, dtype="<f8").reshape(kt, jt, it)[:nk].astype("<f4")
                out["compared"] += 1
                if not np.array_equal(a, b):
                    out["not_identical"].append(f.name)
    return out


def same_statistics(a, b, tmax=None):
    """Bitwise comparison of every statistics variable of two runs over their common times."""
    res = dict(files=0, variables=0, different=[])
    for fa in sorted(a.glob("cass.*.nc")):
        fb = b / fa.name
        if not fb.exists():
            continue
        res["files"] += 1
        with nc.Dataset(fa) as A, nc.Dataset(fb) as B:
            ta, tb = A["time"][:], B["time"][:]
            n = min(ta.size, tb.size)
            if tmax is not None:
                n = min(n, int((ta <= tmax).sum()))
            for g in [None] + list(A.groups):
                GA, GB = (A, B) if g is None else (A[g], B[g])
                for v in GA.variables:
                    if v not in GB.variables or "time" not in GA[v].dimensions:
                        continue
                    x, y = np.ma.filled(GA[v][:n], np.nan), np.ma.filled(GB[v][:n], np.nan)
                    res["variables"] += 1
                    if not np.array_equal(x, y, equal_nan=True):
                        res["different"].append(f"{fa.name}:{g or ''}/{v}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path)
    ap.add_argument("--same", type=Path, nargs=2)
    ap.add_argument("--tmax", type=float, default=None)
    a = ap.parse_args()
    if a.hf:
        r = check_stream(a.hf)
        print({k: (v if not isinstance(v, list) else (len(v), v[:5])) for k, v in r.items()})
    if a.same:
        r = same_statistics(a.same[0], a.same[1], a.tmax)
        print(dict(files=r["files"], variables=r["variables"], different=(len(r["different"]), r["different"][:10])))
