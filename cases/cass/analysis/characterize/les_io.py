"""Direct readers for MicroHH binary output of a CASS run directory (no conversion step)."""
import configparser
from datetime import datetime
from pathlib import Path

import numpy as np


def read_ini(run_dir):
    run_dir = Path(run_dir)
    ini = run_dir / "cass.ini.before_restart"
    if not ini.exists():
        ini = run_dir / "cass.ini"
    cfg = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"), strict=False)
    cfg.optionxform = str
    cfg.read(ini)
    return cfg


def lowest(a):
    """2D field of a cross-section variable: the lowest level when it has several heights."""
    for dim in ("z", "zh"):
        if dim in a.dims:
            a = a.isel({dim: 0})
    return np.asarray(a.values, dtype=float)


def eqtime_h(doy, hour_utc):
    g = 2. * np.pi / 365. * (doy - 1. + hour_utc / 24.)
    m = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                  - 0.014615 * np.cos(2. * g) - 0.040849 * np.sin(2. * g))
    return m / 60.


class Run:
    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        cfg = read_ini(self.dir)
        g = cfg["grid"]
        self.itot, self.jtot, self.ktot = int(g["itot"]), int(g["jtot"]), int(g["ktot"])
        self.xsize, self.ysize, self.zsize = float(g["xsize"]), float(g["ysize"]), float(g["zsize"])
        self.lat, self.lon = float(g["lat"]), float(g["lon"])
        self.dx, self.dy = self.xsize / self.itot, self.ysize / self.jtot
        self.t0 = datetime.strptime(cfg["time"]["datetime_utc"].strip(), "%Y-%m-%d %H:%M:%S")
        raw = np.fromfile(self.dir / "grid.0000000", dtype="<f8")
        n = (self.itot, self.itot, self.jtot, self.jtot, self.ktot, self.ktot)
        if raw.size != sum(n):
            raise ValueError(f"grid.0000000 has {raw.size} values, expected {sum(n)}")
        self.x, self.xh, self.y, self.yh, self.z, zh = np.split(raw, np.cumsum(n)[:-1])
        self.zh = np.append(zh, self.zsize)
        self.ktot_full = self.ktot

    @property
    def shape(self):
        return (self.ktot, self.jtot, self.itot)

    def field(self, var, t):
        f = self.dir / f"{var}.{int(t):07d}"
        nbytes = 8 * self.ktot * self.jtot * self.itot
        if f.stat().st_size != nbytes:
            raise ValueError(f"{f} is {f.stat().st_size} bytes, expected {nbytes}")
        return np.memmap(f, dtype="<f8", mode="r", shape=self.shape)

    def basestate(self, t):
        raw = np.fromfile(self.dir / f"thermo_basestate.{int(t):07d}", dtype="<f8")
        k = self.ktot_full
        names = ("thl0", "qt0", "thvref", "thvrefh", "pref", "prefh", "exnref", "exnrefh", "rhoref", "rhorefh")
        n = (k, k, k, k + 1, k, k + 1, k, k + 1, k, k + 1)
        if raw.size != sum(n):
            raise ValueError(f"thermo_basestate has {raw.size} values, expected {sum(n)}")
        return dict(zip(names, np.split(raw, np.cumsum(n)[:-1])))

    def restrict(self, nz):
        """Keep the lowest nz levels (the 60 s fields stop at hf_zmax)."""
        if nz < self.ktot:
            self.ktot, self.z, self.zh = nz, self.z[:nz], self.zh[:nz + 1]

    def has_hf(self, t):
        return (self.dir / f"thl_hf.{int(t):07d}").exists()

    def field_hf(self, var, t):
        """60 s field (float32, levels up to hf_zmax)."""
        f = self.dir / f"{var}_hf.{int(t):07d}"
        nz = f.stat().st_size // (4 * self.jtot * self.itot)
        return np.memmap(f, dtype="<f4", mode="r", shape=(nz, self.jtot, self.itot))

    def hf_times(self):
        return sorted(int(p.name.split(".")[-1]) for p in self.dir.glob("thl_hf.*"))

    def dump_times(self):
        return sorted(int(p.suffix[1:]) for p in self.dir.glob("thl.*") if p.name.count(".") == 1 and p.suffix[1:].isdigit())

    def basestate_at(self, t):
        """Base state at t: the hourly file, or linear in time between the two bracketing hourly files."""
        t = int(t)
        h0 = (t // 3600) * 3600
        if t == h0 or not (self.dir / f"thermo_basestate.{h0 + 3600:07d}").exists():
            return self.basestate(h0)
        a, b = self.basestate(h0), self.basestate(h0 + 3600)
        w = (t - h0) / 3600.
        return {k: (1. - w) * a[k] + w * b[k] for k in a}

    def lst(self, t):
        """Local apparent solar time [h] at simulation time t [s]."""
        h0 = self.t0.hour + self.t0.minute / 60. + self.t0.second / 3600.
        hu = h0 + np.asarray(t, dtype=float) / 3600.
        return hu + self.lon / 15. + eqtime_h(self.t0.timetuple().tm_yday, hu)
