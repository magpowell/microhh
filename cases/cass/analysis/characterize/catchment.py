"""Catchment of a cloud root from the low-level horizontal pressure forces of the hourly composites.

python catchment.py --expt no_aerosols_zero_wind_v3        (catchment.csv; figure 22 through figures.figure22)
Along the sun-parallel slice (r positive away from the sun), the horizontal buoyancy-pressure and dynamic-pressure
forces averaged over LOW (fractions of cloud base). The convergence point is the sign change of the buoyancy-pressure
force nearest the root centre where it turns from pointing toward the shadow to pointing toward the sun; the reach on
each side is the distance from there to where the force turns outward, or to the slice edge if it never does.
"""
import argparse

import numpy as np
import pandas as pd

import figures as fg

LOW = (0.05, 0.3)


def low_level(expt, rt, h0, h1):
    """xl, h_pb, h_pd [m s-2] averaged over LOW, the mean chord L [m] and the number of events of one hour."""
    e = fg.ens_hour(expt, rt, h0, h1).sel(dir="parallel")
    low = e.sel(znd=slice(*LOW)).mean("znd")
    return low.xl.values, low.h_pb.values, low.h_pd.values, float(e.L_mean), float(e.n_events)


def crossings(xl, f):
    """Positions where f changes sign, linearly interpolated, with the sign of the change (-1 converging, +1 diverging)."""
    s = np.sign(f)
    i = np.flatnonzero((s[:-1] != s[1:]) & (s[:-1] != 0))
    x = xl[i] - f[i] * (xl[i + 1] - xl[i]) / (f[i + 1] - f[i])
    return x, np.sign(f[i + 1] - f[i])


def reach(xl, f):
    """Convergence point and inward reach on the sun side and the shadow side; reach to the edge if no outward turn."""
    x, kind = crossings(xl, f)
    conv = x[kind < 0]
    if conv.size == 0:
        return dict(r_c=np.nan, reach_sun=np.nan, reach_shadow=np.nan, edge_sun=False, edge_shadow=False)
    r_c = conv[np.argmin(np.abs(conv))]
    div = x[kind > 0]
    left, right = div[div < r_c], div[div > r_c]
    return dict(r_c=r_c, reach_sun=r_c - (left.max() if left.size else xl[0]), reach_shadow=(right.min() if right.size else xl[-1]) - r_c,
                edge_sun=left.size == 0, edge_shadow=right.size == 0)


def table(expt):
    rows = []
    for h0, h1 in fg.HOURS:
        for rt, lab in fg.RTS:
            xl, pb, pd_, L, n = low_level(expt, rt, h0, h1)
            r = reach(xl, pb)
            sun, sha = xl < r["r_c"], xl > r["r_c"]
            rows.append(dict(hour=f"{h0:.0f}-{h1:.0f}", rt=lab, L_m=L, n_events=n, **r,
                             pb_sun_max=1.e3 * pb[sun].max(), pb_shadow_max=-1.e3 * pb[sha].min(),
                             pb_sun_edge=1.e3 * pb[0], pb_shadow_edge=-1.e3 * pb[-1],
                             pd_sun_max=1.e3 * pd_[sun].max(), pd_shadow_max=-1.e3 * pd_[sha].min()))
    d = pd.DataFrame(rows)
    d.to_csv(fg.st.outdir(expt).parent / "catchment.csv", index=False)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", default="no_aerosols_zero_wind_v3")
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    d = table(a.expt)
    print("--- low-level (%.2f-%.2f z_b) buoyancy-pressure force: convergence point and inward reach in units of L; peak and edge values in 1e-3 m s-2 (positive = toward the root)" % LOW)
    print(d.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(fg.figure22(a.expt))
