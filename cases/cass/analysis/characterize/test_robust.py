"""Exact checks of the estimators in robust.py on constructed cloud tables."""
import numpy as np
import pandas as pd

import robust as rb


def _table(off, slope3=0.3):
    rng = np.random.default_rng(0)
    rows = []
    for rt, o, s in (("2stream", 0., 0.3), ("raytracer", off, slope3)):
        for rep in range(1, 5):
            D = rng.uniform(300., 2500., 150) * (1.2 if rt == "raytracer" else 1.)
            rows.append(pd.DataFrame(dict(rt=rt, rep=rep, t=39600, lst=14.9, D=D, depth=100. + s * D + o, w=rng.uniform(0.5, 4., 150))))
    d = pd.concat(rows, ignore_index=True)
    d["wb"] = np.digitize(d.w, rb.W_BINS) - 1
    d["db"] = np.digitize(d.D, rb.D_BINS4) - 1
    d["db8"] = np.digitize(d.D, rb.D_BINS8) - 1
    return d


def test_common_slope_fit_recovers_the_offset_exactly():
    assert abs(rb.common_slope_offset(_table(60.)) - 60.) < 1.e-9
    assert abs(rb.common_slope_offset(_table(0.))) < 1.e-9


def test_coarse_bins_leak_width_into_the_offset():
    d = _table(0.)                                   # same relation, 3D clouds 20 % wider: the true offset is zero
    a, b = d[d.rt == "2stream"], d[d.rt == "raytracer"]
    o4 = rb.dw.split(a, b, ["db"])["offset"]
    o8 = rb.dw.split(a, b, ["db8"])["offset"]
    assert 25. < o4 < 40. and abs(o8) < 5.           # measured +31.9 m with 4 bins and -1.5 m with 8 bins


def test_per_member_returns_one_row_per_pair():
    m = rb.per_member(_table(60.))
    assert len(m) == 4 and np.allclose(m.offset_fit, 60.)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
