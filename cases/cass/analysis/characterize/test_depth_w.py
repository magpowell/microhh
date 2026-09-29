"""Exact checks of the split in depth_w.py on constructed cloud tables."""
import numpy as np
import pandas as pd

import depth_w as dw


def _clouds(rt, w, depth):
    d = pd.DataFrame(dict(rt=rt, rep=np.arange(len(w)) % 4 + 1, w=w, depth=depth))
    d["wb"] = np.digitize(d.w, dw.W_BINS) - 1
    return d


def test_pure_shift_along_a_common_relation():
    w1 = np.repeat([0.5, 1.5, 2.5], [40, 40, 40]); w3 = np.repeat([0.5, 1.5, 2.5], [20, 40, 60])
    rel = lambda w: 200. + 300. * np.floor(w)
    s = dw.split(_clouds("2stream", w1, rel(w1)), _clouds("raytracer", w3, rel(w3)), ["wb"])
    assert abs(s["offset"]) < 1.e-9 and abs(s["shift"] - 100.) < 1.e-9 and abs(s["total"] - 100.) < 1.e-9


def test_pure_offset_at_the_same_speed():
    w = np.repeat([0.5, 1.5, 2.5], [40, 40, 40])
    rel = lambda w: 200. + 300. * np.floor(w)
    s = dw.split(_clouds("2stream", w, rel(w)), _clouds("raytracer", w, rel(w) + 80.), ["wb"])
    assert abs(s["shift"]) < 1.e-9 and abs(s["offset"] - 80.) < 1.e-9


def test_parts_add_up_and_jackknife_is_zero_without_member_spread():
    w1 = np.repeat([0.5, 1.5, 2.5], [40, 40, 40]); w3 = np.repeat([0.5, 1.5, 2.5], [20, 40, 60])
    rel = lambda w: 200. + 300. * np.floor(w)
    d = pd.concat([_clouds("2stream", w1, rel(w1)), _clouds("raytracer", w3, rel(w3) + 50.)])
    s = dw.split_jackknife(d, ["wb"])
    assert abs(s["shift"] + s["offset"] - s["total"]) < 1.e-9 and abs(s["offset"] - 50.) < 1.e-9
    assert s["offset_se"] < 1.e-9 and s["shift_se"] < 1.e-9


def test_bootstrap_slope_recovers_a_known_slope():
    rng = np.random.default_rng(1)
    w = rng.uniform(0.5, 4., 800)
    d = pd.DataFrame(dict(rt="2stream", rep=np.arange(800) % 4 + 1, w=w, depth=300. + 120. * w + rng.normal(0., 50., 800)))
    s = dw.bootstrap_slope(d, n=500)
    assert s["lo"] < 120. < s["hi"] and s["hi"] - s["lo"] < 20. and abs(s["slope"] - 120.) < 6.   # se of slope is 1.7


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
