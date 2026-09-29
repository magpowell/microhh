"""Exact checks of the geometry in neighbours.py."""
import numpy as np
import pandas as pd

import neighbours as nb


def test_sun_frame_axes():
    # sun due south (azimuth 180 deg): the shadow is displaced to the north, so +x is north
    xs, ys = nb.to_sun_frame(np.array([0., 100.]), np.array([100., 0.]), np.pi)
    assert np.allclose(xs, [100., 0.]) and np.allclose(ys, [0., -100.])
    # sun due west (270 deg): the shadow is displaced to the east
    xs, ys = nb.to_sun_frame(np.array([100.]), np.array([0.]), 1.5 * np.pi)
    assert np.allclose(xs, [100.]) and np.allclose(ys, [0.])
    # a rotation keeps lengths
    rng = np.random.default_rng(0)
    d = rng.normal(size=(2, 50))
    xs, ys = nb.to_sun_frame(d[0], d[1], 4.1)
    assert np.allclose(np.hypot(xs, ys), np.hypot(d[0], d[1]))


def test_relative_position_uses_the_nearest_image():
    dx, dy = nb.relative(100., 25500., np.array([25500., 300.]), np.array([100., 25400.]), 25600., 25600.)
    assert np.allclose(dx, [-200., 200.]) and np.allclose(dy, [200., -100.])


def test_side_counts_use_the_half_annulus():
    xs = np.array([-1., -0.2, -3.5, 2., 0.4, 1., -2.9])
    ys = np.array([0., 0., 0., 1., 0., 2.9, 0.5])
    assert nb.side_counts(xs, ys) == (2, 1)        # sunlit: -1 and (-2.9, 0.5); shadow: (2, 1); (1, 2.9) is outside


def test_ratio_pools_counts():
    t = [pd.DataFrame(dict(small_sunlit=[3, 1], small_shadow=[1, 1])), pd.DataFrame(dict(small_sunlit=[2], small_shadow=[2]))]
    r = nb.ratio(t, "small")
    assert r["ratio"] == 6. / 4. and r["sunlit"] == 6 and r["shadow"] == 4
    r = nb.ratio(t, "small", np.random.default_rng(1), n=200)
    assert r["lo"] <= 1.5 <= r["hi"]


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
