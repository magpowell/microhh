"""Exact checks of the helper functions in timing.py on constructed series."""
import numpy as np

import timing as tm


def test_running_mean_of_a_line_is_the_line_away_from_the_ends():
    x = 3. + 0.5 * np.arange(40.)
    s = tm.smooth(x, 7)
    assert np.abs(s[3:-3] - x[3:-3]).max() < 1.e-12
    x[10] = np.nan
    assert np.isfinite(tm.smooth(x, 7)).all()


def test_onset_needs_a_sustained_difference_of_the_final_sign():
    lst = 8. + np.arange(60) / 12.
    se = np.full(60, 1.)
    d = np.zeros(60)
    d[10] = 5.                      # isolated spike
    d[20:24] = -5.                  # sustained but wrong sign
    d[30:] = 3.
    assert tm.onset(lst, d, se, hold=6, k=2.) == lst[30]
    assert np.isnan(tm.onset(lst, np.ones(60), se, hold=6, k=2.))


def test_reach_is_the_last_upward_crossing():
    lst = np.arange(50.)
    d = np.r_[np.zeros(10), [6.], np.zeros(9), np.linspace(0., 10., 30)]
    ref = d[-6:].mean()
    k = np.flatnonzero(d / ref >= 0.5)
    assert tm.reach(lst, d, 0.5) == lst[[i for i in k if i > 10][0]]


def test_difference_and_standard_error():
    x = np.zeros((2, 4, 3))
    x[0] = np.array([1., 2., 3., 4.])[:, None]
    x[1] = np.array([3., 4., 5., 6.])[:, None]
    d, se = tm.difference(x)
    assert np.allclose(d, 2.) and np.allclose(se, np.sqrt(2. * (5. / 3.) / 4.))


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
