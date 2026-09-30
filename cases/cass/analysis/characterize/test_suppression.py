"""Exact checks of suppression.py: periodic open-ground mask and the accumulation of area-time and births by class."""
import numpy as np

import suppression as sp

N, DX = 20, 100.


def test_far_from_cloud_is_periodic():
    mask = np.zeros((N, N), dtype=bool)
    mask[0, 0] = True
    far = sp.far_from_cloud(mask, DX, DX, far=250.)
    assert not far[0, 0] and not far[0, 2] and not far[19, 0] and not far[18, 0]      # 200 m across the edge
    assert far[0, 3] and far[17, 0] and far[3, 3]
    assert far.sum() == N * N - 21                                                     # the disc of radius 2.5 cells holds 21


def test_accumulate_counts_by_class():
    A = np.zeros((N, N))
    A[:, :10] = -150.       # shaded half
    A[:, 10:] = 30.         # lit half
    open_cols = np.ones((N, N), dtype=bool)
    open_cols[5, 5] = False
    out = {k: np.zeros((sp.HOURS.size, sp.BINS.size - 1)) for k in ("area_time", "births")}
    births = {"births": (np.array([2, 5, 7]), np.array([3, 5, 15]))}      # shaded, in a cloud column (dropped), lit
    sp.accumulate(A, open_cols, births, 1., out, h=2)
    shaded = np.digitize(-150., sp.BINS) - 1
    lit = np.digitize(30., sp.BINS) - 1
    assert out["area_time"][2, shaded] == 10 * N - 1 and out["area_time"][2, lit] == 10 * N
    assert out["births"][2, shaded] == 1 and out["births"][2, lit] == 1 and out["births"].sum() == 2
