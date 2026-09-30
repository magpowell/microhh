"""Exact checks of widening.py on a constructed frame: periodic birth counts, width change, absorbed clouds, footprint anomaly."""
import numpy as np
import pandas as pd

import widening as wd

N, DX, DT = 20, 100., 60.


def test_nearby_births_wrap_around_the_edge():
    cx, cy, W = np.array([150.]), np.array([150.]), np.array([200.])       # cloud near the origin, radius 100 m
    bx = np.array([1950., 1050., 450.])                                    # 200 m to the left across the edge; far; 300 m to the right
    by = np.array([150., 150., 150.])
    assert wd.nearby_births(bx, by, cx, cy, W, N * DX, N * DX, radius=250.).tolist() == [2]
    assert wd.nearby_births(bx, by, cx, cy, W, N * DX, N * DX, radius=150.).tolist() == [1]
    assert wd.nearby_births(np.zeros(0), np.zeros(0), cx, cy, W, N * DX, N * DX).tolist() == [0]


def test_sample_frame():
    path = np.zeros((N, N), dtype=np.float32)
    path[2:6, 2:6] = 1.          # cloud 1: 16 cells, track 11
    path[10, 10] = 1.            # cloud 2: one cell, dropped
    sw = np.full((N, N), 800.)
    sw[2:6, 2:6] = 700.          # 100 W/m2 darker than the rest
    tracks_now = np.array([11, 12])
    # five minutes later track 11 has 25 cells and absorbed one cloud at frame 3
    feats_idx = {3: {11: (25 * DX * DX, 1)}, 5: {11: (25 * DX * DX, 0)}}
    births = pd.DataFrame(dict(frame_first=[-1, -3, -2], x=[350., 650., 1500.], y=[850., 350., 1500.]))   # first two within 1 km
    rows = wd.sample_frame(path, sw, tracks_now, feats_idx, births, f=0, lag=5, dx=DX, dy=DX, dt=DT)
    assert len(rows) == 1 and rows.track.iloc[0] == 11
    W = 2. * np.sqrt(16. * DX * DX / np.pi)
    assert np.isclose(rows.W.iloc[0], W) and np.isclose(rows.dW.iloc[0], 2. * np.sqrt(25. * DX * DX / np.pi) - W)
    assert rows.absorbed.iloc[0] == 1 and rows.n_births.iloc[0] == 2 and rows.n_births_ring.iloc[0] == 1
    assert np.isclose(rows.dSW_root.iloc[0], 700. - sw.mean()) and rows.lwp.iloc[0] == 1.
