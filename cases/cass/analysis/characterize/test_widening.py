"""Exact checks of widening.py on a constructed frame: periodic birth counts, width change, ended and merged clouds, shadow shift."""
import numpy as np
import pandas as pd

import widening as wd

N, DX = 20, 100.


def test_nearby_births_wrap_around_the_edge():
    cx, cy, W = np.array([150.]), np.array([150.]), np.array([200.])       # cloud near the origin, radius 100 m
    bx = np.array([1950., 1050., 450.])                                    # 200 m to the left across the edge; far; 300 m to the right
    by = np.array([150., 150., 150.])
    assert wd.nearby_births(bx, by, cx, cy, W, N * DX, N * DX, radius=250.).tolist() == [2]
    assert wd.nearby_births(bx, by, cx, cy, W, N * DX, N * DX, radius=150.).tolist() == [1]
    assert wd.nearby_births(np.zeros(0), np.zeros(0), cx, cy, W, N * DX, N * DX).tolist() == [0]


def _frame():
    path = np.zeros((N, N), dtype=np.float32)
    path[2:6, 2:6] = 1.          # cloud 1: 16 cells, track 11, survives and grows
    path[10:14, 10:14] = 1.      # cloud 2: 16 cells, track 12, absorbed by a neighbour before f + 5
    path[16:20, 2:6] = 1.        # cloud 3: 16 cells, track 13, dissolves before f + 5
    path[19, 19] = 1.            # a speck, labelled last, dropped
    sw = np.full((N, N), 800.); sw[2:6, 2:6] = 700.
    shift = np.zeros((N, N)); shift[2:6, 2:6] = 150.; shift[10:14, 10:14] = 50.
    tracks_now = np.array([11, 12, 13, 14])
    feats_idx = {3: {11: (25 * DX * DX, 1)}, 5: {11: (25 * DX * DX, 0)}}
    death = {11: "end", 12: "merge", 13: "gone", 14: "gone"}
    births = pd.DataFrame(dict(frame_first=[-1, -3, -2], x=[350., 650., 1500.], y=[850., 350., 1500.]))   # first two within 1 km of cloud 1
    return path, sw, shift, tracks_now, feats_idx, death, births


def test_sample_frame():
    path, sw, shift, tracks_now, feats_idx, death, births = _frame()
    rows = wd.sample_frame(path, sw, shift, tracks_now, feats_idx, death, births, f=0, lag=5, dx=DX, dy=DX).set_index("track")
    assert list(rows.index) == [11, 12, 13]
    W = 2. * np.sqrt(16. * DX * DX / np.pi)
    assert np.isclose(rows.loc[11, "dW"], 2. * np.sqrt(25. * DX * DX / np.pi) - W) and not rows.loc[11, "ended"]
    assert rows.loc[11, "absorbed"] == 1 and rows.loc[11, "n_births"] == 2
    assert np.isclose(rows.loc[11, "dSW_root"], 700. - sw.mean()) and rows.loc[11, "dSW_shift"] == 150. and rows.loc[11, "lwp"] == 1.
    assert rows.loc[12, "ended"] and rows.loc[12, "merged"] and np.isclose(rows.loc[12, "dW"], -W) and rows.loc[12, "dSW_shift"] == 50.
    assert rows.loc[13, "ended"] and not rows.loc[13, "merged"] and np.isclose(rows.loc[13, "dW"], -W) and rows.loc[13, "dSW_shift"] == 0.


def test_sample_frame_without_shift_field():
    path, sw, shift, tracks_now, feats_idx, death, births = _frame()
    rows = wd.sample_frame(path, sw, None, tracks_now, feats_idx, death, births, f=0, lag=5, dx=DX, dy=DX)
    assert (rows.dSW_shift == 0.).all()


def test_fit_recovers_a_constructed_relation():
    rng = np.random.default_rng(0)
    rows = []
    for rt in wd.RTS:
        for rep in range(1, 5):
            n = 400
            W = rng.uniform(300., 2000., n); lwp = rng.uniform(0.01, 0.1, n)
            nb = rng.integers(0, 6, n); ab = rng.integers(0, 3, n); mg = rng.integers(0, 2, n)
            shift = rng.uniform(0., 300., n) if rt == "raytracer" else np.zeros(n)
            dW = 0.1 * shift + 10. * nb + 20. * ab - 50. * mg - 80. * np.log(W) + 5. * np.log(lwp) + (30. if rt == "raytracer" else 0.) + rng.normal(0., 1., n)
            rows.append(pd.DataFrame(dict(rt=rt, rep=rep, W=W, lwp=lwp, n_births=nb, absorbed=ab, merged=mg.astype(bool), dSW_shift=shift, dW=dW)))
    d = pd.concat(rows, ignore_index=True)
    f3 = wd.fit(d[d.rt == "raytracer"], "forcing")
    assert abs(f3["dSW_shift"] - 0.1) < 0.002 and abs(f3["n_births"] - 10.) < 0.3 and abs(f3["logW"] + 80.) < 1.
    # pooled without the forcing term: the 3D offset absorbs the mean forcing (0.1 x 150) plus the 30 m
    fp = wd.fit(d, "pooled", offset=True)
    assert abs(fp["is3D"] - 45.) < 3. and fp["is3D_lo"] < fp["is3D"] < fp["is3D_hi"] and fp["clusters"] == 8
