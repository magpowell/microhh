"""Exact checks of persistence.py on a constructed cloud history."""
import numpy as np
import pandas as pd

import masks as mk
import persistence as ps

N, NT, DT = 16, 12, 60.


def _history():
    path = np.zeros((NT, N, N), dtype=np.float32)
    path[:, 2:5, 2:5] = 1.                 # site A: cloudy throughout (12 frames)
    path[5:, 8:10, 8:10] = 1.              # site B: cloudy for the last 7 frames
    path[:, 12:14, 1] = 1.                 # site C: two columns; one continuous ...
    path[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], 12:14, 0] = 1.   # ... one with a gap at frame 3 (wraps to C across x)
    path[3, 12:14, 0] = 0.
    return path


def test_column_duration_and_object_means():
    path = _history()
    f = NT - 1
    dur = ps.cloudy_duration(path, f)
    assert dur[3, 3] == 12 and dur[8, 8] == 7 and dur[12, 1] == 12 and dur[12, 0] == 8 and dur[0, 0] == 0
    lab, n = mk.label_periodic(path[f] > 0.)
    assert n == 3
    o = ps.per_object(lab, n, dur, path, f, DT)
    by = {int(lab[3, 3]): "A", int(lab[8, 8]): "B", int(lab[12, 0]): "C"}
    got = {by[i + 1]: (o["dur_mean"][i], o["dur_max"][i], o["cloudy_10min_ago"][i]) for i in range(n)}
    assert got["A"] == (12., 12., 1.) and got["B"] == (7., 7., 0.)
    assert got["C"] == (10., 12., 1.)                                    # mean of 12 and 8 minutes
    assert o["cloudy_30min_ago"].tolist() == [0., 0., 0.]                # before the history starts


def test_track_columns():
    feats = pd.DataFrame(dict(frame=[0, 1, 1, 2, 2], track=[1, 1, 2, 1, 2], n_pred=[0, 3, 1, 1, 1],
                              age=[0., 60., 0., 120., 60.], area=[1., 2., 1., 3., 1.]))
    tracks = pd.DataFrame(dict(track=[1, 2], family=[1, 1], frame_first=[0, 1]))
    now = ps.track_columns(feats, tracks, 2, DT)
    assert now.merges.tolist() == [2, 0] and now.age.tolist() == [2., 1.] and now.family_age.tolist() == [2., 2.]
