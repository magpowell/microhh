"""Checks of track3d.py on constructed 3D cloud histories."""
import numpy as np

import track3d as t3
from track import link, overlaps

NZ, N = 6, 12


def test_label_periodic_3d_joins_across_edges_by_faces_only():
    m = np.zeros((NZ, N, N), dtype=bool)
    m[2, 3, 0] = m[2, 3, N - 1] = True          # face neighbours across x
    m[1, 0, 5] = m[1, N - 1, 5] = True          # across y
    m[4, 6, 6] = m[5, 7, 7] = True              # diagonal only: two objects
    lab, n = t3.label_periodic_3d(m)
    assert n == 4 and lab[2, 3, 0] == lab[2, 3, N - 1] and lab[1, 0, 5] == lab[1, N - 1, 5] and lab[4, 6, 6] != lab[5, 7, 7]


def test_props():
    m = np.zeros((NZ, N, N), dtype=bool)
    m[1:4, 2:4, 2:5] = True                     # 3 levels x 6 columns
    core = np.zeros_like(m); core[2, 2, 2] = True
    lab, n = t3.label_periodic_3d(m)
    z = 100. * np.arange(NZ)
    p = t3.props(lab, n, core, z, 50., 50.)
    assert n == 1 and p["volume"][0] == 18 * 2500. and p["core_volume"][0] == 2500. and p["area"][0] == 6 * 2500.
    assert p["z_base"][0] == 100. and p["z_top"][0] == 300. and p["depth"][0] == 200.


def test_merge_is_linked_to_the_larger_overlap():
    a = np.zeros((NZ, N, N), dtype=bool); a[1:3, 1:4, 1:4] = True; a[1:3, 1:3, 6:8] = True
    b = np.zeros_like(a); b[1:3, 1:4, 1:8] = True
    la, na = t3.label_periodic_3d(a); lb, nb = t3.label_periodic_3d(b)
    x, y, k = overlaps(la, lb)
    pred = link(x, y, k, na, nb)
    assert nb == 1 and pred[1] == la[1, 1, 1]       # the bigger parent continues, the smaller one merges
