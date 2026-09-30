"""Exact check of the absorber search in merges.py on constructed label maps."""
import numpy as np

import merges as mg


def test_absorber_is_the_largest_overlap():
    last = np.zeros((10, 10), dtype=int)
    last[2:4, 2:4] = 1          # the dying cloud, 4 cells
    last[6:9, 6:9] = 2
    nxt = np.zeros((10, 10), dtype=int)
    nxt[1:4, 1:3] = 5           # overlaps 2 cells of cloud 1
    nxt[3:5, 3:5] = 7           # overlaps 1 cell of cloud 1
    nxt[6:9, 6:9] = 2
    assert mg.absorber(last, 1, nxt) == 5
    assert mg.absorber(last, 2, nxt) == 2
    assert mg.absorber(last, 1, np.zeros((10, 10), dtype=int)) == 0
