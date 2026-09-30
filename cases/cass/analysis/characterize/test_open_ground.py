"""Checks of open_ground.py: mass divergence on a constructed wind, classes, and the lifted parcel's LCL."""
import numpy as np

import open_ground as og
import thermo as th

N, DX = 16, 100.


def test_mass_divergence_of_a_constructed_wind():
    x = (np.arange(N)) * DX                       # u lives on the left faces xh = i dx
    kx = 2. * np.pi / (N * DX)
    u = np.broadcast_to(np.sin(kx * x)[None, None, :], (3, N, N)).astype(float)
    v = np.zeros((3, N, N))
    rho = np.array([1.2, 1.1, 1.0])
    d = og.mass_divergence(u, v, rho, DX, DX)
    expect = rho[:, None, None] * (np.sin(kx * (x + DX)) - np.sin(kx * x))[None, None, :] / DX
    assert np.allclose(d, expect)
    assert abs(d.sum()) < 1.e-12                 # periodic: no net divergence


def test_classes():
    qc = np.zeros((4, N, N))
    qc[1, 2:6, 2:6] = 1.e-4                      # one cloud of 16 cells at 100 m spacing = 1.6e5 m2
    qc[2, 10, 10] = 1.e-4                        # a speck: not a cloud
    anomaly = np.full((N, N), 30.)
    anomaly[:, 12:] = -100.
    c = og.classes(qc, DX, DX, anomaly, far=250., lit=-20., min_area=16 * DX * DX)
    assert c[3, 3] == 0 and c[3, 7] == 1 and c[3, 8] == 2 and c[3, 9] == 2 and c[3, 13] == 3 and c[10, 10] == 2   # 200 m near, 300 m open


def test_lifted_parcel_lcl():
    p = np.array([100000., 99000., 98000., 97000., 96000.])
    exn = th.exner(p)
    thl = np.array([300.])
    # choose qt so that the parcel saturates exactly at level 3 and not at level 2
    T3 = thl * exn[3]
    qt = np.array([float(th.qsat(p[3], T3[0]) * 1.0000001)])
    lcl, ql_kb = og.lifted_parcel(thl, qt, p, exn, kb=4)
    assert lcl.tolist() == [3] and ql_kb[0] > 0.
    lcl_dry, ql_dry = og.lifted_parcel(thl, qt * 0.9, p, exn, kb=4)
    assert lcl_dry.tolist() == [-1] and ql_dry[0] == 0.
