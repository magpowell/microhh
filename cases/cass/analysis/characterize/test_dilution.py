"""Checks of dilution.py on constructed clouds whose cores dilute at a known rate."""
import numpy as np

import dilution as dl
from parcel import relax

NZ, N, DX, DZ = 60, 32, 50., 25.
Z = (np.arange(NZ) + 0.5) * DZ
EPS_A, EPS_B = 1.2e-3, 0.4e-3     # 1/m
K0, K1 = 10, 49


def _env():
    thl = 298. + 0.004 * Z            # increases with height
    qt = 0.012 - 2.e-6 * Z            # decreases with height
    return thl, qt


def _field():
    thl_e, qt_e = _env()
    thl = np.broadcast_to(thl_e[:, None, None], (NZ, N, N)).copy()
    qt = np.broadcast_to(qt_e[:, None, None], (NZ, N, N)).copy()
    qc = np.zeros((NZ, N, N)); w = np.full((NZ, N, N), -0.1); thv = np.full((NZ, N, N), 300.)
    clouds = {}
    # cloud A wraps across x = 0, 4 x 4 columns, dilutes at EPS_A; cloud B 4 x 4 columns at EPS_B, half its levels
    for name, eps, jj, ii, ka, kb in (("A", EPS_A, slice(4, 8), [30, 31, 0, 1], K0, K1), ("B", EPS_B, slice(20, 24), slice(10, 14), K0, 29)):
        tp = relax(Z, thl_e, thl_e[ka] - 1.5, ka, kb, eps)
        qp = relax(Z, qt_e, qt_e[ka] + 2.e-3, ka, kb, eps)
        for k in range(ka, kb + 1):
            thl[k, jj, ii] = tp[k]; qt[k, jj, ii] = qp[k]
            qc[k, jj, ii] = 1.e-4; w[k, jj, ii] = 1.; thv[k, jj, ii] = 301.
        clouds[name] = (tp, qp)
    # cloud C: cloudy but too shallow to fit (3 levels)
    qc[12:15, 26:28, 20:22] = 1.e-4; w[12:15, 26:28, 20:22] = 1.; thv[12:15, 26:28, 20:22] = 301.
    h = thl.copy()     # any conserved field works for the estimator; h is only carried through
    return dict(qc=qc, w=w, thv=thv, thl=thl, qt=qt, h=h), clouds


def _table():
    f, clouds = _field()
    t = dl.table(f, Z, DX, DX)
    order = np.argsort(-t["depth"])       # A deepest, then B, then C
    return t, order, clouds


def test_recovers_the_rate_of_dilution():
    t, (A, B, C), _ = _table()
    assert t["depth"].size == 3 and t["core_n_levels"][C] == 3 and np.isnan(t["core_eps_qt"][C])
    assert t["area"][A] == 16 * DX * DX and t["core_z_own"][A] == Z[K0] and t["core_z_fit1"][A] == Z[K1]
    # the trapezoidal integral biases eps low by (eps dz)^2 / 12: measured -7.50e-5 (A) and -8.33e-6 (B), qt and thl alike
    for c, eps in ((A, EPS_A), (B, EPS_B)):
        for v in ("qt", "thl"):
            assert abs(t[f"core_eps_{v}"][c] / eps - 1.) < 1.5 * (eps * DZ) ** 2 / 12., (v, t[f"core_eps_{v}"][c], eps)
            assert t[f"core_r_{v}"][c] < -0.999999
    assert np.allclose(t["cu_eps_qt"][[A, B]], t["core_eps_qt"][[A, B]])    # same points here


def test_mixing_fraction_and_base_excess():
    t, (A, B, C), clouds = _table()
    tp, qp = clouds["A"]
    thl_e, qt_e = _env()
    kf = K0 + int(np.searchsorted(Z[K0:], Z[K0] + dl.Z_F))
    assert t["core_z_f"][A] == Z[kf]
    cb = qp[K0:K0 + dl.NLEV + 1].mean()
    f_exact = (cb - qp[kf]) / (cb - qt_e[kf])
    assert abs(t["core_f_qt"][A] - f_exact) < 1.e-12
    assert abs(t["core_excess_qt"][A] - (cb - qt_e[K0:K0 + dl.NLEV + 1].mean())) < 1.e-15
    assert 0.1 < t["core_f_qt"][A] < 0.3 and t["core_f_qt"][B] < t["core_f_qt"][A]


def test_fit_run_stops_at_a_gap():
    cnt = np.array([0, 0, 5, 5, 5, 1, 5, 5])
    assert dl.fit_run(cnt, 3) == (2, 4)
    assert dl.fit_run(np.zeros(5, dtype=int), 3) == (-1, -1)
    assert dl.fit_run(np.array([4, 4, 4]), 3) == (0, 2)
