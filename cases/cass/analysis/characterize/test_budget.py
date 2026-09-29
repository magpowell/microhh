"""Analytic checks of budget.py on constructed profiles. Tolerances are about 10x the errors measured on 2026-09-29."""
import numpy as np
import xarray as xr

import budget as bd

NZ, DZ = 40, 25.
ZH = np.arange(NZ + 1) * DZ
Z = 0.5 * (ZH[1:] + ZH[:-1])
RHO = 1.15 * np.exp(-Z / 9000.)
TIME = np.arange(0., 3601., 300.)


def _data(advec, ls):
    """qt that changes by the two given tendencies (time, z), integrated exactly by the caller."""
    d = xr.Dataset(dict(qtt_advec=(("time", "z"), advec), qtt_ls=(("time", "z"), ls),
                        qtt_total=(("time", "z"), advec + ls), rho=("z", RHO), zh=("zh", ZH)),
                   coords=dict(time=TIME, z=Z))
    return d


def test_thickness_counts_partial_levels():
    w = bd.thickness(ZH, 30., 110.)
    assert w.sum() == 80. and w[1] == 20. and w[4] == 10. and w[0] == 0. and w[5] == 0.


def test_integral_of_a_known_profile():
    d = _data(np.zeros((TIME.size, NZ)), np.zeros((TIME.size, NZ)))
    one = xr.DataArray(np.ones(NZ), dims="z", coords=dict(z=Z))
    exact = 1.15 * 9000. * (np.exp(-100. / 9000.) - np.exp(-600. / 9000.))
    assert abs(float(bd.integral(one, d, 100., 600.)) / exact - 1.) < 1.e-5      # measured 3.2e-7, midpoint rule


def test_closure_is_exact_for_tendencies_linear_in_time():
    a = 1.e-8 * (1. + TIME[:, None] / 3600.) * np.exp(-Z[None, :] / 1000.)
    l = -2.e-9 * np.ones((TIME.size, NZ))
    d = _data(a, l)
    # exact time integral of the tendencies from 0 to t
    q = (1.e-8 * (TIME[:, None] + TIME[:, None]**2 / 7200.) * np.exp(-Z[None, :] / 1000.) - 2.e-9 * TIME[:, None])
    d["qt"] = (("time", "z"), 0.01 + q)
    c = bd.closure(d, "qt", 0., 750., 600., 3000.)
    assert abs(c["residual"]) < 1.e-12 * abs(c["change"])                       # measured 1e-16
    assert abs(c["transport"] + c["ls"] - c["total"]) < 1.e-12 * abs(c["total"])
    assert abs(c["ls"] / (-2.e-9 * 2400. * float(bd.integral(xr.ones_like(d["rho"]), d, 0., 750.))) - 1.) < 1.e-12


def test_closure_reports_the_sampling_error():
    """A tendency that oscillates between the snapshots is missed by them; the residual must show it."""
    a = np.zeros((TIME.size, NZ))
    d = _data(a, a.copy())
    d["qt"] = (("time", "z"), 0.01 + 1.e-5 * (TIME[:, None] / 3600.) * np.ones((1, NZ)))
    c = bd.closure(d, "qt", 0., 1000., 0., 3600.)
    assert c["total"] == 0. and abs(c["residual"] / c["change"] - 1.) < 1.e-12
