"""Exact check of the cloud-base bookkeeping of cloudbase_series.py on a constructed field."""
import numpy as np

import cloudbase_series as cb
import masks as mk


class _Run:
    def __init__(self, nz, n, dz, dx):
        self.z = (np.arange(nz) + 0.5) * dz
        self.ktot, self.itot, self.jtot, self.dx, self.dy = nz, n, n, dx, dx

    def lst(self, t):
        return 12. + t / 3600.


def test_mass_flux_and_root_anomaly(monkeypatch):
    nz, n, dz = 40, 16, 25.
    run = _Run(nz, n, dz, 50.)
    qc = np.zeros((nz, n, n)); w = np.zeros((nz, n, n)); thv = np.full((nz, n, n), 300.) + 0.01 * np.arange(nz)[:, None, None]
    thl = np.full((nz, n, n), 298.); qt = np.full((nz, n, n), 0.010); T = np.full((nz, n, n), 290.); qv = qt.copy()
    # one cloud of 4 columns from level 20 to 30, rising at 2 m/s, buoyant by 0.5 K; root below it warmer by 0.2 K at level 10
    qc[20:31, 4:6, 4:6] = 1.e-4; w[20:31, 4:6, 4:6] = 2.; thv[20:31, 4:6, 4:6] += 0.5
    thv[10, 4:6, 4:6] += 0.2; w[10, 4:6, 4:6] = 1.
    rho = np.full(nz, 1.1); rho[20] = 1.0
    f = dict(qc=qc, w=w, thv=thv, thl=thl, qt=qt, T=T, qv=qv, ql=qc, qi=np.zeros_like(qc))
    bs = dict(rhoref=rho, pref=1.e5 - 10. * run.z, exnref=np.linspace(1., 0.9, nz))
    monkeypatch.setattr(cb, "load", lambda r, t: (f, bs))
    monkeypatch.setattr(cb, "lift_thl", lambda env, kb, ktop, thl0, qt0, eps: np.where(np.arange(nz) < 25, -0.01, 0.01))
    out = cb.frame(run, 0)
    assert out["zb"] == run.z[20]
    assert out["n_core"] == 4 and np.isclose(out["a_core"], 4. / 256.) and out["w_core"] == 2.
    assert np.isclose(out["M_core"], 1.0 * 4. / 256. * 2.)
    assert np.isclose(out["thv_core"], 0.5 - 0.5 * 4. / 256.)            # anomaly against the slab mean at cloud base
    k50 = int(np.argmin(np.abs(run.z - 0.5 * run.z[20])))
    assert k50 == 10 and np.isclose(out["thv_root_50"], 0.2 - 0.2 * 4. / 256.) and np.isclose(out["w_root_50"], 1. - 4. / 256.)
    assert out["cin_undilute"] > 0. and out["frac_above_undilute"] in (0., 1.) and out["lfc_undilute"] == 1
    assert np.isclose(out["cover"], 4. / 256.)
    monkeypatch.setattr(cb, "lift_thl", lambda env, kb, ktop, thl0, qt0, eps: np.full(nz, -0.01))     # never buoyant: no barrier defined
    out = cb.frame(run, 0)
    assert out["lfc_entraining"] == 0 and np.isnan(out["cin_entraining"]) and np.isnan(out["wcrit_entraining"]) and np.isnan(out["frac_above_entraining"])
