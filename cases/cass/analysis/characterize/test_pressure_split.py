"""Checks of pressure_split.py on constructed run folders.

Tolerances are about 5x the errors measured on 2026-09-29; the errors come from the float32 files.
"""
import numpy as np
import pytest

import pressure as pr
import pressure_split as ps
import thermo as th
from les_io import Run

NX, NY, NZ, D = 32, 24, 40, 25.
THL0, AMP, MODE = 300., 0.5, 3
TOL_REST = 2.e-5      # measured 4.1e-6


def _folder(tmp_path, thl, qt, p, nk):
    ini = (f"[grid]\nitot={NX}\njtot={NY}\nktot={NZ}\nxsize={NX * D}\nysize={NY * D}\nzsize={NZ * D}\nlat=36.5\nlon=-97.5\n"
           "[time]\ndatetime_utc=2005-07-24 12:00:00\n")
    (tmp_path / "cass.ini").write_text(ini)
    x, y, z = ((np.arange(n) + 0.5) * D for n in (NX, NY, NZ))
    np.concatenate([x, x - 0.5 * D, y, y - 0.5 * D, z, z - 0.5 * D]).tofile(tmp_path / "grid.0000000")
    one, oneh = np.ones(NZ), np.ones(NZ + 1)
    np.concatenate([one, oneh]).tofile(tmp_path / "rhoref.0000000")
    np.concatenate([THL0 * one, 0. * one, THL0 * one, THL0 * oneh, 1.e5 * one, 1.e5 * oneh, one, oneh, one, oneh]
                   ).tofile(tmp_path / "thermo_basestate.0000000")
    for name, f in (("thl", thl), ("qt", qt), ("p", p)):
        f[:nk].astype("<f4").tofile(tmp_path / f"{name}_hf.0000000")
    return Run(tmp_path)


def _mode():
    """Dry temperature wave whose buoyancy and pressure are known exactly on the grid (constant density, walls)."""
    H, kx = NZ * D, 2. * np.pi * MODE / (NX * D)
    x, z, zh = (np.arange(NX) + 0.5) * D, (np.arange(NZ) + 0.5) * D, np.arange(NZ + 1) * D
    c = np.cos(kx * x)[None, None, :] * np.ones((1, NY, 1))
    thl = THL0 + AMP * c * np.sin(np.pi * z / H)[:, None, None]
    B = th.grav * AMP * np.cos(0.5 * np.pi * D / H) / THL0
    b = B * c * np.sin(np.pi * zh / H)[:, None, None]
    k2 = (2. - 2. * np.cos(kx * D)) / D**2 + (2. - 2. * np.cos(np.pi * D / H)) / D**2
    p = -B * (2. / D) * np.sin(0.5 * np.pi * D / H) * c * np.cos(np.pi * z / H)[:, None, None] / k2
    return thl, b, p


def test_buoyancy_and_its_pressure_match_the_exact_grid_solution(tmp_path):
    thl, b, p = _mode()
    run = _folder(tmp_path, thl, np.zeros_like(thl), p + 3., NZ)       # the constant must drop out
    s = ps.split(run, 0)
    assert np.abs(s["b"] - b).max() < 1.5e-4 * np.abs(b).max()        # measured 3.0e-5
    assert np.abs(s["pb"] - p).max() < 1.e-3 * np.abs(p).max()         # measured 1.8e-4
    assert np.abs(s["pd"]).max() < 1.e-3 * np.abs(p).max()             # the dumped pressure is all buoyancy here
    assert np.abs(s["b"] + s["a_pb"])[1:-1].max() > 0.1 * np.abs(b).max()   # not hydrostatic: kx is not zero


def test_rest_is_the_dumped_pressure_minus_the_buoyancy_part(tmp_path):
    thl, b, p = _mode()
    rng = np.random.default_rng(3)
    extra = rng.normal(size=p.shape)
    run = _folder(tmp_path, thl, np.zeros_like(thl), p + extra, NZ)
    s = ps.split(run, 0)
    assert np.abs(s["pd"] - ps.anomaly(extra)).max() < TOL_REST * np.abs(extra).max()


@pytest.mark.parametrize("nk", [30])
def test_padding_continues_the_column_to_the_model_top(tmp_path, nk):
    z = (np.arange(NZ) + 0.5) * D
    x, y = (np.arange(NX) + 0.5) * D, (np.arange(NY) + 0.5) * D
    blob = np.exp(-((z[:, None, None] - 300.)**2 + (y[None, :, None] - 300.)**2 + (x[None, None, :] - 400.)**2) / 100.**2)
    thl = THL0 + AMP * blob                                          # below 1e-9 of its peak at the top of the data
    run = _folder(tmp_path, thl, np.zeros_like(thl), np.zeros_like(thl), nk)
    thlh = 0.5 * (thl[1:] + thl[:-1])
    bh = np.zeros((NZ + 1, NY, NX))
    bh[1:NZ] = th.grav * ps.anomaly(thlh) / THL0
    ref = ps.anomaly(pr.project(None, None, bh, np.ones(NZ), np.ones(NZ + 1), D, D, D))[:nk]
    padded, wall = ps.split(run, 0)["pb"], ps.split(run, 0, pad=False)["pb"]
    scale = np.abs(ref).max()
    assert np.abs(padded - ref).max() < 1.e-4 * scale                # measured 1.9e-5
    assert np.abs(wall - ref).max() > 1.e-3 * scale                  # the wall at the top of the data is felt
