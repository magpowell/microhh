"""Analytic checks of drift.py on constructed fields. Tolerances are about 10x the errors measured on 2026-09-29,
except for the part-cell pattern shift (1.5x: the parabolic fit of the peak is good to 0.03 cells)."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np

import drift as dr
import track as tr

N, DX, DT = 64, 50., 60.
NOCLOUD = -1.e9
AZ = np.radians(225.)                                   # sun in the south-west
E = np.array([np.sin(AZ), np.cos(AZ)])                  # toward the sun
L = np.array([-np.cos(AZ), np.sin(AZ)])                 # to the left of that


def _fixed_sun(monkeypatch):
    monkeypatch.setattr(dr, "solar_azimuth", lambda run, t: (np.full(np.shape(t), AZ), None))


def _track(path):
    base, top = np.where(path > 0., 1000., NOCLOUD), np.where(path > 0., 1500., NOCLOUD)
    return tr.track(path, base, top, np.full_like(path, 9.97e36), np.arange(path.shape[0]) * DT, DX, DX)[0]


def _boxes(nt, clouds):
    """clouds: (first frame, last frame, j0, i0, size, cells per frame in y, in x)."""
    path = np.zeros((nt, N, N))
    for f0, f1, j0, i0, s, dj, di in clouds:
        for f in range(f0, f1 + 1):
            jj = np.arange(j0 + dj * (f - f0), j0 + dj * (f - f0) + s) % N
            ii = np.arange(i0 + di * (f - f0), i0 + di * (f - f0) + s) % N
            path[f][np.ix_(jj, ii)] = 0.1
    return path


def test_wrap():
    assert dr.wrap(np.array([3150., -3150., 100., 1600., -1600.]), 3200.).tolist() == [-50., 50., 100., -1600., -1600.]


def test_sun_direction_at_the_site():
    run = SimpleNamespace(t0=datetime(2005, 7, 24, 12), lon=-97.5, lat=36.5)
    e, n = dr.sun_direction(run, np.array([2. * 3600., 6.6 * 3600., 11. * 3600.]))    # 14:00, 18:36, 23:00 UTC
    assert e[0][0] > 0.9 and abs(e[1][0]) < 0.4          # morning: the sun is in the east
    assert abs(e[0][1]) < 0.02 and e[1][1] < -0.999      # solar noon: due south; measured x component 0.008
    assert e[0][2] < -0.9                                # evening: in the west
    assert np.allclose(n[0], -e[1]) and np.allclose(n[1], e[0])                      # left of the sun direction


def test_drift_of_a_cloud_crossing_the_boundary(monkeypatch):
    _fixed_sun(monkeypatch)
    feats = _track(_boxes(14, [(1, 12, 58, 50, 12, 1, 2)]))      # through the corner of the domain
    st = dr.project(dr.steps(feats, N * DX, N * DX), None)
    assert len(st) == 11
    assert np.abs(st["u"] - 2. * DX / DT).max() < 1.e-11 and np.abs(st["v"] - DX / DT).max() < 1.e-11   # measured 3e-13
    v = np.array([2., 1.]) * DX / DT
    assert np.abs(st["along"] - v @ E).max() < 1.e-11 and np.abs(st["across"] - v @ L).max() < 1.e-11
    assert (st["along"] < 0.).all()                      # toward the north-east is away from the sun
    t = dr.per_track(st)
    assert len(t) == 1 and t["n"].iloc[0] == 11


def test_small_clouds_are_left_out():
    feats = _track(_boxes(8, [(0, 7, 10, 10, 12, 0, 1), (0, 7, 40, 40, 6, 0, 1)]))    # 677 m and 339 m across
    assert dr.steps(feats, N * DX, N * DX)["track"].nunique() == 1
    assert dr.steps(feats, N * DX, N * DX, dmin=0.)["track"].nunique() == 2
    assert np.allclose(dr.diameter(feats["area"].max()), 2. * 600. / np.sqrt(np.pi))


def test_steps_with_a_merge_are_left_out():
    nt = 10
    path = _boxes(nt, [(0, 9, 10, 10, 12, 0, 0), (0, 3, 12, 24, 10, 0, 0)])
    path[4:, 10:22, 22:34] = 0.1                                 # from frame 4 the two are one cloud
    st = dr.steps(_track(path), N * DX, N * DX)
    assert not ((st["time"] > 3. * DT) & (st["time"] < 4. * DT)).any()
    assert len(st) > 0 and (st["u"].abs() < 1.e-12).all() and (st["v"].abs() < 1.e-12).all()


def test_growth_on_one_side_counts_as_drift():
    nt = 8
    path = np.zeros((nt, N, N))
    for f in range(nt):
        path[f, 20:32, 20:32 + 2 * f] = 0.1                      # east edge grows 2 cells per frame
    st = dr.steps(_track(path), N * DX, N * DX)
    assert np.abs(st["u"] - DX / DT).max() < 1.e-11 and st["v"].abs().max() < 1.e-11     # half the growth rate


def test_root_offset_and_drift_across_the_boundary(monkeypatch):
    _fixed_sun(monkeypatch)
    nt, s = 7, 12
    path = _boxes(nt, [(0, nt - 1, 56, 54, s, 0, 1)])            # cloud moves 1 cell per frame in x
    x = y = (np.arange(N) + 0.5) * DX
    fields = []
    for f in range(nt):                                          # root starts 3 cells west and 2 north of the cloud
        xr_ = ((54 + 0.5 * s) + f - 3. + 0.5 * f) * DX           # and moves 1.5 cells per frame
        yr_ = ((56 + 0.5 * s) + 2.) * DX
        d2 = dr.wrap(x[None, :] - xr_, N * DX)**2 + dr.wrap(y[:, None] - yr_, N * DX)**2
        fields.append(np.exp(-0.5 * d2 / (1.5 * DX)**2))
    st = dr.steps(_track(path), N * DX, N * DX)
    r = dr.root_steps(st, lambda f: fields[f], DX, DX, N * DX, N * DX)
    assert len(r) == nt - 1
    f = r["frame"].values
    tol = 3.e-4 * DX                                             # measured 3e-5 cells
    assert np.abs(r["rx"] - (-3. + 0.5 * f) * DX).max() < tol and np.abs(r["ry"] - 2. * DX).max() < tol
    assert np.abs(r["ur"] - 1.5 * DX / DT).max() < tol / DT and np.abs(r["vr"]).max() < tol / DT
    p = dr.project(r, None, [("ur", "vr", "_root"), ("rx", "ry", "_off")])
    assert np.abs(p["along_root"] - 1.5 * DX / DT * E[0]).max() < tol / DT
    assert np.abs(p["along_off"] - (r["rx"] * E[0] + r["ry"] * E[1])).max() < 1.e-9


def test_root_without_a_marker_is_missing():
    path = _boxes(4, [(0, 3, 20, 20, 12, 0, 0)])
    r = dr.root_steps(dr.steps(_track(path), N * DX, N * DX), lambda f: np.zeros((N, N)), DX, DX, N * DX, N * DX)
    assert r[["rx", "ry", "ur", "vr"]].isna().all().all()


def test_pattern_shift_recovers_whole_and_part_cells():
    rng = np.random.default_rng(5)
    ky, kx = np.meshgrid(np.fft.fftfreq(N), np.fft.rfftfreq(N), indexing="ij")
    spec = np.fft.rfft2(rng.normal(size=(N, N))) * np.exp(-(kx**2 + ky**2) / 0.05**2)     # smooth field
    a = np.fft.irfft2(spec, s=(N, N))
    for sx, sy, tol in ((3., -5., 1.e-9), (2.5, 0.25, 0.05)):   # measured 0 and 0.033 cells
        b = np.fft.irfft2(spec * np.exp(-2.j * np.pi * (kx * sx + ky * sy)), s=(N, N))
        dx, dy = dr.pattern_shift(a, b, DX, DX)
        assert abs(dx / DX - sx) < tol and abs(dy / DX - sy) < tol


def test_layer_mean_wind():
    nz = 30
    time = np.array([0., 300., 600.])
    u = np.tile(np.arange(nz, dtype=float), (3, 1)) * np.array([1., 2., 3.])[:, None]
    v = -0.5 * u
    frac = np.zeros((3, nz))
    frac[1:, 10:20] = 0.05                                       # no cloud at the first time
    uw, vw = dr.layer_mean_wind(time, u, v, frac, np.array([300., 450., 600.]))
    assert np.allclose(uw, [29., 36.25, 43.5]) and np.allclose(vw, [-14.5, -18.125, -21.75])
    assert dr.layer_mean_wind(time, u, v, frac, np.array([0.]))[0][0] == 0.
