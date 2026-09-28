"""Checks of the 2026-09-28 fixes in cass_analysis.py and updrafts/diagnostics.py."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "updrafts"))
import cass_analysis as ca
import diagnostics as dg


def _disk(n, R, cx, cy):
    jj, ii = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="ij")
    dx = np.minimum(abs(ii - cx), n - abs(ii - cx))
    dy = np.minimum(abs(jj - cy), n - abs(jj - cy))
    return dx**2 + dy**2 <= R**2


def test_wrapped_cloud_is_one_object_with_true_centroid_and_chord():
    n, d = 128, 50.
    m = _disk(n, 12, 2.5, 126.5)
    lab, props = ca.find_cloud_objects(m, d, d, min_L=500.)
    assert len(props) == 1 and props[0]["cx"] == 2 and props[0]["cy"] == 126
    p = props[0]
    assert abs(p["area_m2"] - m.sum() * d * d) < 1e-6
    row, col = m[p["cy"], :], m[:, p["cx"]]
    assert row[0] and row[-1] and col[0] and col[-1]                   # the chord crosses both edges
    assert ca.chord_length_1d(lab, p["label"], p["cy"], p["cx"], d, d, "y") == row.sum() * d
    assert ca.chord_length_1d(lab, p["label"], p["cy"], p["cx"], d, d, "x") == col.sum() * d
    assert row.sum() == 25                                             # |i - 2| <= 12


def test_small_objects_removed_and_non_convex_chord_is_contiguous():
    m = np.zeros((64, 64), bool)
    m[10, 5:9] = True                      # small object
    m[30:50, 20:24] = True                 # U shape: two arms joined at the bottom
    m[30:50, 36:40] = True
    m[46:50, 20:40] = True
    lab, props = ca.find_cloud_objects(m, 50., 50., min_L=500.)
    assert len(props) == 1 and lab[10, 5] == 0
    p = props[0]
    assert ca.chord_length_1d(lab, p["label"], 35, 30, 50., 50., "y") == 4 * 50.   # one arm, not 8 pixels
    assert ca.chord_length_1d(lab, p["label"], 48, 30, 50., 50., "y") == 20 * 50.


def test_centre_on_chord_wrapped_run():
    n = 32
    row = np.zeros(n, bool); row[[30, 31, 0, 1, 2]] = True
    f = np.tile(np.arange(n, dtype=float), (3, 1))
    (g, none), off, n_run = ca.centre_on_chord((f, None), row, 0)
    assert none is None and n_run == 5
    assert list(g[0][np.abs(off) <= 2]) == [30., 31., 0., 1., 2.] and list(off[np.abs(off) <= 2]) == [-2, -1, 0, 1, 2]
    assert np.all(np.diff(off) == 1)
    row = np.zeros(n, bool); row[10:14] = True                 # even length: offsets are half-integers
    (g,), off, n_run = ca.centre_on_chord((f,), row, 11)
    assert n_run == 4 and list(g[0][np.abs(off) < 2]) == [10., 11., 12., 13.] and off[np.abs(off) < 2][0] == -1.5


def test_cache_stamp_netcdf_and_sidecar():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        xr.Dataset({"a": ("x", [1., 2.])}).to_netcdf(t / "old.nc")
        ca.stamp_cache(xr.Dataset({"a": ("x", [1., 2.])})).to_netcdf(t / "new.nc")
        assert not ca.cache_is_current(t / "old.nc") and ca.cache_is_current(t / "new.nc")
        assert not ca.cache_is_current(t / "missing.nc")
        (t / "c.pkl").write_bytes(b"x")
        assert not ca.cache_is_current(t / "c.pkl")
        ca.write_stamp(t / "c.pkl")
        assert ca.cache_is_current(t / "c.pkl")
        try:
            ca.require_current_cache(t / "old.nc")
        except RuntimeError as e:
            assert "recompute" in str(e)
        else:
            raise AssertionError("stale cache accepted")


def _ds3d():
    z = (np.arange(8) + 0.5) * 25.
    w = xr.DataArray(np.ones((1, 8, 4, 4)) * 2., dims=("time", "z", "y", "x"), coords=dict(z=z))
    rho = xr.DataArray(1.1 - 1e-4 * z, dims="z", coords=dict(z=z))
    C = xr.DataArray(np.ones((1, 8, 4, 4)), dims=("time", "z", "y", "x"), coords=dict(z=z))
    return xr.Dataset(dict(w_cc=w, rhoref=rho, couvreux=C)), z


def test_mass_flux_uses_rhoref_profile():
    ds, z = _ds3d()
    mask = xr.zeros_like(ds["w_cc"], dtype=bool); mask[..., :2, :2] = True
    mf = dg.updraft_mass_flux(ds, dict(mask_paper=mask))
    assert mf["M_up_paper"].dims == ("time", "z")
    assert np.allclose(mf["M_up_paper"].values[0], (1.1 - 1e-4 * z) * 0.25 * 2., rtol=0, atol=1e-15)
    try:
        dg.updraft_mass_flux(ds.drop_vars("rhoref"), dict(mask_paper=mask))
    except ValueError:
        pass
    else:
        raise AssertionError("missing rhoref accepted")


def test_siebesma_guard_masks_small_contrast():
    ds, z = _ds3d()
    mask = xr.zeros_like(ds["w_cc"], dtype=bool); mask[..., :2, :2] = True
    C = ds["couvreux"].copy()
    excess = np.array([1., 1., 1., 1., 1., 1e-3, 1e-6, 0.])       # in-updraft excess over the rest
    C.values[..., :2, :2] += excess[None, :, None, None]
    ds["couvreux"] = C
    mf = dg.updraft_mass_flux(ds, dict(mask_cloudy_up=mask))
    out = dg.entrainment_rate_siebesma(ds, dict(mask_cloudy_up=mask), mf)
    e = out["eps_sc"].values[0]
    assert np.isfinite(e[:5]).all() and np.isnan(e[5:]).all()      # contrast below 5 % of its maximum is masked


def test_w_cc_has_no_nan_at_top_and_correct_bottom():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        z = (np.arange(6) + 0.5) * 25.; zh = np.arange(6) * 25.; x = (np.arange(4) + 0.5) * 50.
        w = np.arange(6, dtype=float)[None, :, None, None] * np.ones((1, 6, 4, 4))
        xr.Dataset({"w": (("time", "zh", "y", "x"), w)}, coords=dict(time=[0.], zh=zh, y=x, x=x)).to_netcdf(t / "w.nc")
        xr.Dataset({"ql": (("time", "z", "y", "x"), np.zeros((1, 6, 4, 4)))},
                   coords=dict(time=[0.], z=z, y=x, x=x)).to_netcdf(t / "ql.nc")
        ds = ca.load_3d_nc(t, variables=["ql", "w"])
        wcc = ds["w_cc"].values[0, :, 0, 0]
        assert np.allclose(wcc[:5], np.arange(5) + 0.5) and wcc[5] == 2.5 and not np.isnan(wcc).any()


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
