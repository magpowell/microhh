"""End-to-end check of snapshot.py on a constructed run directory with known answers."""
import os
import tempfile
from pathlib import Path

import numpy as np

import thermo as th

N, K, DX, DZ = 64, 96, 50., 25.
T_SNAP = 3600


def disk(R, cx, cy):
    jj, ii = np.meshgrid(np.arange(N) + 0.5, np.arange(N) + 0.5, indexing="ij")
    dx = np.minimum(abs(ii - cx), N - abs(ii - cx))
    dy = np.minimum(abs(jj - cy), N - abs(jj - cy))
    return dx**2 + dy**2 <= R**2


def build(d):
    z = (np.arange(K) + 0.5) * DZ
    zh = np.arange(K) * DZ
    x = (np.arange(N) + 0.5) * DX
    xh = np.arange(N) * DX
    np.concatenate([x, xh, x, xh, z, zh]).tofile(d / "grid.0000000")
    (d / "cass.ini").write_text(
        f"[grid]\nitot={N}\njtot={N}\nktot={K}\nxsize={N*DX}\nysize={N*DX}\nzsize={K*DZ}\nlat=36.5\nlon=-97.5\n"
        "[time]\ndatetime_utc=2005-07-24 10:30:00\n")
    pref = 97000. * np.exp(-th.grav * z / (th.Rd * 295.))
    prefh = 97000. * np.exp(-th.grav * np.arange(K + 1) * DZ / (th.Rd * 295.))
    exn, exnh = th.exner(pref), th.exner(prefh)
    thvref, rho = np.full(K, 302.), pref / (th.Rd * exn * 302.)
    np.concatenate([np.full(K, 300.), np.full(K, 0.012), thvref, np.full(K + 1, 302.), pref, prefh, exn, exnh,
                    rho, np.full(K + 1, 1.)]).tofile(d / f"thermo_basestate.{T_SNAP:07d}")

    thl = np.where(z > 1000., 300. + 0.003 * (z - 1000.), 300.)[:, None, None] * np.ones((1, N, N))
    qt = np.where(z > 1000., 0.012 - 4.e-6 * (z - 1000.), 0.012)[:, None, None] * np.ones((1, N, N))
    wh = np.full((K, N, N), -0.05)
    c1, c2, r1, r2 = disk(5, 20, 20), disk(4, 0.3, 62), disk(8, 20, 20), disk(6, 0.3, 62)
    cl = (z > 1000.) & (z < 1500.)
    clh = (zh >= 1000.) & (zh <= 1500.)
    sub = zh < 1000.
    for k in np.flatnonzero(cl):
        thl[k][c1], qt[k][c1] = 300.5, 0.014
        thl[k][c2], qt[k][c2] = 297.5, 0.0115
    for k in np.flatnonzero(clh):
        wh[k][c1], wh[k][c2] = 2., 1.
    for k in np.flatnonzero(sub & (zh > 0)):
        wh[k][r1], wh[k][r2] = 1., 0.5
        thl[k][r1] += 0.2
    wh[0] = 0.
    a = th.sat_adjust(thl, qt, pref[:, None, None], exn[:, None, None], tol=1.e-14, nitermax=50)
    for name, arr in (("thl", thl), ("qt", qt), ("w", wh), ("T", a["T"]), ("ql", a["ql"]), ("qi", a["qi"])):
        arr.astype("<f8").tofile(d / f"{name}.{T_SNAP:07d}")
    return dict(z=z, pref=pref, exn=exn, thl=thl, qt=qt, T=a["T"], ql=a["ql"], c1=c1, c2=c2, r1=r1, r2=r2)


def test_snapshot_on_constructed_run():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SCRATCH"] = tmp
        import snapshot
        d = Path(tmp) / "CASS_LES" / "experiments" / "synthetic" / "2stream" / "rep_01"
        d.mkdir(parents=True)
        s = build(d)
        ds = snapshot.analyse("synthetic", "2stream", 1, T_SNAP)
        z, kb = s["z"], 40
        assert z[kb] == 1012.5 and list(ds["kb"].values) == [kb, kb, kb]
        n1, n2 = int(s["c1"].sum()), int(s["c2"].sum())
        assert int(ds["n_core_zb"][0]) == n1                                   # cloud 2 is cloudy but not core
        assert abs(float(ds["frac_cu"][kb]) - (n1 + n2) / N**2) < 1e-15
        assert abs(ds.attrs["clear_column_frac"] - (1. - (n1 + n2) / N**2)) < 1e-15
        assert float(ds["w_core_zb"][0]) == 2.
        j, i = np.argwhere(s["c1"])[0]
        Tc, qlc = s["T"][kb, j, i], s["ql"][kb, j, i]
        assert qlc > 0
        hc = th.mse(Tc, z[kb], 0.014 - qlc)
        assert abs(float(ds["h_core_zb"][0]) - hc) < 1e-6
        je, ie = np.argwhere(~(s["c1"] | s["c2"]))[0]
        he = th.mse(s["T"][kb, je, ie], z[kb], s["qt"][kb, je, ie])
        assert abs(float(ds["h_env"][kb]) - he) < 1e-6
        assert abs(float(ds["hsat_env"][kb]) - th.mse_sat(s["T"][kb, je, ie], z[kb], s["pref"][kb])) < 1e-6
        assert int(ds.attrs["n_cloud"]) == 2                                   # wrapped cloud counted once
        D = np.sort(ds["root_D_w0"].values)
        exp = np.sort([2. * np.sqrt(s["r1"].sum() * DX * DX / np.pi), 2. * np.sqrt(s["r2"].sum() * DX * DX / np.pi)])
        assert np.abs(D - exp).max() < 1e-9
        assert abs(exp[1] / (2 * 8 * DX) - 1.) < 3e-2                          # pixelisation, measured 1.7e-2
        assert list(np.sort(ds["cloud_has_core"].values)) == [0., 1.]
        k5 = int(np.argmin(np.abs(z - 0.5 * z[kb])))
        thv = th.theta_v(s["thl"][k5], s["qt"][k5], 0., 0., s["exn"][k5])
        exp_anom = thv[s["c1"]].mean() - thv.mean()
        assert abs(float(ds["thv_root_anom"].sel(root_frac=0.5)[0]) - exp_anom) < 1e-10
        assert exp_anom > 0.19
        assert abs(ds.attrs["lst_solar"] - (3.899 + 1.)) < 0.02                # cass_analysis offset 3.899 h
        pdf = ds["w_core_pdf"].values[0]
        assert pdf.sum() == n1 and pdf[np.searchsorted(snapshot.W_BINS, 2., side="right") - 1] == n1

        import parcel, collect
        par = parcel.analyse(ds)
        assert np.abs(par["eps_int_qt"]).max() < 1e-12 and np.abs(par["eps_int_thl"]).max() < 1e-12   # uniform core
        par = par.isel(zb_thr=0)
        c = par.sel(start="core")
        assert float(c["CIN"].sel(kind="undilute")) == 0. and float(c["z_LFC"].sel(kind="undilute")) == z[kb]
        assert np.allclose(c["h_parcel"].sel(kind="undilute").values[kb:kb + 10], hc, atol=1e-6, rtol=0)
        kg = int(np.searchsorted(z, z[kb] + parcel.G_DEPTH, side="right") - 1)
        G = ds["hsat_env"].values[kb:kg + 1] - hc
        assert abs(float(c["G_max"].sel(kind="undilute")) - G.max()) < 1e-6
        u = par.sel(start="cu", kind="undilute")
        hm, qm = float(ds["h_cu_zb"][0]), float(ds["qt_cu_zb"][0])
        Tm, qlm = th.T_from_mse(hm, qm, z[kb], s["pref"][kb])
        Bm = th.virtual_temperature(Tm, qm - qlm, qlm) - th.virtual_temperature(s["T"][kb, je, ie], s["qt"][kb, je, ie], 0.)
        assert (float(u["CIN"]) == 0.) == bool(Bm > 0)                       # independent buoyancy sign at z_b
        assert float(u["w_crit"]) == np.sqrt(2. * float(u["CIN"]))
        ds.to_netcdf(snapshot.out_path("synthetic", "2stream", 1, T_SNAP))
        parcel.analyse(ds).to_netcdf(
            snapshot.out_path("synthetic", "2stream", 1, T_SNAP).with_name(f"parcel_{T_SNAP:07d}.nc"))
        collect.RTS["2stream"] = "1D"
        try:
            collect.main("synthetic")
        except KeyError as e:                 # only one RT type exists in the synthetic set
            assert "3D" in str(e)
        long = __import__("pandas").read_csv(Path(tmp) / "CASS_LES/analysis/characterize/synthetic/table_long.csv")
        got = long[long["zb_thr"].isna() | (long["zb_thr"] == 1e-4)].set_index("metric")["value"]
        assert got["w_core_zb"] == 2. and abs(got["h_core_zb"] - hc) < 1e-6 and got["n_cloud"] == 2


if __name__ == "__main__":
    test_snapshot_on_constructed_run(); print("ok test_snapshot_on_constructed_run")
