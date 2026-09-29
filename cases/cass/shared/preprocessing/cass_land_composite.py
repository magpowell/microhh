"""Soil and vegetation for the CASS composite from ERA5 at the site: mean over the case days at one hour.

Soil moisture, soil temperature, vegetation cover and leaf area index are averaged over the case days at the grid
point nearest the site; soil and vegetation types are taken there. Minimum canopy resistance, vapour-pressure-deficit
coefficient, vegetation density and root coefficients come from Table 8.1 of the IFS documentation Cy41r2 (the ERA5
cycle), skin conductivity from its Table 8.2.

python cass_land_composite.py --download          fetch the ERA5 single-level fields for the case days (small files)
python cass_land_composite.py --hour 12           write cass_land_composite_12utc.nc and print the values
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cass_utils import read_composite_days

LAT, LON = 36.5, -97.5
AREA = [37., -98., 36., -97.]                      # north, west, south, east
OUT = Path(os.environ["SCRATCH"]) / "CASS_LES" / "shared_data"
RAW = OUT / "era5_land"
STATE = ["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2", "volumetric_soil_water_layer_3",
         "volumetric_soil_water_layer_4", "soil_temperature_level_1", "soil_temperature_level_2",
         "soil_temperature_level_3", "soil_temperature_level_4", "soil_type", "type_of_low_vegetation",
         "type_of_high_vegetation", "low_vegetation_cover", "high_vegetation_cover",
         "leaf_area_index_low_vegetation", "leaf_area_index_high_vegetation", "forecast_surface_roughness",
         "forecast_logarithm_of_surface_roughness_for_heat", "skin_temperature"]
FLUX = ["instantaneous_surface_sensible_heat_flux", "instantaneous_moisture_flux"]
Z_SOIL = np.array([-0.035, -0.175, -0.64, -1.945])  # layer centres, top first

# IFS documentation Cy41r2, Part IV, Table 8.1: index: (name, H/L, rs_min [s/m], cveg, gD [1/hPa], a_r, b_r)
TABLE_8_1 = {1: ("Crops, mixed farming", "L", 100., 0.90, 0., 5.558, 2.614),
             2: ("Short grass", "L", 100., 0.85, 0., 10.739, 2.608),
             3: ("Evergreen needleleaf trees", "H", 250., 0.90, 0.03, 6.706, 2.175),
             4: ("Deciduous needleleaf trees", "H", 250., 0.90, 0.03, 7.066, 1.953),
             5: ("Deciduous broadleaf trees", "H", 175., 0.90, 0.03, 5.990, 1.955),
             6: ("Evergreen broadleaf trees", "H", 240., 0.99, 0.03, 7.344, 1.303),
             7: ("Tall grass", "L", 100., 0.70, 0., 8.235, 1.627),
             9: ("Tundra", "L", 80., 0.50, 0., 8.992, 8.992),
             10: ("Irrigated crops", "L", 180., 0.90, 0., 5.558, 2.614),
             11: ("Semidesert", "L", 150., 0.10, 0., 4.372, 0.978),
             13: ("Bogs and marshes", "L", 240., 0.60, 0., 7.344, 1.303),
             16: ("Evergreen shrubs", "L", 225., 0.50, 0., 6.326, 1.567),
             17: ("Deciduous shrubs", "L", 225., 0.50, 0., 6.326, 1.567),
             18: ("Mixed forest/woodland", "H", 250., 0.90, 0.03, 4.453, 1.631),
             19: ("Interrupted forest", "H", 175., 0.90, 0.03, 4.453, 1.631)}


def groups():
    d = pd.to_datetime(pd.Series(list(read_composite_days())))
    return {k: sorted(v.dt.day.tolist()) for k, v in d.groupby([d.dt.year, d.dt.month])}, d


def download(kind, workers=4):
    """One request per year (larger requests exceed the service's limit): all days of the months that hold case
    days; the case days are selected on reading."""
    import cdsapi
    from concurrent.futures import ThreadPoolExecutor
    RAW.mkdir(parents=True, exist_ok=True)
    _, d = groups()
    times = ["10:00", "12:00"] if kind == "state" else [f"{h:02d}:00" for h in range(12, 24)]

    def one(y):
        f = RAW / f"{kind}_{y}.nc"
        if not f.exists():
            months = sorted(d[d.dt.year == y].dt.month.unique())
            cdsapi.Client(quiet=True).retrieve(
                "reanalysis-era5-single-levels",
                dict(product_type=["reanalysis"], variable=STATE if kind == "state" else FLUX, year=[str(y)],
                     month=[f"{m:02d}" for m in months], day=[f"{x:02d}" for x in range(1, 32)], time=times,
                     area=AREA, data_format="netcdf", download_format="unarchived"), str(f))
        return f.name

    with ThreadPoolExecutor(workers) as ex:
        for name in ex.map(one, sorted(d.dt.year.unique())):
            print(f"fetched {name}", flush=True)


def root_frac(a_r, b_r, z=Z_SOIL):
    """IFS eq. 8.13; layers top first, the remainder goes to the deepest layer."""
    zh = np.zeros(z.size + 1)
    for k in range(z.size):
        zh[k + 1] = zh[k] + 2. * (-z[k] - zh[k])
    r = 0.5 * (np.exp(-a_r * zh[:-1]) + np.exp(-b_r * zh[:-1]) - np.exp(-a_r * zh[1:]) - np.exp(-b_r * zh[1:]))
    r[-1] += 1. - r.sum()
    return r


def site(ds):
    t = "valid_time" if "valid_time" in ds.dims else "time"
    return ds.sel(latitude=LAT, longitude=LON, method="nearest").rename({t: "time"})


def state(hour):
    _, days = groups()
    ds = site(xr.open_mfdataset(sorted(RAW.glob("state_*.nc")), combine="nested", concat_dim="valid_time").load())
    want = pd.DatetimeIndex([d + pd.Timedelta(hours=hour) for d in days])
    missing = want.difference(pd.DatetimeIndex(ds["time"].values))
    if len(missing):
        raise SystemExit(f"{len(missing)} case days are missing from the download, e.g. {missing[0]}")
    return ds.sel(time=want)


def composite(hour):
    d = state(hour)
    m = d.mean("time")
    one = lambda v: int(np.round(float(d[v].isel(time=0))))
    for v in ("slt", "tvl", "tvh"):
        assert float(d[v].max() - d[v].min()) == 0., v
    tl, th, st = one("tvl"), one("tvh"), one("slt")
    AL, AH = float(m["cvl"]), float(m["cvh"])
    L, H = TABLE_8_1[tl], TABLE_8_1.get(th)
    cl, ch = AL * L[3], (AH * H[3] if H else 0.)
    w = np.array([cl, ch]) / (cl + ch)
    lai = np.array([float(m["lai_lv"]), float(m["lai_hv"])])
    out = xr.Dataset(coords=dict(z=Z_SOIL[::-1]))
    out["theta_soil"] = ("z", np.array([float(m[f"swvl{k}"]) for k in (4, 3, 2, 1)]))
    out["t_soil"] = ("z", np.array([float(m[f"stl{k}"]) for k in (4, 3, 2, 1)]))
    out["index_soil"] = ("z", np.full(4, st - 1.))
    out["root_frac"] = ("z", (w[0] * root_frac(L[5], L[6]) + (w[1] * root_frac(H[5], H[6]) if H else 0.))[::-1])
    out["theta_soil_sd"] = ("z", np.array([float(d[f"swvl{k}"].std("time", ddof=1)) for k in (4, 3, 2, 1)]))
    a = dict(hour_utc=hour, n_days=d.sizes["time"], latitude=float(d.latitude), longitude=float(d.longitude),
             soil_type=st, type_low=tl, type_low_name=L[0], type_high=th, type_high_name=H[0] if H else "none",
             cover_low=AL, cover_high=AH, density_low=L[3], density_high=H[3] if H else 0.,
             lai_low=float(lai[0]), lai_high=float(lai[1]), c_veg=cl + ch, lai=float((w * lai).sum()),
             rs_veg_min=float(w[0] * L[2] + (w[1] * H[2] if H else 0.)), gD=float(w[0] * L[4] + (w[1] * H[4] if H else 0.)),
             weight_low=float(w[0]), weight_high=float(w[1]), z0m_era5=float(m["fsr"]), z0h_era5=float(np.exp(m["flsr"])),
             skin_temperature=float(m["skt"]),
             source="ERA5 single levels at the nearest grid point; IFS documentation Cy41r2 Part IV Tables 8.1")
    out.attrs.update(a)
    return out, d


def fluxes():
    f = sorted(RAW.glob("flux_*.nc"))
    if not f:
        return None
    _, days = groups()
    d = site(xr.open_mfdataset(f, combine="nested", concat_dim="valid_time").load())
    d = d.sel(time=d["time"].dt.floor("D").isin(days.values))
    g = d.groupby(d["time"].dt.hour).mean()
    return pd.DataFrame(dict(hour_utc=g["hour"].values, H=-g["ishf"].values, LE=-2.5e6 * g["ie"].values))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--fluxes", action="store_true")
    ap.add_argument("--hour", type=int, default=12)
    a = ap.parse_args()
    if a.download:
        download("flux" if a.fluxes else "state")
        sys.exit()
    out, d = composite(a.hour)
    out.to_netcdf(OUT / f"cass_land_composite_{a.hour:02d}utc.nc")
    print(out)
    fl = fluxes()
    if fl is not None:
        fl.to_csv(OUT / "era5_surface_fluxes_composite.csv", index=False)
        print(fl.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
