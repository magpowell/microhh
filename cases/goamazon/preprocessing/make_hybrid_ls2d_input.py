#
# HYBRID goamazon_ls2d_input.nc — production fallback while the ERA5
# model-level file sits in the CDS/MARS queue.
#
#   soil        REAL ERA5 (surface_an.nc, snapped request, 12 UTC 2014-10-05):
#               stl1-4 / swvl1-4 / slt at the site box mean. Root fraction
#               from the IFS profile for evergreen broadleaf (a_r=7.344,
#               b_r=1.303), normalized over the 4 ECMWF layers.
#   radiation   RCEMIP analytic tropical sounding (Wing et al. 2018) for the
#   + padding   background column, o3, and above-IOP-top padding profiles.
#               Only affects the column above ~16.5 km and the sponge zone;
#               everything below comes from the IOP file at input-generation
#               time.
#
# The scientifically material surface state is real; upgrade to the full
# ERA5 file (goamazon_ls2d_input.py) whenever MARS delivers, if desired.
#
# Also prints the ERA5 vegetation properties (cvh, tvh, lai_hv, fsr) to
# cross-check the values hardcoded in config/goamazon_{2stream,raytracer}.ini.
#
# Output: $SCRATCH/GOAMAZON_LES/shared_data/goamazon_ls2d_input_HYBRID.nc
#

import os
from pathlib import Path

import numpy as np
import netCDF4 as nc
import xarray as xr

Rd = 287.04
Rv = 461.5
ep = Rd/Rv

co2 = 397.e-6
ch4 = 1822.e-9
n2o = 327.e-9
n2  = 0.7808
o2  = 0.2095

T_0 = 300.
q_0 = 0.01864

SFC_AN = '/pscratch/sd/m/mpowell/LS2D_ERA5/goamazon/2014/10/05/surface_an.nc'
HOUR_UTC = 12


def rcemip_profiles(z):
    """RCEMIP analytic tropical sounding (Wing et al. 2018)."""
    z_q1, z_q2, z_t, q_t = 4.0e3, 7.5e3, 15.e3, 1.e-14

    q = q_0 * np.exp(-z/z_q1) * np.exp(-(z/z_q2)**2)
    q_tb = q_0 * np.exp(-z_t/z_q1) * np.exp(-(z_t/z_q2)**2)
    q -= q_tb + q_t
    above = z >= z_t
    q[above] = q_t

    gamma = 6.7e-3
    Tv_0 = (1. + 0.608*q_0)*T_0
    Tv = Tv_0 - gamma*z
    Tv_t = Tv_0 - gamma*z_t
    Tv[above] = Tv_t
    T = Tv / (1. + 0.608*q)

    g, cp, p0, p00 = 9.79764, 1005., 101480., 1e5
    p = p0 * (Tv/Tv_0)**(g/(Rd*gamma))
    p_strat = p0 * (Tv_t/Tv_0)**(g/(Rd*gamma)) * np.exp(-(g*(z - z_t))/(Rd*Tv_t))
    p[above] = p_strat[above]

    thl = T*(p00/p)**(Rd/cp)

    g1, g2, g3 = 3.6478, 0.83209, 11.3515
    o3 = g1 * (p/100.)**g2 * np.exp(-(p/100.)/g3) * 1e-6

    return p, q, T, thl, o3


def ifs_root_frac(a_r, b_r):
    """IFS root fraction over the 4 ECMWF soil layers, deepest first."""
    bounds = np.array([0., 0.07, 0.28, 1.00, 2.89])
    cum = 0.5*(np.exp(-a_r*bounds) + np.exp(-b_r*bounds))
    frac = cum[:-1] - cum[1:]          # per layer, top first
    frac /= frac.sum()
    return frac[::-1]                  # deepest first


# ------ real ERA5 soil at 12 UTC (site box mean) ------
sfc = xr.open_dataset(SFC_AN).isel(valid_time=HOUR_UTC)
box = sfc.mean(['latitude', 'longitude'])

t_soil = np.array([float(box[f'stl{k}']) for k in (4, 3, 2, 1)])       # deepest first
theta_soil = np.array([float(box[f'swvl{k}']) for k in (4, 3, 2, 1)])
# slt is categorical: use the value at the central grid point, not a mean
slt = int(sfc['slt'].sel(latitude=-3.25, longitude=-60.5, method='nearest'))

print('--- ERA5 surface_an site properties (12 UTC 2014-10-05) ---')
print(f'  soil type slt      : {slt} (ECMWF; index_soil = {slt-1})')
print(f'  t_soil (deep->sfc) : {np.array2string(t_soil, precision=2)}')
print(f'  theta  (deep->sfc) : {np.array2string(theta_soil, precision=3)}')
print(f'  cvh / tvh          : {float(box.cvh):.2f} / {int(sfc.tvh.sel(latitude=-3.25, longitude=-60.5, method="nearest"))} (6 = evergreen broadleaf)')
print(f'  lai_hv             : {float(box.lai_hv):.2f}')
print(f'  fsr (roughness)    : {float(box.fsr):.2f} m')
print(f'  skin T             : {float(box.skt):.2f} K')

z_soil = np.array([-1.945, -0.64, -0.175, -0.035])
root_frac = ifs_root_frac(a_r=7.344, b_r=1.303)   # IFS evergreen broadleaf

# ------ write output ------
SCRATCH  = Path(os.environ.get('SCRATCH', '/pscratch/sd/m/mpowell'))
OUT_DIR  = SCRATCH / 'GOAMAZON_LES' / 'shared_data'
OUT_FILE = OUT_DIR / 'goamazon_ls2d_input_HYBRID.nc'
OUT_DIR.mkdir(parents=True, exist_ok=True)

f = nc.Dataset(OUT_FILE, 'w', datamodel='NETCDF4', clobber=True)


def add_var(grp, name, dims, data):
    v = grp.createVariable(name, 'f8', dims)
    v[:] = data
    return v


dz_bg = 500.
z_top = 70.e3
z_lay = np.arange(dz_bg/2, z_top, dz_bg)
z_lev = np.append(np.arange(0, z_top - dz_bg/2, dz_bg), z_top)

p_lay, q_lay, t_lay, _, o3_lay = rcemip_profiles(z_lay)
p_lev, _,     t_lev, _, _      = rcemip_profiles(z_lev)

z_ref = np.arange(25., 26000., 50.)
p_ref, q_ref, _, _, o3_ref = rcemip_profiles(z_ref)

rad = f.createGroup('radiation')
rad.createDimension('lay', z_lay.size)
rad.createDimension('lev', z_lev.size)
rad.createDimension('z_ref', z_ref.size)
add_var(rad, 'z_lay', ('lay',), z_lay)
add_var(rad, 'p_lay', ('lay',), p_lay)
add_var(rad, 't_lay', ('lay',), t_lay)
add_var(rad, 'o3',    ('lay',), o3_lay)
add_var(rad, 'h2o',   ('lay',), q_lay / (ep - ep*q_lay))
add_var(rad, 'z_lev', ('lev',), z_lev)
add_var(rad, 'p_lev', ('lev',), p_lev)
add_var(rad, 't_lev', ('lev',), t_lev)
add_var(rad, 'z_ref_grid', ('z_ref',), z_ref)
add_var(rad, 'o3_z',  ('z_ref',), o3_ref)
add_var(rad, 'h2o_z', ('z_ref',), q_ref / (ep - ep*q_ref))
for name, val in (('co2', co2), ('ch4', ch4), ('n2o', n2o), ('n2', n2), ('o2', o2)):
    v = rad.createVariable(name, 'f8')
    v[:] = val

soil = f.createGroup('soil')
soil.createDimension('z', z_soil.size)
add_var(soil, 'z', ('z',), z_soil)
add_var(soil, 'theta_soil', ('z',), theta_soil)
add_var(soil, 't_soil', ('z',), t_soil)
add_var(soil, 'index_soil', ('z',), np.full(4, float(slt - 1)))
add_var(soil, 'root_frac', ('z',), root_frac)

prof = f.createGroup('era5_profiles')
t_prof = np.array([0., 43200.])
prof.createDimension('time', t_prof.size)
prof.createDimension('z', z_ref.size)
add_var(prof, 'time_sec', ('time',), t_prof)
add_var(prof, 'z', ('z',), z_ref)
_, q_p, _, thl_p, _ = rcemip_profiles(z_ref)
add_var(prof, 'thl', ('time', 'z'), np.tile(thl_p, (2, 1)))
add_var(prof, 'qt',  ('time', 'z'), np.tile(q_p,  (2, 1)))
add_var(prof, 'u',   ('time', 'z'), np.zeros((2, z_ref.size)))
add_var(prof, 'v',   ('time', 'z'), np.zeros((2, z_ref.size)))

f.description = ('HYBRID: soil = real ERA5 surface_an (2014-10-05 12 UTC, '
                 'site box mean); radiation background / o3 / padding = '
                 'RCEMIP analytic tropical sounding. Production fallback '
                 'while the ERA5 model-level file is queued at CDS/MARS.')
f.close()

print(f'\nWrote {OUT_FILE}')
