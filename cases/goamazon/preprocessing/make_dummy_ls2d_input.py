#
# DUMMY stand-in for goamazon_ls2d_input.nc — memory fit test ONLY.
#
# Builds the same NetCDF schema as goamazon_ls2d_input.py but from the
# RCEMIP analytic tropical sounding (Wing et al. 2018, SST 300 K; same
# formulas as cases/rcemip/rcemip_input.py) instead of ERA5, plus generic
# wet-tropical soil. Raytracer memory is allocation-dominated (grid dims),
# so this is sufficient to pick the grid while the CDS request is queued.
#
# The file is tagged with global attr dummy=1; goamazon_input.py prints a
# loud warning when it sees it. Replace with the real ERA5 file (and re-run
# setup_runs.py) before any science run.
#
# Output: $SCRATCH/GOAMAZON_LES/shared_data/goamazon_ls2d_input_DUMMY.nc
#

import os
from pathlib import Path

import numpy as np
import netCDF4 as nc

Rd = 287.04
Rv = 461.5
ep = Rd/Rv

# 2014 gas concentrations (same as the real preprocessing script)
co2 = 397.e-6
ch4 = 1822.e-9
n2o = 327.e-9
n2  = 0.7808
o2  = 0.2095

T_0 = 300.
q_0 = 0.01864


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
    o3 = g1 * (p/100.)**g2 * np.exp(-(p/100.)/g3) * 1e-6   # vmr

    return p, q, T, thl, o3


SCRATCH  = Path(os.environ.get('SCRATCH', '/pscratch/sd/m/mpowell'))
OUT_DIR  = SCRATCH / 'GOAMAZON_LES' / 'shared_data'
OUT_FILE = OUT_DIR / 'goamazon_ls2d_input_DUMMY.nc'

OUT_DIR.mkdir(parents=True, exist_ok=True)
f = nc.Dataset(OUT_FILE, 'w', datamodel='NETCDF4', clobber=True)


def add_var(grp, name, dims, data):
    v = grp.createVariable(name, 'f8', dims)
    v[:] = data
    return v


# ------ radiation group: background column to 70 km, dz=500 m ------
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

# ------ soil group: generic wet-tropical soil, ECMWF 4-layer ------
soil = f.createGroup('soil')
z_soil = np.array([-1.945, -0.64, -0.175, -0.035])   # deepest first
soil.createDimension('z', z_soil.size)
add_var(soil, 'z', ('z',), z_soil)
add_var(soil, 'theta_soil', ('z',), np.full(4, 0.35))
add_var(soil, 't_soil', ('z',), np.array([299.5, 300.0, 300.3, 300.5]))
add_var(soil, 'index_soil', ('z',), np.full(4, 2.))  # medium-texture vG class
add_var(soil, 'root_frac', ('z',), np.array([0.05, 0.25, 0.35, 0.35]))

# ------ era5_profiles group: time-constant analytic profiles, calm winds ------
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

f.dummy = 1
f.description = ('DUMMY RCEMIP-analytic stand-in for goamazon_ls2d_input.nc. '
                 'Memory fit test only -- NOT for science runs.')
f.close()

print(f'Wrote {OUT_FILE}')
print('DUMMY data (RCEMIP analytic + generic soil) -- fit test only.')
