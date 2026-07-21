#
# GoAmazon single-pulse case: ERA5 (LS2D) preprocessing.
#
# Downloads ERA5 for 2014-10-05 12:00-23:00 UTC at the GoAmazon T3 site
# (-3.2, -60.6) and writes goamazon_ls2d_input.nc with three groups:
#
#   radiation      time-mean ERA5 background column (for RRTMGP above the
#                  LES domain) + o3/h2o on a fine reference z grid
#   soil           ERA5 soil state at 12:00 UTC (simulation start)
#   era5_profiles  hourly thl/qt/u/v on the reference z grid, used by
#                  goamazon_input.py to pad IOP profiles above ~110 hPa
#
# The reference z grid here is deliberately finer than any LES candidate
# grid (dz=50 m to 26 km) so the output is independent of the eventual
# LES ktot choice; goamazon_input.py interpolates to the actual grid.
#
# Output goes to $SCRATCH/GOAMAZON_LES/shared_data/ (data on scratch,
# code in the repo); symlink into the case data/ dir afterwards:
#
#   ln -sf $SCRATCH/GOAMAZON_LES/shared_data/goamazon_ls2d_input.nc \
#       /global/homes/m/mpowell/repos/microhh/cases/goamazon/data/
#
# Re-run friendly: LS2D raises SystemExit while CDS requests are queued;
# simply re-run once the download completes.
#

import sys
sys.path.insert(0, '/global/homes/m/mpowell/repos/LS2D')

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import netCDF4 as nc

import ls2d

# Constants
Rd = 287.04
Rv = 461.5
ep = Rd/Rv

# NOTE: central lat/lon are snapped to the 0.25-degree ERA5 grid (the T3 site
# is at -3.2, -60.6). Off-grid area bounds make the CDS surface and
# model-level datasets round to different grid-point counts, which crashes
# LS2D's Read_era5 with an IndexError.
settings = {
    'central_lat' : -3.25,
    'central_lon' : -60.5,
    'area_size'   : 1,
    'case_name'   : 'goamazon',
    'era5_path'   : '/pscratch/sd/m/mpowell/LS2D_ERA5',
    'start_date'  : datetime(2014, 10, 5, 12),
    'end_date'    : datetime(2014, 10, 5, 23),
    'write_log'   : False,
    'data_source' : 'CDS',
}

# Fixed background gas concentrations for RRTMGP (2014 global means)
co2 = 397.e-6
ch4 = 1822.e-9
n2o = 327.e-9
n2  = 0.7808
o2  = 0.2095

SCRATCH  = Path(os.environ.get('SCRATCH', '/pscratch/sd/m/mpowell'))
OUT_DIR  = SCRATCH / 'GOAMAZON_LES' / 'shared_data'
OUT_FILE = OUT_DIR / 'goamazon_ls2d_input.nc'

# Fine reference grid, finer than any LES candidate (dz=50 m to 26 km)
z_ref = np.arange(25., 26000., 50.)

# -----------------------------------------------------------------------
# Download (SystemExit while CDS request is queued -- re-run later)
# -----------------------------------------------------------------------
ls2d.download_era5(settings)

era = ls2d.Read_era5(settings)
era.calculate_forcings(n_av=0, method='2nd')

les_input = era.get_les_input(z_ref)

# Remove top ERA5 level to keep pressure above RRTMGP minimum (as in CASS):
les_input = les_input.sel(lay=slice(0, 135), lev=slice(0, 136))

# -----------------------------------------------------------------------
# Report site surface/vegetation properties for the ini cross-check
# -----------------------------------------------------------------------
print('--- ERA5 site properties (cross-check goamazon_2stream/raytracer.ini) ---')
print(f'  type_high_veg : {int(les_input.type_high_veg):d} (IFS; 6 = evergreen broadleaf)')
print(f'  type_low_veg  : {int(les_input.type_low_veg):d}')
print(f'  c_high_veg    : {les_input.c_high_veg.values.mean():.3f}')
print(f'  lai_high_veg  : {les_input.lai_high_veg.values.mean():.2f}')
print(f'  z0m           : {les_input.z0m.values.mean():.3f} m')
print(f'  z0h           : {les_input.z0h.values.mean():.4f} m')
print(f'  soil type     : {int(les_input.type_soil):d} (Fortran indexing)')
print(f'  skin T @12UTC : {float(les_input.ts[0]):.2f} K')

# -----------------------------------------------------------------------
# Write output NetCDF
# -----------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
f = nc.Dataset(OUT_FILE, 'w', datamodel='NETCDF4', clobber=True)


def add_var(grp, name, dims, data):
    v = grp.createVariable(name, 'f8', dims)
    v[:] = data
    return v


# ------ radiation group: time-mean background column ------
rad = f.createGroup('radiation')
rad.createDimension('lay', les_input.sizes['lay'])
rad.createDimension('lev', les_input.sizes['lev'])
rad.createDimension('z_ref', z_ref.size)

for name in ('z_lay', 'p_lay', 't_lay', 'o3_lay', 'h2o_lay'):
    add_var(rad, name.replace('_lay', '') if name in ('o3_lay', 'h2o_lay') else name,
            ('lay',), les_input[name].mean('time').values)
for name in ('z_lev', 'p_lev', 't_lev'):
    add_var(rad, name, ('lev',), les_input[name].mean('time').values)

# o3 is in ppmv from LS2D; convert to vmr as in CASS
rad['o3'][:] = rad['o3'][:] * 1e-6

# o3/h2o also on the reference z grid for the LES-domain init profiles
qt_mean = les_input['qt'].mean('time').values
add_var(rad, 'z_ref_grid', ('z_ref',), z_ref)
add_var(rad, 'o3_z', ('z_ref',),
        np.interp(z_ref, rad['z_lay'][:], rad['o3'][:]))
add_var(rad, 'h2o_z', ('z_ref',), qt_mean / (ep - ep*qt_mean))

for name, val in (('co2', co2), ('ch4', ch4), ('n2o', n2o), ('n2', n2), ('o2', o2)):
    v = rad.createVariable(name, 'f8')
    v[:] = val

# ------ soil group: state at 12:00 UTC (time index 0) ------
soil = f.createGroup('soil')
z_soil = les_input['zs'][::-1].values
soil.createDimension('z', z_soil.size)
add_var(soil, 'z', ('z',), z_soil)
add_var(soil, 'theta_soil', ('z',), les_input['theta_soil'][0, ::-1].values)
add_var(soil, 't_soil', ('z',), les_input['t_soil'][0, ::-1].values)
add_var(soil, 'index_soil', ('z',),
        np.ones_like(z_soil) * (les_input['type_soil'].values - 1))
# Site is evergreen broadleaf forest: use the ERA5 high-vegetation root profile
add_var(soil, 'root_frac', ('z',), les_input['root_frac_high_veg'][::-1].values)

# ------ era5_profiles group: hourly profiles on the reference grid ------
prof = f.createGroup('era5_profiles')
prof.createDimension('time', les_input.sizes['time'])
prof.createDimension('z', z_ref.size)
add_var(prof, 'time_sec', ('time',), les_input['time_sec'].values)
add_var(prof, 'z', ('z',), z_ref)
for name in ('thl', 'qt', 'u', 'v'):
    add_var(prof, name, ('time', 'z'), les_input[name].values)

f.description = ('GoAmazon single-pulse LS2D/ERA5 input: 2014-10-05 12-23 UTC '
                 'at (-3.2, -60.6). time_sec is seconds since 12:00 UTC.')
f.close()

print(f'\nSuccessfully wrote {OUT_FILE}')
