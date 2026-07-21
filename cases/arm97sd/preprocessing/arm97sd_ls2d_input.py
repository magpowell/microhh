#
# ARM97 shallow-to-deep case (SGP, 27 June 1997): ERA5 (LS2D) preprocessing.
#
# Downloads ERA5 for 1997-06-27 11:00 UTC -> 1997-06-28 04:00 UTC at the ARM
# SGP central facility (site 36.605, -97.485; request snapped to the 0.25-deg
# grid at 36.5, -97.5 -- REQUIRED, off-grid bounds crash LS2D's Read_era5)
# and writes arm97sd_ls2d_input.nc with three groups:
#
#   radiation      time-mean ERA5 background column + o3/h2o on a fine
#                  reference z grid
#   soil           ERA5 soil state at 11:00 UTC (simulation starts 11:30)
#   era5_profiles  hourly thl/qt/u/v on the reference z grid, used by
#                  arm97sd_input.py to pad IOP profiles above ~115 hPa.
#                  time_sec is seconds since 1997-06-27 11:30 UTC (sim t=0).
#
# Output: $SCRATCH/ARM97SD_LES/shared_data/arm97sd_ls2d_input.nc
# Symlink into the case data/ dir afterwards. Re-run friendly (CDS queue).
#

import sys
sys.path.insert(0, '/global/homes/m/mpowell/repos/LS2D')

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import netCDF4 as nc

import ls2d

Rd = 287.04
Rv = 461.5
ep = Rd/Rv

settings = {
    'central_lat' : 36.5,
    'central_lon' : -97.5,
    'area_size'   : 1,
    'case_name'   : 'arm97sd',
    'era5_path'   : '/pscratch/sd/m/mpowell/LS2D_ERA5',
    'start_date'  : datetime(1997, 6, 27, 11),
    'end_date'    : datetime(1997, 6, 27, 23),  # day-28 00-04 UTC dropped: MARS lagged; padding holds last profile (dark decay phase)
    'write_log'   : False,
    'data_source' : 'CDS',
}

# Fixed background gas concentrations for RRTMGP (1997 global means)
co2 = 363.7e-6
ch4 = 1754.e-9
n2o = 314.e-9
n2  = 0.7808
o2  = 0.2095

T0_OFFSET = 1800.   # sim t=0 (11:30 UTC) minus LS2D start (11:00 UTC)

SCRATCH  = Path(os.environ.get('SCRATCH', '/pscratch/sd/m/mpowell'))
OUT_DIR  = SCRATCH / 'ARM97SD_LES' / 'shared_data'
OUT_FILE = OUT_DIR / 'arm97sd_ls2d_input.nc'

# Fine reference grid, finer than any LES candidate (dz=50 m to 26 km)
z_ref = np.arange(25., 26000., 50.)

ls2d.download_era5(settings)

era = ls2d.Read_era5(settings)
era.calculate_forcings(n_av=0, method='2nd')

les_input = era.get_les_input(z_ref)
les_input = les_input.sel(lay=slice(0, 135), lev=slice(0, 136))

print('--- ERA5 site properties (cross-check arm97sd_2stream/raytracer.ini) ---')
print(f'  type_low_veg  : {int(les_input.type_low_veg):d} (IFS; 1 = crops)')
print(f'  type_high_veg : {int(les_input.type_high_veg):d}')
print(f'  c_low_veg     : {les_input.c_low_veg.values.mean():.3f}')
print(f'  lai_low_veg   : {les_input.lai_low_veg.values.mean():.2f}')
print(f'  z0m           : {les_input.z0m.values.mean():.4f} m')
print(f'  z0h           : {les_input.z0h.values.mean():.5f} m')
print(f'  soil type     : {int(les_input.type_soil):d} (Fortran indexing)')
print(f'  skin T @11UTC : {float(les_input.ts[0]):.2f} K')

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

rad['o3'][:] = rad['o3'][:] * 1e-6   # ppmv -> vmr

qt_mean = les_input['qt'].mean('time').values
add_var(rad, 'z_ref_grid', ('z_ref',), z_ref)
add_var(rad, 'o3_z', ('z_ref',),
        np.interp(z_ref, rad['z_lay'][:], rad['o3'][:]))
add_var(rad, 'h2o_z', ('z_ref',), qt_mean / (ep - ep*qt_mean))

for name, val in (('co2', co2), ('ch4', ch4), ('n2o', n2o), ('n2', n2), ('o2', o2)):
    v = rad.createVariable(name, 'f8')
    v[:] = val

# ------ soil group: state at 11:00 UTC (time index 0) ------
soil = f.createGroup('soil')
z_soil = les_input['zs'][::-1].values
soil.createDimension('z', z_soil.size)
add_var(soil, 'z', ('z',), z_soil)
add_var(soil, 'theta_soil', ('z',), les_input['theta_soil'][0, ::-1].values)
add_var(soil, 't_soil', ('z',), les_input['t_soil'][0, ::-1].values)
add_var(soil, 'index_soil', ('z',),
        np.ones_like(z_soil) * (les_input['type_soil'].values - 1))
# SGP is cropland/grassland: use the ERA5 low-vegetation root profile
add_var(soil, 'root_frac', ('z',), les_input['root_frac_low_veg'][::-1].values)

# ------ era5_profiles group: hourly profiles, time in sim seconds ------
prof = f.createGroup('era5_profiles')
prof.createDimension('time', les_input.sizes['time'])
prof.createDimension('z', z_ref.size)
add_var(prof, 'time_sec', ('time',), les_input['time_sec'].values - T0_OFFSET)
add_var(prof, 'z', ('z',), z_ref)
for name in ('thl', 'qt', 'u', 'v'):
    add_var(prof, name, ('time', 'z'), les_input[name].values)

f.description = ('ARM97 shallow-to-deep (SGP 1997-06-27) LS2D/ERA5 input at '
                 '(36.5, -97.5). era5_profiles time_sec is seconds since '
                 '1997-06-27 11:30 UTC (sim t=0).')
f.close()

print(f'\nSuccessfully wrote {OUT_FILE}')
