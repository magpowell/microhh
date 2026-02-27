"""
Save per-day ERA5 thl and qt profiles for all composite days.

Reads from the already-downloaded ERA5 cache (no new CDS requests).
Skips days whose ERA5 files have not been downloaded yet.

Output: cass_era5_profiles.nc
    Dimensions: day (up to 119), z (LES grid levels)
    Variables:
        thl  (day, z)  - liquid-water pot. temp. at 10 UTC [K]
        qt   (day, z)  - specific humidity     at 10 UTC [kg/kg]
        date (day)     - calendar date as YYYYMMDD integer
        z    (z)       - LES grid heights [m]

Usage:
    python save_era5_profiles.py
"""

import sys
sys.path.insert(0, '/global/homes/m/mpowell/repos/LS2D')

from datetime import datetime
import numpy as np
import netCDF4 as nc
import ls2d

from cass_utils import read_composite_days

# Same settings and grid as cass_ls2d_input.py
base_settings = {
    'central_lat' : 36.5,
    'central_lon' : -97.5,
    'area_size'   : 1,
    'case_name'   : 'cass',
    'era5_path'   : '/pscratch/sd/m/mpowell/LS2D_ERA5',
    'write_log'   : False,
    'data_source' : 'CDS',
}

import matplotlib
matplotlib.use('Agg')

grid = ls2d.grid.Grid_linear_stretched(kmax=176, dz0=20, alpha=0.009)

composite_days = read_composite_days()  # all years 1997-2009

thl_list  = []
qt_list   = []
date_list = []

for dt in composite_days:
    settings = dict(base_settings)
    settings['start_date'] = datetime(year=dt.year, month=dt.month, day=dt.day, hour=10)
    settings['end_date']   = datetime(year=dt.year, month=dt.month, day=dt.day, hour=23)

    try:
        # Read from cache only — raises if files aren't downloaded yet
        era = ls2d.Read_era5(settings)
        era.calculate_forcings(n_av=0, method='2nd')
        les_input = era.get_les_input(grid.z)

        # Extract at 10 UTC (time index 0)
        thl_list.append(les_input['thl'][0, :].values)
        qt_list.append( les_input['qt'][0,  :].values)
        date_list.append(int(dt.strftime('%Y%m%d')))

        print(f'  {dt.strftime("%Y-%m-%d")} ({len(thl_list)}/{len(composite_days)})')

    except Exception as e:
        print(f'  Skipping {dt.strftime("%Y-%m-%d")} (not yet cached): {e}')
        continue

n = len(thl_list)
if n == 0:
    raise RuntimeError('No ERA5 profiles available — run cass_ls2d_input.py first.')

thl_arr  = np.array(thl_list)   # (n, z)
qt_arr   = np.array(qt_list)    # (n, z)
date_arr = np.array(date_list)  # (n,)

# -----------------------------------------------------------------------
# Write NetCDF
# -----------------------------------------------------------------------
with nc.Dataset('cass_era5_profiles.nc', 'w') as ds:
    ds.description = (
        f'ERA5 thl and qt profiles at 10 UTC for {n}/{len(composite_days)} '
        'CASS composite days (dscu==1, 1997-2009). '
        'Profiles interpolated to the CASS LES grid.'
    )
    ds.createDimension('day', n)
    ds.createDimension('z',   grid.kmax)

    v = ds.createVariable('z', 'f4', ('z',))
    v[:] = grid.z
    v.units = 'm'
    v.long_name = 'LES grid height'

    v = ds.createVariable('date', 'i4', ('day',))
    v[:] = date_arr
    v.long_name = 'calendar date'
    v.units = 'YYYYMMDD'

    v = ds.createVariable('thl', 'f4', ('day', 'z'))
    v[:] = thl_arr
    v.units = 'K'
    v.long_name = 'liquid-water potential temperature at 10 UTC'

    v = ds.createVariable('qt', 'f4', ('day', 'z'))
    v[:] = qt_arr
    v.units = 'kg kg-1'
    v.long_name = 'specific humidity at 10 UTC'

print(f'\nWrote cass_era5_profiles.nc  ({n} days)')
if n < len(composite_days):
    print(f'Note: {len(composite_days)-n} days not yet cached — re-run after more ERA5 downloads complete.')
