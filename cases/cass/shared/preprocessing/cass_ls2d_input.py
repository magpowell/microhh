#
# This file is ADAPTED from LS2D.
#
# Copyright (c) 2017-2024 Wageningen University & Research
# Author: Bart van Stratum (WUR)
#
# LS2D is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# LS2D is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with LS2D.  If not, see <http://www.gnu.org/licenses/>.
#

# Python modules
import sys
sys.path.insert(0, '/global/homes/m/mpowell/repos/LS2D')

from datetime import datetime

# Third party modules
import numpy as np
import matplotlib
matplotlib.use('Agg')
import ls2d
import LS2D.examples.microhh.microhh_ls2d_tools as mlt

from cass_utils import read_composite_days

# Constants
Rd = 287.04
Rv = 461.5
ep = Rd/Rv

#
# Download ERA5 and generate composite LES initialisation and forcings.
# Loops over all shallow-cumulus composite days from shcu_sgp_summer_97to09.nc
# (dscu == 1, years 1997-2009).
#
# Soil:      composite-mean ERA5 soil state at 10 UTC (simulation start hour)
# Radiation: composite-mean ERA5 state averaged over hours 10-23 UTC and all days
#

base_settings = {
    'central_lat' : 36.5,
    'central_lon' : -97.5,
    'area_size'   : 1,
    'case_name'   : 'cass',
    'era5_path'   : '/pscratch/sd/m/mpowell/LS2D_ERA5',
    'write_log'   : False,
    'data_source' : 'CDS',
}

# Fixed background concentrations RRTMGP:
co2 = 380.e-6
ch4 = 1774.e-9
n2o = 319.e-9
n2  = 0.7808
o2  = 0.2095

# Root fraction coefficients (see IFS documentation):
a_r = 10.739
b_r = 2.608

# Nudging time scale atmosphere
tau_nudge = 10800

# Define vertical grid LES (same for all composite days):
grid = ls2d.grid.Grid_linear_stretched(kmax=176, dz0=20, alpha=0.009)
grid.plot()

# -----------------------------------------------------------------------
# Loop over composite days and accumulate ERA5 averages
# -----------------------------------------------------------------------
composite_days = read_composite_days()  # all years 1997-2009

n_days         = 0
n_pending      = 0
MAX_PENDING    = 4  # submit up to this many CDS requests before stopping
soil_theta_sum = None
soil_t_sum     = None
soil_type      = None
z_soil         = None
qt_sum         = None
o3_sum         = None
rad_sum        = {}

for dt in composite_days:
    settings = dict(base_settings)
    settings['start_date'] = datetime(year=dt.year, month=dt.month, day=dt.day, hour=10)
    settings['end_date']   = datetime(year=dt.year, month=dt.month, day=dt.day, hour=23)

    # Download required ERA5 files (cached by LS2D; skips existing files).
    # LS2D raises SystemExit when a CDS request is submitted but not yet ready,
    # or when an existing request is still queued/running.
    # We allow up to MAX_PENDING days' requests in-flight simultaneously before
    # stopping — re-run once CDS delivers the files.
    try:
        ls2d.download_era5(settings)
    except SystemExit:
        n_pending += 1
        print(f'  CDS request pending for {dt.strftime("%Y-%m-%d")} '
              f'({n_days} days processed, {n_pending}/{MAX_PENDING} pending).')
        if n_pending >= MAX_PENDING:
            print(f'  Reached {MAX_PENDING} pending requests. Re-run once CDS delivers files.')
            break
        continue
    except Exception as e:
        print(f'  Warning: download failed for {dt.strftime("%Y-%m-%d")}: {e}')
        continue

    try:
        # Read ERA5 data and calculate derived properties (thl, etc.):
        era = ls2d.Read_era5(settings)
        era.calculate_forcings(n_av=0, method='2nd')

        # Interpolate ERA5 onto LES grid:
        les_input = era.get_les_input(grid.z)

        # Remove top ERA5 level to keep pressure above RRTMGP minimum:
        les_input = les_input.sel(lay=slice(0, 135), lev=slice(0, 136))

        # ------ Soil: extract at 10 UTC (time index 0) ------
        st = les_input['theta_soil'][0, ::-1].values
        ts = les_input['t_soil'][0, ::-1].values

        # ------ Radiation / atmosphere: time-mean over 10-23 UTC ------
        qt_day  = les_input['qt'].mean(axis=0).values
        o3_day  = les_input['o3'].mean(axis=0).values
        zlay    = les_input['z_lay'].mean(axis=0).values
        zlev    = les_input['z_lev'].mean(axis=0).values
        play    = les_input['p_lay'].mean(axis=0).values
        plev    = les_input['p_lev'].mean(axis=0).values
        tlay    = les_input['t_lay'].mean(axis=0).values
        tlev    = les_input['t_lev'].mean(axis=0).values
        o3lay   = les_input['o3_lay'].mean(axis=0).values
        h2olay  = les_input['h2o_lay'].mean(axis=0).values

        if n_days == 0:
            # Initialise accumulators from first successful day
            z_soil         = les_input['zs'][::-1].values
            soil_type      = les_input['type_soil'].values
            soil_theta_sum = st.copy()
            soil_t_sum     = ts.copy()
            qt_sum         = qt_day.copy()
            o3_sum         = o3_day.copy()
            rad_sum = {
                'z_lay':   zlay.copy(),
                'z_lev':   zlev.copy(),
                'p_lay':   play.copy(),
                'p_lev':   plev.copy(),
                't_lay':   tlay.copy(),
                't_lev':   tlev.copy(),
                'o3_lay':  o3lay.copy(),
                'h2o_lay': h2olay.copy(),
            }
        else:
            soil_theta_sum += st
            soil_t_sum     += ts
            qt_sum         += qt_day
            o3_sum         += o3_day
            rad_sum['z_lay']   += zlay
            rad_sum['z_lev']   += zlev
            rad_sum['p_lay']   += play
            rad_sum['p_lev']   += plev
            rad_sum['t_lay']   += tlay
            rad_sum['t_lev']   += tlev
            rad_sum['o3_lay']  += o3lay
            rad_sum['h2o_lay'] += h2olay

        n_days += 1
        print(f'  Processed {dt.strftime("%Y-%m-%d")} ({n_days}/{len(composite_days)})')

    except Exception as e:
        print(f'  Warning: processing failed for {dt.strftime("%Y-%m-%d")}: {e}')
        continue

if n_days == 0:
    if n_pending > 0:
        print(f'  No days processed yet — {n_pending} CDS request(s) submitted. Re-run once complete.')
        sys.exit(1)
    raise RuntimeError(
        'No composite days were successfully processed. '
        'Wait for CDS downloads to complete and re-run.'
    )

print(f'\nAveraging over {n_days} composite days...')

# -----------------------------------------------------------------------
# Compute composite means
# -----------------------------------------------------------------------
theta_soil = soil_theta_sum / n_days
t_soil     = soil_t_sum     / n_days
qt_les     = qt_sum         / n_days
o3_les     = o3_sum         / n_days
h2o_les    = qt_les / (ep - ep * qt_les)

for k in rad_sum:
    rad_sum[k] /= n_days

soil_index_val = soil_type - 1
soil_index     = np.ones_like(z_soil) * soil_index_val
root_frac      = mlt.calc_root_frac(z_soil, a_r, b_r)
nudge_fac      = np.ones(grid.kmax) / tau_nudge

# -----------------------------------------------------------------------
# Write NetCDF input file for MicroHH
# -----------------------------------------------------------------------
init_profiles = {'z': grid.z}

radiation = {
    'z_lay': rad_sum['z_lay'],
    'z_lev': rad_sum['z_lev'],
    'p_lay': rad_sum['p_lay'],
    'p_lev': rad_sum['p_lev'],
    't_lay': rad_sum['t_lay'],
    't_lev': rad_sum['t_lev'],
    'o3':    rad_sum['o3_lay'] * 1e-6,
    'h2o':   rad_sum['h2o_lay'],
    'co2': co2,
    'ch4': ch4,
    'n2o': n2o,
    'n2':  n2,
    'o2':  o2,
}

soil = {
    'z':           z_soil,
    'theta_soil':  theta_soil,
    't_soil':      t_soil,
    'index_soil':  soil_index,
    'root_frac':   root_frac,
}

mlt.write_netcdf_input(
    'cass_ls2d', 'f8', init_profiles, radiation=radiation, soil=soil)

# Tag the file with the number of composite days so we can verify later
import netCDF4 as _nc
with _nc.Dataset('cass_ls2d_input.nc', 'a') as _f:
    _f.n_composite_days = n_days
    _f.composite_day_range = '1997-2009 (dscu==1 from shcu_sgp_summer_97to09.nc)'

print(f'Successfully wrote cass_ls2d_input.nc (composite of {n_days} days)')

if n_pending > 0:
    print(f'  Note: {n_pending} CDS request(s) still pending. Re-run to process remaining days.')
    sys.exit(1)
