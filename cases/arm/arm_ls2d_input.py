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
from datetime import datetime
import sys

# Third party modules
import numpy as np
import ls2d
import LS2D.examples.microhh.microhh_ls2d_tools as mlt

# Constants
Rd = 287.04
Rv = 461.5
ep = Rd/Rv

#
# Download ERA5 and generate LES initialisation and forcings
#
settings = {
    'central_lat' : 36.5,
    'central_lon' : -97.5,
    'area_size'   : 1,
    'case_name'   : 'arm97',
    'era5_path'   : '/pscratch/sd/m/mpowell/LS2D_ERA5', 
    'start_date'  : datetime(year=1997, month=6, day=21, hour=10),
    'end_date'    : datetime(year=1997, month=6, day=21, hour=23),
    'write_log'   : True,
    'data_source' : 'CDS'
    }

# Download required ERA5 files:
ls2d.download_era5(settings)

# Read ERA5 data, and calculate derived properties (thl, etc.):
era = ls2d.Read_era5(settings)

# Calculate initial conditions and large-scale forcings for LES:
era.calculate_forcings(n_av=0, method='2nd')

# Define vertical grid LES:
grid = ls2d.grid.Grid_linear_stretched(kmax=176, dz0=20, alpha=0.009)
grid.plot()

# Interpolate ERA5 variables and forcings onto LES grid.
# In addition, `get_les_input` returns additional variables needed to init LES.
les_input = era.get_les_input(grid.z)

# Remove top ERA5 level, to ensure that pressure stays
# above the minimum reference pressure in RRTMGP
les_input = les_input.sel(lay=slice(0,135), lev=slice(0,136))

#
# MicroHH specific initialisation
#
# Settings:
# ------------------
# Root fraction coefficients (see IFS documentation):
a_r = 10.739
b_r = 2.608

# Nudging time scale atmosphere
tau_nudge = 10800
# ------------------

# Fixed background concentrations RRTMGP:
co2 = 348.e-6
ch4 = 1650.e-9
n2o = 306.e-9
n2  = 0.7808
o2  = 0.2095

# # Mean radiation profiles on LES grid:
qt_les = les_input['qt'].mean(axis=0)
o3_les = les_input['o3'].mean(axis=0)

# # Conversion moisture from mass to volume mixing ratio
h2o_les = qt_les / (ep - ep*qt_les)

# ## Soil
z_soil     = les_input['zs'][::-1].values
soil_index = les_input['type_soil'].values-1
soil_index = np.ones_like(z_soil)*soil_index
root_frac  = mlt.calc_root_frac(z_soil, a_r, b_r)

# Nudge factor
nudge_fac = np.ones(grid.kmax) / tau_nudge

#
# Write NetCDF input file for MicroHH
#

init_profiles = {
        'z': grid.z}

radiation  = {
        'z_lay': les_input['z_lay'].mean(axis=0),
        'z_lev': les_input['z_lev'].mean(axis=0),
        'p_lay': les_input['p_lay'].mean(axis=0),
        'p_lev': les_input['p_lev'].mean(axis=0),
        't_lay': les_input['t_lay'].mean(axis=0),
        't_lev': les_input['t_lev'].mean(axis=0),
        'o3': les_input['o3_lay'].mean(axis=0)*1e-6,
        'h2o': les_input['h2o_lay'].mean(axis=0),
        'co2': co2,
        'ch4': ch4,
        'n2o': n2o,
        'n2': n2,
        'o2': o2}

soil = {
        'z': z_soil,
        'theta_soil': les_input['theta_soil'][0,::-1],
        't_soil': les_input['t_soil'][0,::-1],
        'index_soil': soil_index,
        'root_frac': root_frac}

mlt.write_netcdf_input(
        'arm_ls2d', 'f8', init_profiles, radiation = radiation, soil = soil)

