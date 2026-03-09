"""
CASS (Composite Atmospheric Sounding for Shallow-cumulus) case
Data processing script for MicroHH

Based on long-term ARM SGP observations
Location: 36.5N, 97.5W (ARM SGP central facility)
Day: July 24 (DOY 205)
"""

import sys
sys.path.insert(0, '/global/homes/m/mpowell/repos/LS2D')

import argparse
import numpy as np
import netCDF4 as nc
import xarray as xr
from datetime import datetime
import ls2d as ls2d_pkg

from cass_utils import read_composite_days

parser = argparse.ArgumentParser(description='Generate cass_input.nc for MicroHH')
parser.add_argument('--zero-winds', action='store_true',
                    help='Set u=v=0 in init and u_nudge=v_nudge=0 in timedep')
parser.add_argument('--theta-nudge', type=float, default=None,
                    help='Add uniform theta_nudge[4] to soil group (e.g. 0.3)')
args = parser.parse_args()

float_type = "f8"

def add_nc_var(name, dims, nc, data):
    """
    Add NetCDF variable to `nc` file or group.
    """
    if name not in nc.variables:
        if dims is None:
            var = nc.createVariable(name, np.float64)
        else:
            var = nc.createVariable(name, np.float64, dims)
        var[:] = data

def add_nc_dim(name, size, nc):
    """
    Add NetCDF dimension, if it does not already exist.
    """
    if name not in nc.dimensions:
        nc.createDimension(name, size)

# Get number of vertical levels and size from .ini file
with open('cass.ini') as f:
    in_grid_section = False
    for line in f:
        line_stripped = line.strip()
        if line_stripped.startswith('['):
            in_grid_section = (line_stripped == '[grid]')
        if in_grid_section:
            key = line.split('=')[0].strip()
            if key == 'ktot':
                kmax = int(line.split('=')[1])
            if key == 'zsize':
                zsize = float(line.split('=')[1])

dz = zsize / kmax

# Set the height
z = np.linspace(0.5*dz, zsize-0.5*dz, kmax)

# Read atmospheric sounding data (initial profiles)
print("Reading atmospheric sounding data...")
snd_data = np.loadtxt('cass_snd.txt', skiprows=2, max_rows=48)

# snd format: z[m], p[mb], tp[K], q[g/kg], u[m/s], v[m/s]
z_snd = snd_data[:, 0]
p_snd = snd_data[:, 1]
tp_snd = snd_data[:, 2]  # Potential Temperature (K)
q_snd = snd_data[:, 3]   # Specific humidity (g/kg)
u_snd = snd_data[:, 4]   # U wind component
v_snd = snd_data[:, 5]   # V wind component

# Interpolate initial profiles to model grid
thl = np.interp(z, z_snd, tp_snd)  # assuming unsaturated initial profile
qt = np.interp(z, z_snd, q_snd) / 1000.  # Convert g/kg to kg/kg
u = np.interp(z, z_snd, u_snd)
v = np.interp(z, z_snd, v_snd)

# Read surface flux data (time-dependent)
print("Reading surface flux data...")
sfc_data = np.loadtxt('cass_sfc.txt', skiprows=1)

# sfc format: day, sst(K), H(W/m2), LE(W/m2), TAU(m2/s2)
day_sfc = sfc_data[:, 0]
sst = sfc_data[:, 1]
H = sfc_data[:, 2]     # Sensible heat flux (W/m2)
LE = sfc_data[:, 3]    # Latent heat flux (W/m2)
TAU = sfc_data[:, 4]   # Surface stress (not used for prescribed flux)

# Convert day of year to seconds since start (assume start is day 205.5)
start_day = 205.5
time_surface = (day_sfc - start_day) * 86400.  # Convert days to seconds

# Calculate surface fluxes in MicroHH units (kinematic)
Rd = 287.04          # Gas constant for dry air (J/kg/K)
cp = 1005.0          # Specific heat at constant pressure (J/kg/K)
Lv = 2.5e6           # Latent heat of vaporization (J/kg)
p0 = 97300.0         # Surface pressure (Pa) - from CASS data
T_sfc = thl[0]       # Surface temperature (K)
q_sfc = qt[0]        # Surface specific humidity (kg/kg)

# Surface density
rho = p0 / (Rd * T_sfc * (1. + 0.61 * q_sfc))

# Convert to kinematic fluxes
thl_flux = H / (rho * cp)      # K m/s
qt_flux = LE / (rho * Lv)      # kg/kg m/s

# Read large-scale forcing data (time-dependent profiles)
print("Reading large-scale forcing data...")

# Parse the lsf file which has time blocks
with open('cass_lsf.txt', 'r') as f:
    lines = f.readlines()

# Find time blocks
time_blocks = []
current_block = None

for i, line in enumerate(lines):
    if 'day, levels, pres0' in line:
        time_line = line.strip().split()
        day_lsf = float(time_line[0].rstrip(','))
        nlevels = int(time_line[1].rstrip(','))

        # Create new block
        current_block = {
            'day': day_lsf,
            'nlevels': nlevels,
            'start_line': i + 1,
            'data': []
        }
        time_blocks.append(current_block)
    elif current_block is not None and len(current_block['data']) < current_block['nlevels']:
        # Parse data line
        parts = line.strip().split()
        if len(parts) >= 7:
            current_block['data'].append([float(x) for x in parts[:7]])

# Convert to numpy arrays
n_times = len(time_blocks)
n_levels_max = max([b['nlevels'] for b in time_blocks])

time_ls = np.array([b['day'] for b in time_blocks])
time_ls = (time_ls - start_day) * 86400.  # Convert to seconds

# Initialize forcing arrays
thlls = np.zeros((n_times, kmax))
qtls = np.zeros((n_times, kmax))
uls = np.zeros((n_times, kmax))
vls = np.zeros((n_times, kmax))
wls = np.zeros((n_times, kmax))

for t, block in enumerate(time_blocks):
    data = np.array(block['data'])
    z_lsf = data[:, 0]
    tls = data[:, 2]   # Temperature tendency (K/s)
    qls = data[:, 3]   # Moisture tendency (kg/kg/s)
    uls_data = data[:, 4]  # U wind (m/s)
    vls_data = data[:, 5]  # V wind (m/s)
    wls_data = data[:, 6]  # Subsidence velocity (m/s)

    # Interpolate to model grid
    thlls[t, :] = np.interp(z, z_lsf, tls)
    qtls[t, :] = np.interp(z, z_lsf, qls)
    uls[t, :] = np.interp(z, z_lsf, uls_data)
    vls[t, :] = np.interp(z, z_lsf, vls_data)
    wls[t, :] = np.interp(z, z_lsf, wls_data)


# -----------------------------------------------------------------------
# Read CAMS aerosol data: composite-mean over 2003-2009 composite days.
# Aerosols and gases are averaged over all hours (10-23 UTC) and all days.
# CAMS EAC4 is available from 2003 onwards; earlier years are excluded.
# -----------------------------------------------------------------------
print("Reading CAMS aerosol data (composite 2003-2009)...")

cams_vars = {
    'eac4_ml': [
        'dust_aerosol_0.03-0.55um_mixing_ratio',
        'dust_aerosol_0.55-0.9um_mixing_ratio',
        'dust_aerosol_0.9-20um_mixing_ratio',
        'hydrophilic_black_carbon_aerosol_mixing_ratio',
        'hydrophilic_organic_matter_aerosol_mixing_ratio',
        'hydrophobic_black_carbon_aerosol_mixing_ratio',
        'hydrophobic_organic_matter_aerosol_mixing_ratio',
        'sea_salt_aerosol_0.03-0.5um_mixing_ratio',
        'sea_salt_aerosol_0.5-5um_mixing_ratio',
        'sea_salt_aerosol_5-20um_mixing_ratio',
        'sulphate_aerosol_mixing_ratio',
        'specific_humidity',
        'temperature'],
    'eac4_sfc': ['surface_pressure'],
}

base_cams_settings = {
    'central_lat' : 36.5,
    'central_lon' : -97.5,
    'area_size'   : 2,
    'case_name'   : 'cass',
    'cams_path'   : '/pscratch/sd/m/mpowell/LS2D_CAMS',
    'cdsapirc'    : '/global/homes/m/mpowell/.cdsapirc',
    'write_log'   : False,
    'data_source' : 'CDS',
    'ntasks'      : 1,
}

aerosol_names = [f'aermr{i:02d}' for i in range(1, 12)]

# Composite days with available CAMS data (2003-2009)
cams_days = read_composite_days(year_min=2003, year_max=2009)

n_cams = 0
aer_les_sum  = {}   # time-and-day mean on LES z grid: {name: array(z)}
aer_lay_sum  = {}   # time-and-day mean on CAMS model levels: {name: array(lay)}
z_lay_cams_sum = None

for dt in cams_days:
    cams_settings = dict(base_cams_settings)
    cams_settings['start_date'] = datetime(year=dt.year, month=dt.month, day=dt.day, hour=10)
    cams_settings['end_date']   = datetime(year=dt.year, month=dt.month, day=dt.day, hour=23)

    try:
        cams = ls2d_pkg.Read_cams(cams_settings, variables=cams_vars)
        cams_les = cams.get_les_input(z)

        # Clip aerosol mixing ratios to non-negative values
        for name in aerosol_names:
            cams_les[name]           = np.maximum(cams_les[name], 0.)
            cams_les[f'{name}_lay']  = np.maximum(cams_les[f'{name}_lay'], 0.)

        # Time-mean for this day (axis 0 = time)
        zlay_day = cams_les.z_lay.mean(axis=0).values

        if n_cams == 0:
            z_lay_cams_sum = zlay_day.copy()
            for name in aerosol_names:
                aer_les_sum[name]          = cams_les[name].mean(axis=0).values.copy()
                aer_lay_sum[f'{name}_lay'] = cams_les[f'{name}_lay'].mean(axis=0).values.copy()
        else:
            z_lay_cams_sum += zlay_day
            for name in aerosol_names:
                aer_les_sum[name]          += cams_les[name].mean(axis=0).values
                aer_lay_sum[f'{name}_lay'] += cams_les[f'{name}_lay'].mean(axis=0).values

        n_cams += 1
        print(f'  CAMS: {dt.strftime("%Y-%m-%d")} ({n_cams}/{len(cams_days)})')

    except Exception as e:
        print(f'  Warning: CAMS read failed for {dt.strftime("%Y-%m-%d")}: {e}')
        continue

if n_cams == 0:
    raise RuntimeError('No CAMS composite days were successfully read.')

print(f'Averaged CAMS aerosols over {n_cams} days')

# Composite means
z_lay_cams = z_lay_cams_sum / n_cams
for name in aerosol_names:
    aer_les_sum[name]          /= n_cams
    aer_lay_sum[f'{name}_lay'] /= n_cams


# -----------------------------------------------------------------------
# Save all the input data to NetCDF
# -----------------------------------------------------------------------
print("Saving to NetCDF...")
nc_file = nc.Dataset("cass_input.nc", mode="w", datamodel="NETCDF4", clobber=True)

add_nc_dim("z", kmax, nc_file)
add_nc_var("z", ("z",), nc_file, z)

# Create a group called "init" for the initial profiles
nc_group_init = nc_file.createGroup("init")

add_nc_var('z', ("z",), nc_group_init, z)
add_nc_var("thl", ("z",), nc_group_init, thl)
add_nc_var("qt", ("z",), nc_group_init, qt)
add_nc_var("u", ("z",), nc_group_init, np.zeros(kmax) if args.zero_winds else u)
add_nc_var("v", ("z",), nc_group_init, np.zeros(kmax) if args.zero_winds else v)
add_nc_var("nudgefac", ("z",), nc_group_init, np.ones(kmax)/10800)

# Aerosol initial profiles: composite-mean on LES z grid
for name in aerosol_names:
    add_nc_var(name, ('z',), nc_group_init, aer_les_sum[name])

# Create a group called "timedep" for the time-dependent forcings
nc_group_timedep = nc_file.createGroup("timedep")
add_nc_dim("time_surface", time_surface.size, nc_group_timedep)
add_nc_dim("time_ls", time_ls.size, nc_group_timedep)
add_nc_dim("z", kmax, nc_group_timedep)

# Surface fluxes
add_nc_var("time_surface", ("time_surface",), nc_group_timedep, time_surface)
add_nc_var("thl_sbot", ("time_surface",), nc_group_timedep, thl_flux)
add_nc_var("qt_sbot", ("time_surface",), nc_group_timedep, qt_flux)

# Large-scale forcings
add_nc_var("time_ls", ("time_ls",), nc_group_timedep, time_ls)
add_nc_var("thl_ls", ("time_ls", "z"), nc_group_timedep, thlls)
add_nc_var("qt_ls", ("time_ls", "z"), nc_group_timedep, qtls)
add_nc_var("w_ls", ("time_ls", "z"), nc_group_timedep, wls)
add_nc_var("u_nudge", ("time_ls", "z"), nc_group_timedep, np.zeros_like(uls) if args.zero_winds else uls)
add_nc_var("v_nudge", ("time_ls", "z"), nc_group_timedep, np.zeros_like(vls) if args.zero_winds else vls)


# Radiation variables on LES grid.
ls2d_radiation = xr.open_dataset('cass_ls2d_input.nc', group = 'radiation')
o3_z = np.interp(z, ls2d_radiation.z_lay.values, ls2d_radiation.o3.values)
h2o_z = np.interp(z, ls2d_radiation.z_lay.values, ls2d_radiation.h2o.values)
# add radiation to init group
add_nc_var('h2o', ('z'), nc_group_init, h2o_z)
add_nc_var('o3',  ('z'), nc_group_init, o3_z)
add_nc_var('co2', (), nc_group_init, ls2d_radiation.co2.values)
add_nc_var('ch4', (), nc_group_init, ls2d_radiation.ch4.values)
add_nc_var('n2o', (), nc_group_init, ls2d_radiation.n2o.values)
add_nc_var('n2', (), nc_group_init, ls2d_radiation.n2.values)
add_nc_var('o2', (), nc_group_init, ls2d_radiation.o2.values)

# and to radiation group
nc_rad = nc_file.createGroup('radiation')
add_nc_dim('lay', ls2d_radiation.sizes['lay'], nc_rad)
add_nc_dim('lev', ls2d_radiation.sizes['lev'], nc_rad)

# Layer variables
add_nc_var('z_lay', ('lay',), nc_rad, ls2d_radiation.z_lay.values)
add_nc_var('p_lay', ('lay',), nc_rad, ls2d_radiation.p_lay.values)
add_nc_var('t_lay', ('lay',), nc_rad, ls2d_radiation.t_lay.values)
add_nc_var('o3', ('lay',), nc_rad, ls2d_radiation.o3.values)
add_nc_var('h2o', ('lay',), nc_rad, ls2d_radiation.h2o.values)

# Level variables
add_nc_var('z_lev', ('lev',), nc_rad, ls2d_radiation.z_lev.values)
add_nc_var('p_lev', ('lev',), nc_rad, ls2d_radiation.p_lev.values)
add_nc_var('t_lev', ('lev',), nc_rad, ls2d_radiation.t_lev.values)

# Scalar gas concentrations
add_nc_var('co2', (), nc_rad, ls2d_radiation.co2.values)
add_nc_var('ch4', (), nc_rad, ls2d_radiation.ch4.values)
add_nc_var('n2o', (), nc_rad, ls2d_radiation.n2o.values)
add_nc_var('n2', (), nc_rad, ls2d_radiation.n2.values)
add_nc_var('o2', (), nc_rad, ls2d_radiation.o2.values)

# Aerosols on radiation levels: interpolate composite-mean CAMS profiles to ERA5 z_lay
z_lay_era5 = ls2d_radiation.z_lay.values
sort_idx = np.argsort(z_lay_cams)
z_lay_cams_sorted = z_lay_cams[sort_idx]
for name in aerosol_names:
    aer_cams_sorted = aer_lay_sum[f'{name}_lay'][sort_idx]
    aer_era5 = np.interp(z_lay_era5, z_lay_cams_sorted, aer_cams_sorted)
    add_nc_var(name, ('lay',), nc_rad, np.maximum(aer_era5, 0.))


# land surface model: composite-mean soil from cass_ls2d_input.nc
ls2d_soil = xr.open_dataset('cass_ls2d_input.nc', group = 'soil')
nc_soil = nc_file.createGroup('soil')
add_nc_dim('z', ls2d_soil.sizes['z'], nc_soil)
add_nc_var('z', ('z'), nc_soil, ls2d_soil.z.values)
add_nc_var('theta_soil', ('z'), nc_soil, ls2d_soil.theta_soil.values)
add_nc_var('t_soil', ('z'), nc_soil, ls2d_soil.t_soil.values)
add_nc_var('index_soil', ('z'), nc_soil, ls2d_soil.index_soil.values)
add_nc_var('root_frac', ('z'), nc_soil, ls2d_soil.root_frac.values)

if args.theta_nudge is not None:
    add_nc_var('theta_nudge', ('z'), nc_soil, np.full(ls2d_soil.sizes['z'], args.theta_nudge))

nc_file.close()

print(f"Successfully created cass_input.nc")
print(f"   Initial profiles at z = {z[0]:.1f} to {z[-1]:.1f} m")
print(f"   Surface fluxes from t = {time_surface[0]:.0f} to {time_surface[-1]:.0f} s")
print(f"   Large-scale forcings from t = {time_ls[0]:.0f} to {time_ls[-1]:.0f} s")
print(f"   Aerosols: composite mean over {n_cams} CAMS days (2003-2009)")
