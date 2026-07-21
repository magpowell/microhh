"""
CASS (Composite Atmospheric Sounding for Shallow-cumulus) case
Data processing script for MicroHH

Based on long-term ARM SGP observations
Location: 36.5N, 97.5W (ARM SGP central facility)
Day: July 24 (DOY 205)
"""

import argparse
import numpy as np
import netCDF4 as nc
import xarray as xr

parser = argparse.ArgumentParser(description='Generate cass_input.nc for MicroHH')
parser.add_argument('--zero-winds', action='store_true',
                    help='Set u=v=0 in init and u_nudge=v_nudge=0 in timedep')
parser.add_argument('--wind-u', type=float, default=None,
                    help='Set constant u=WIND_U (m/s) in init and u_nudge in timedep; v=0. '
                         'Mutually exclusive with --zero-winds.')
parser.add_argument('--geo-wind', type=float, default=None,
                    help='Set geostrophic wind: u_geo=GEO_WIND (m/s) constant in z, v_geo=0. '
                         'Initialises u=GEO_WIND, v=0. Mutually exclusive with --wind-u and --zero-winds.')
parser.add_argument('--qt-ls', type=float, default=None,
                    metavar='VALUE',
                    help='Replace qt_ls with a constant profile: VALUE (g kg-1 day-1) '
                         'below 2000 m, zero above, constant in time. '
                         'Overrides the CASS composite qt_ls entirely.')
parser.add_argument('--sun-wind', type=float, default=None, metavar='U',
                    help='Wind direction tracks the sun: target wind vector points in the '
                         'anti-solar horizontal direction (cloud advected toward its shadow) '
                         'with magnitude U (m/s). Writes time-dependent u_nudge, v_nudge, '
                         'u_geo, v_geo on a resampled (600 s) time_ls. '
                         'Requires [force] swtimedep_geo=true in cass.ini. '
                         'Mutually exclusive with --zero-winds, --wind-u, --geo-wind.')
parser.add_argument('--nudge-timescale', type=float, default=10800., metavar='SECONDS',
                    help='Nudging timescale for u,v (s). Default: 10800 (3 h).')
args = parser.parse_args()

if args.wind_u is not None and args.zero_winds:
    parser.error("--wind-u and --zero-winds are mutually exclusive")
if args.geo_wind is not None and args.zero_winds:
    parser.error("--geo-wind and --zero-winds are mutually exclusive")
if args.geo_wind is not None and args.wind_u is not None:
    parser.error("--geo-wind and --wind-u are mutually exclusive")
if args.sun_wind is not None and (args.zero_winds or args.wind_u is not None or args.geo_wind is not None):
    parser.error("--sun-wind is mutually exclusive with --zero-winds, --wind-u, --geo-wind")

float_type = "f8"


def solar_azimuth_deg(t_sec, lat_deg, lon_deg, doy_start=205, hour_utc_start=12.0):
    """Solar azimuth (deg from N, clockwise) at simulation time t_sec.

    NOAA Solar Position Algorithm (simplified; accurate to ~0.5 deg).
    For CASS: t=0 corresponds to day 205.5 = July 24, 12:00 UTC, so
    defaults doy_start=205 and hour_utc_start=12.0 are correct. Accepts
    scalar or numpy array t_sec.
    """
    hour_utc_cont = hour_utc_start + np.asarray(t_sec, dtype=float) / 3600.0
    doy_cont = doy_start + hour_utc_cont / 24.0
    gamma = 2.0 * np.pi / 365.0 * (doy_cont - 1.0)
    eqtime = 229.18 * (0.000075
                       + 0.001868 * np.cos(gamma)
                       - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2.0 * gamma)
                       - 0.040849 * np.sin(2.0 * gamma))
    decl = (0.006918
            - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2.0 * gamma) + 0.000907 * np.sin(2.0 * gamma)
            - 0.002697 * np.cos(3.0 * gamma) + 0.00148 * np.sin(3.0 * gamma))
    hour_utc = hour_utc_cont % 24.0
    tst = hour_utc * 60.0 + eqtime + 4.0 * lon_deg
    ha = np.radians(tst / 4.0 - 180.0)
    lat = np.radians(lat_deg)
    az = np.arctan2(-np.cos(decl) * np.sin(ha),
                    np.sin(decl) * np.cos(lat) - np.cos(decl) * np.sin(lat) * np.cos(ha))
    return np.degrees(np.mod(az, 2.0 * np.pi))

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

# Get number of vertical levels, size, and lat/lon from .ini file
lat = None
lon = None
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
            if key == 'lat':
                lat = float(line.split('=')[1])
            if key == 'lon':
                lon = float(line.split('=')[1])

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

# Replace qt_ls with a constant-in-time, height-stepped profile if requested.
# Profile: VALUE (g/kg/day → kg/kg/s) below 2000 m, zero above.
# thl_ls and w_ls are left unchanged (CASS composite).
if args.qt_ls is not None:
    qt_ls_rate = args.qt_ls / 86400. / 1e3   # g/kg/day → kg/kg/s
    qt_ls_profile = np.where(z <= 2000., qt_ls_rate, 0.)
    qtls = np.tile(qt_ls_profile, (n_times, 1))

# For --sun-wind, resample time_ls to a finer grid so solar-azimuth tracking
# is smooth. Azimuth rotates ~16 deg/h; 600 s ≈ 2.7 deg per step.
if args.sun_wind is not None:
    if lat is None or lon is None:
        raise RuntimeError("--sun-wind requires lat and lon in [grid] of cass.ini")
    dt_new = 600.0
    time_ls_new = np.arange(time_ls[0], time_ls[-1] + 1.0, dt_new)
    thlls = np.array([np.interp(time_ls_new, time_ls, thlls[:, k]) for k in range(kmax)]).T
    qtls  = np.array([np.interp(time_ls_new, time_ls, qtls[:, k])  for k in range(kmax)]).T
    wls   = np.array([np.interp(time_ls_new, time_ls, wls[:, k])   for k in range(kmax)]).T
    time_ls = time_ls_new
    n_times = len(time_ls)


# -----------------------------------------------------------------------
# Read pre-computed CAMS aerosol composite (from compute_cams_composite.py).
# Profiles are on CAMS native model levels; interpolated to LES z below.
# -----------------------------------------------------------------------
print("Reading CAMS aerosol composite (cass_cams_composite.nc)...")

aerosol_names = [f'aermr{i:02d}' for i in range(1, 12)]

cams_composite = xr.open_dataset('cass_cams_composite.nc')
z_lay_cams = cams_composite['z_lay'].values
aer_lay_sum = {name: cams_composite[name].values for name in aerosol_names}
n_cams = int(cams_composite.attrs.get('n_composite_days', -1))

# Interpolate composite aerosol profiles from CAMS levels to LES z grid
sort_idx = np.argsort(z_lay_cams)
z_lay_cams_sorted = z_lay_cams[sort_idx]
aer_les_sum = {}
for name in aerosol_names:
    arr_sorted = aer_lay_sum[name][sort_idx]
    aer_les_sum[name] = np.maximum(np.interp(z, z_lay_cams_sorted, arr_sorted), 0.)

print(f'Loaded CAMS composite ({n_cams} days)')


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
add_nc_var("couvreux", ("z",), nc_group_init, np.zeros(kmax))
if args.zero_winds:
    u_init = np.zeros(kmax); v_init = np.zeros(kmax)
elif args.wind_u is not None:
    u_init = np.full(kmax, args.wind_u); v_init = np.zeros(kmax)
elif args.geo_wind is not None:
    u_init = np.full(kmax, args.geo_wind); v_init = np.zeros(kmax)
elif args.sun_wind is not None:
    az0 = np.radians(solar_azimuth_deg(0.0, lat, lon))
    u_init = np.full(kmax, -args.sun_wind * np.sin(az0))
    v_init = np.full(kmax, -args.sun_wind * np.cos(az0))
else:
    u_init = u; v_init = v
add_nc_var("u", ("z",), nc_group_init, u_init)
add_nc_var("v", ("z",), nc_group_init, v_init)
if args.geo_wind is not None:
    add_nc_var("u_geo", ("z",), nc_group_init, np.full(kmax, args.geo_wind))
    add_nc_var("v_geo", ("z",), nc_group_init, np.zeros(kmax))
# For --sun-wind, u_geo/v_geo are written to the timedep group below.
nudge_timescale = args.nudge_timescale
add_nc_var("nudgefac", ("z",), nc_group_init, np.ones(kmax) / nudge_timescale)

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
if args.zero_winds:
    u_nudge_arr = np.zeros_like(uls); v_nudge_arr = np.zeros_like(vls)
elif args.wind_u is not None:
    u_nudge_arr = np.full_like(uls, args.wind_u); v_nudge_arr = np.zeros_like(vls)
elif args.geo_wind is not None:
    u_nudge_arr = np.full_like(uls, args.geo_wind); v_nudge_arr = np.zeros_like(vls)
elif args.sun_wind is not None:
    az_t = np.radians(solar_azimuth_deg(time_ls, lat, lon))
    u_target = -args.sun_wind * np.sin(az_t)   # shape (n_times,)
    v_target = -args.sun_wind * np.cos(az_t)
    u_nudge_arr = np.broadcast_to(u_target[:, None], (n_times, kmax)).copy()
    v_nudge_arr = np.broadcast_to(v_target[:, None], (n_times, kmax)).copy()
else:
    u_nudge_arr = uls; v_nudge_arr = vls
add_nc_var("u_nudge", ("time_ls", "z"), nc_group_timedep, u_nudge_arr)
add_nc_var("v_nudge", ("time_ls", "z"), nc_group_timedep, v_nudge_arr)

# For --sun-wind: write u_geo, v_geo to the timedep group (aligned with the
# nudge target) so swtimedep_geo=true picks them up; Coriolis tendency then
# vanishes when the BL wind matches the prescription.
if args.sun_wind is not None:
    add_nc_var("u_geo", ("time_ls", "z"), nc_group_timedep, u_nudge_arr)
    add_nc_var("v_geo", ("time_ls", "z"), nc_group_timedep, v_nudge_arr)


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
for name in aerosol_names:
    aer_cams_sorted = aer_lay_sum[name][sort_idx]
    aer_era5 = np.interp(z_lay_era5, z_lay_cams_sorted, aer_cams_sorted)
    add_nc_var(name, ('lay',), nc_rad, np.maximum(aer_era5, 0.))


# land surface model: composite-mean soil from cass_ls2d_input.nc
ls2d_soil = xr.open_dataset('cass_ls2d_input.nc', group = 'soil')
nc_soil = nc_file.createGroup('soil')
add_nc_dim('z', ls2d_soil.sizes['z'], nc_soil)
add_nc_var('z', ('z'), nc_soil, ls2d_soil.z.values)
theta_soil_init = ls2d_soil.theta_soil.values
add_nc_var('theta_soil', ('z'), nc_soil, theta_soil_init)
add_nc_var('t_soil', ('z'), nc_soil, ls2d_soil.t_soil.values)
add_nc_var('index_soil', ('z'), nc_soil, ls2d_soil.index_soil.values)
add_nc_var('root_frac', ('z'), nc_soil, ls2d_soil.root_frac.values)


nc_file.close()

print(f"Successfully created cass_input.nc")
print(f"   Initial profiles at z = {z[0]:.1f} to {z[-1]:.1f} m")
print(f"   Surface fluxes from t = {time_surface[0]:.0f} to {time_surface[-1]:.0f} s")
print(f"   Large-scale forcings from t = {time_ls[0]:.0f} to {time_ls[-1]:.0f} s")
print(f"   Aerosols: composite mean over {n_cams} CAMS days (2003-2009)")
wind_desc = ("zero winds" if args.zero_winds
             else f"u={args.wind_u:.1f} m/s (constant), v=0" if args.wind_u is not None
             else f"geo wind ug={args.geo_wind:.1f} m/s, vg=0" if args.geo_wind is not None
             else f"sun-tracking wind, U={args.sun_wind:.1f} m/s anti-solar (time-dep u_nudge, v_nudge, u_geo, v_geo)" if args.sun_wind is not None
             else "ERA5 composite winds")
print(f"   winds: {wind_desc}")
print(f"   nudgefac = 1/{nudge_timescale:.0f} s [u,v only]")
if args.qt_ls is not None:
    print(f"   qt_ls: constant {args.qt_ls:+.1f} g/kg/day below 2000 m (overrides CASS composite)")
