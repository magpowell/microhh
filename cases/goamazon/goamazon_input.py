"""
GoAmazon single-pulse case: input generator for MicroHH.

Converts the DP-SCREAM IOP forcing (GOAMAZON_singlepulse_iopfile_4scam.nc,
Tian & Zhang 2025; day 278, "early single peak convection") plus the LS2D/ERA5
background (goamazon_ls2d_input.nc) into goamazon_input.nc.

Design (mirrors the DP-SCREAM cntl forcing, intland surface):
  - t=0 is 2014-10-05 12:00 UTC (IOP tsec=43200); 12 h of forcing.
  - Initial thl/qt/u/v from IOP t=0, hydrostatically mapped p->z,
    padded with ERA5 above the IOP top (~110 hPa, ~16.5 km).
  - divT/divq applied as time-dependent large-scale thl/qt tendencies,
    tapered to zero over the top 1 km of IOP coverage. No subsidence
    (matches iop_dosubsidence=false; divT/divq are total advective
    forcings), no Coriolis (iop_coriolis=false).
  - u/v nudged to IOP u_ls/v_ls (the EAMxx target when *_ls present),
    tau = 10800 s (EAMxx iop_nudge_tscale default), ERA5 above IOP top.
  - Surface: interactive LSM (soil from ERA5); the IOP shflx/lhflx/Tg are
    stored in a "validation" group (ignored by MicroHH) for comparison.

Reads goamazon.ini from CWD (ktot, zsize, lat, lon) -- run from the run dir.
"""

import numpy as np
import netCDF4 as nc
import xarray as xr

float_type = "f8"

# Constants
Rd  = 287.04
Rv  = 461.5
cp  = 1005.
g   = 9.81
p00 = 1.e5

IOP_FILE  = 'GOAMAZON_singlepulse_iopfile_4scam.nc'
LS2D_FILE = 'goamazon_ls2d_input.nc'

IOP_START_SEC = 43200      # tsec at 12:00 UTC = t=0 of the simulation
TAU_NUDGE     = 10800.     # EAMxx iop_nudge_tscale default
BLEND_DZ      = 1000.      # IOP->ERA5 blend zone below the IOP top


def add_nc_var(name, dims, ncgrp, data):
    if name not in ncgrp.variables:
        if dims is None:
            var = ncgrp.createVariable(name, np.float64)
        else:
            var = ncgrp.createVariable(name, np.float64, dims)
        var[:] = data


def add_nc_dim(name, size, ncgrp):
    if name not in ncgrp.dimensions:
        ncgrp.createDimension(name, size)


# -----------------------------------------------------------------------
# LES grid from goamazon.ini
# -----------------------------------------------------------------------
kmax = zsize = None
with open('goamazon.ini') as f:
    in_grid = False
    for line in f:
        s = line.strip()
        if s.startswith('['):
            in_grid = (s == '[grid]')
        if in_grid and '=' in s:
            key = s.split('=')[0].strip()
            if key == 'ktot':
                kmax = int(s.split('=')[1].split('#')[0])
            if key == 'zsize':
                zsize = float(s.split('=')[1].split('#')[0])

dz = zsize / kmax
z = np.linspace(0.5*dz, zsize - 0.5*dz, kmax)

# -----------------------------------------------------------------------
# Read IOP file; reorder levels bottom->top
# -----------------------------------------------------------------------
print(f"Reading {IOP_FILE}...")
iop = xr.open_dataset(IOP_FILE, decode_times=False).squeeze(['lat', 'lon'])
iop = iop.isel(lev=slice(None, None, -1))          # now p descending = bottom->top

time_ls = iop['tsec'].values.astype(float) - IOP_START_SEC
n_t = time_ls.size
p_iop = iop['lev'].values                          # (lev,) Pa, bottom->top
ps = iop['Ps'].values                              # (time,)

T   = iop['T'].values                              # (time, lev) K
qmr = iop['q'].values                              # (time, lev) mixing ratio
qv  = qmr / (1. + qmr)                             # specific humidity
u_i = iop['u'].values
v_i = iop['v'].values
u_ls = iop['u_ls'].values
v_ls = iop['v_ls'].values
divT = iop['divT'].values                          # (time, lev) K/s
divq = iop['divq'].values                          # (time, lev) kg/kg/s (mixing ratio)

# Hydrostatic z(p, t) from the surface up (phis = 0 for this site)
Tv = T * (1. + 0.61*qv)
z_iop = np.zeros_like(T)
for t in range(n_t):
    z_iop[t, 0] = Rd*Tv[t, 0]/g * np.log(ps[t]/p_iop[0])
    for k in range(1, p_iop.size):
        z_iop[t, k] = z_iop[t, k-1] \
            + Rd*0.5*(Tv[t, k-1] + Tv[t, k])/g * np.log(p_iop[k-1]/p_iop[k])

z_iop_top = z_iop[:, -1].min()
print(f"  IOP top: p={p_iop[-1]:.0f} Pa, z~{z_iop_top:.0f} m; "
      f"LES top: {zsize:.0f} m")

exner = (p_iop / p00)**(Rd/cp)                     # (lev,)
th = T / exner[None, :]

# Saturation check at t=0 (thl = th assumes no condensate in the profile)
esat0 = 611.21*np.exp(17.502*(T[0]-273.16)/(T[0]-32.19))
qsat0 = Rd/Rv*esat0/(p_iop - (1.-Rd/Rv)*esat0)
if np.any(qv[0] > 0.99*qsat0):
    ksat = np.where(qv[0] > 0.99*qsat0)[0]
    print(f"  WARNING: initial profile near-saturated at "
          f"z={z_iop[0, ksat[0]]:.0f}-{z_iop[0, ksat[-1]]:.0f} m; "
          f"thl=th assumption marginal there.")

# -----------------------------------------------------------------------
# ERA5 padding profiles (above IOP top) and radiation/soil groups
# -----------------------------------------------------------------------
ls2d_root = xr.open_dataset(LS2D_FILE)
if int(ls2d_root.attrs.get('dummy', 0)) == 1:
    print('\n' + '!'*70)
    print('!! WARNING: goamazon_ls2d_input.nc is the DUMMY (RCEMIP) stand-in.')
    print('!! Memory fit tests only -- NOT valid for science runs.')
    print('!'*70 + '\n')
ls2d_root.close()

era = xr.open_dataset(LS2D_FILE, group='era5_profiles')
z_era = era['z'].values
t_era = era['time_sec'].values                     # seconds since 12:00 UTC


def era_on_les(name, t_sec):
    """ERA5 profile interpolated to LES z at simulation time t_sec."""
    prof_t = np.empty((t_era.size, kmax))
    for i in range(t_era.size):
        prof_t[i] = np.interp(z, z_era, era[name].values[i])
    out = np.empty(kmax)
    for k in range(kmax):
        out[k] = np.interp(t_sec, t_era, prof_t[:, k])
    return out


def blend_iop_era(iop_prof, era_prof, ztop):
    """IOP below ztop-BLEND_DZ, ERA5 above ztop, linear blend between."""
    w = np.clip((z - (ztop - BLEND_DZ)) / BLEND_DZ, 0., 1.)
    return (1. - w)*iop_prof + w*era_prof


# -----------------------------------------------------------------------
# Initial profiles (IOP t=0, ERA5 above)
# -----------------------------------------------------------------------
thl0 = blend_iop_era(np.interp(z, z_iop[0], th[0]),  era_on_les('thl', 0.), z_iop[0, -1])
qt0  = blend_iop_era(np.interp(z, z_iop[0], qv[0]),  era_on_les('qt', 0.),  z_iop[0, -1])
u0   = blend_iop_era(np.interp(z, z_iop[0], u_i[0]), era_on_les('u', 0.),   z_iop[0, -1])
v0   = blend_iop_era(np.interp(z, z_iop[0], v_i[0]), era_on_les('v', 0.),   z_iop[0, -1])

# -----------------------------------------------------------------------
# Time-dependent forcings on the LES grid
# -----------------------------------------------------------------------
thl_ls   = np.zeros((n_t, kmax))
qt_ls    = np.zeros((n_t, kmax))
u_nudge  = np.zeros((n_t, kmax))
v_nudge  = np.zeros((n_t, kmax))

for t in range(n_t):
    zt = z_iop[t]
    ztop = zt[-1]
    # Tendency taper: 1 below ztop-BLEND_DZ, ->0 at ztop, 0 above
    taper = np.clip((ztop - z) / BLEND_DZ, 0., 1.)

    # divT (dT/dt) -> dthl/dt via local exner; divq (mixing ratio) -> qt
    thl_tend = divT[t] / exner
    qt_tend  = divq[t] / (1. + qmr[t])**2

    thl_ls[t]  = np.interp(z, zt, thl_tend, right=0.) * taper
    qt_ls[t]   = np.interp(z, zt, qt_tend,  right=0.) * taper

    # Nudge targets: IOP u_ls/v_ls below, ERA5 above
    u_nudge[t] = blend_iop_era(np.interp(z, zt, u_ls[t]),
                               era_on_les('u', time_ls[t]), ztop)
    v_nudge[t] = blend_iop_era(np.interp(z, zt, v_ls[t]),
                               era_on_les('v', time_ls[t]), ztop)

# -----------------------------------------------------------------------
# Radiation background + gases, soil (from LS2D file, as in CASS)
# -----------------------------------------------------------------------
rad  = xr.open_dataset(LS2D_FILE, group='radiation')
soil = xr.open_dataset(LS2D_FILE, group='soil')

o3_z  = np.interp(z, rad['z_ref_grid'].values, rad['o3_z'].values)
h2o_z = np.interp(z, rad['z_ref_grid'].values, rad['h2o_z'].values)

# -----------------------------------------------------------------------
# Write goamazon_input.nc
# -----------------------------------------------------------------------
print("Saving goamazon_input.nc...")
f = nc.Dataset('goamazon_input.nc', mode='w', datamodel='NETCDF4', clobber=True)

add_nc_dim('z', kmax, f)
add_nc_var('z', ('z',), f, z)

grp_init = f.createGroup('init')
add_nc_var('z',   ('z',), grp_init, z)
add_nc_var('thl', ('z',), grp_init, thl0)
add_nc_var('qt',  ('z',), grp_init, qt0)
add_nc_var('u',   ('z',), grp_init, u0)
add_nc_var('v',   ('z',), grp_init, v0)
add_nc_var('nudgefac', ('z',), grp_init, np.ones(kmax)/TAU_NUDGE)

# Radiation gases on the LES grid
add_nc_var('h2o', ('z',), grp_init, h2o_z)
add_nc_var('o3',  ('z',), grp_init, o3_z)
for gas in ('co2', 'ch4', 'n2o', 'n2', 'o2'):
    add_nc_var(gas, (), grp_init, rad[gas].values)

grp_td = f.createGroup('timedep')
add_nc_dim('time_ls', n_t, grp_td)
add_nc_dim('z', kmax, grp_td)
add_nc_var('time_ls', ('time_ls',), grp_td, time_ls)
add_nc_var('thl_ls',  ('time_ls', 'z'), grp_td, thl_ls)
add_nc_var('qt_ls',   ('time_ls', 'z'), grp_td, qt_ls)
add_nc_var('u_nudge', ('time_ls', 'z'), grp_td, u_nudge)
add_nc_var('v_nudge', ('time_ls', 'z'), grp_td, v_nudge)

# Radiation background column
grp_rad = f.createGroup('radiation')
add_nc_dim('lay', rad.sizes['lay'], grp_rad)
add_nc_dim('lev', rad.sizes['lev'], grp_rad)
for name in ('z_lay', 'p_lay', 't_lay', 'o3', 'h2o'):
    add_nc_var(name, ('lay',), grp_rad, rad[name].values)
for name in ('z_lev', 'p_lev', 't_lev'):
    add_nc_var(name, ('lev',), grp_rad, rad[name].values)
for gas in ('co2', 'ch4', 'n2o', 'n2', 'o2'):
    add_nc_var(gas, (), grp_rad, rad[gas].values)

# Soil
grp_soil = f.createGroup('soil')
add_nc_dim('z', soil.sizes['z'], grp_soil)
add_nc_var('z', ('z',), grp_soil, soil['z'].values)
for name in ('theta_soil', 't_soil', 'index_soil', 'root_frac'):
    add_nc_var(name, ('z',), grp_soil, soil[name].values)

# Validation group (ignored by MicroHH): IOP surface obs for LSM checks
grp_val = f.createGroup('validation')
add_nc_dim('time', n_t, grp_val)
add_nc_var('time',  ('time',), grp_val, time_ls)
add_nc_var('shflx', ('time',), grp_val, iop['shflx'].values)
add_nc_var('lhflx', ('time',), grp_val, iop['lhflx'].values)
add_nc_var('Tg',    ('time',), grp_val, iop['Tg'].values)
add_nc_var('Ps',    ('time',), grp_val, ps)

f.close()

print("Successfully created goamazon_input.nc")
print(f"   grid: ktot={kmax}, dz={dz:.1f} m, zsize={zsize:.0f} m")
print(f"   forcings: t = {time_ls[0]:.0f} to {time_ls[-1]:.0f} s "
      f"({n_t} times)")
print(f"   nudging: u,v -> IOP u_ls/v_ls, tau={TAU_NUDGE:.0f} s")
print(f"   init surface: thl={thl0[0]:.2f} K, qt={qt0[0]*1e3:.2f} g/kg, "
      f"u={u0[0]:.2f}, v={v0[0]:.2f} m/s")
