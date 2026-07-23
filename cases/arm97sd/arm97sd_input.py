"""
ARM97 shallow-to-deep case (SGP, 27 June 1997): input generator for MicroHH.

Carves the canonical shallow-to-deep transition day (EUROCS/GCSS continental
deep-convection diurnal cycle, Guichard et al. 2004) out of the ARM97 IOP
forcing file (variational analysis, 20-min cadence, 35 p-levels to ~115 hPa)
plus the LS2D/ERA5 background (arm97sd_ls2d_input.nc), into arm97sd_input.nc.

Mirrors the DP-SCREAM ARM97 forcing treatment (and the GoAmazon sibling case):
  - t=0 is 1997-06-27 11:30 UTC (tsec=819000 rel. bdate 1997-06-18 00Z).
  - divT/divq applied as timedep LS thl/qt tendencies; no subsidence
    (iop_dosubsidence=false; vertdivT unused by EAMxx), no Coriolis.
  - u/v nudged to IOP u/v (no u_ls/v_ls in this file -- the EAMxx fallback),
    tau = 10800 s; ERA5 above the IOP top.
  - Interactive LSM; IOP shflx/lhflx/Tg stored in the "validation" group.

Reads arm97sd.ini from CWD (ktot, zsize, endtime) -- run from the run dir.
"""

import os

import numpy as np
import netCDF4 as nc
import xarray as xr

float_type = "f8"

Rd  = 287.04
Rv  = 461.5
cp  = 1005.
g   = 9.81
p00 = 1.e5

IOP_FILE  = 'ARM97_iopfile_4scam.nc'
LS2D_FILE = 'arm97sd_ls2d_input.nc'

IOP_START_SEC = 819000.    # 1997-06-27 11:30 UTC relative to bdate 06-18 00Z
TAU_NUDGE     = 10800.     # EAMxx iop_nudge_tscale default
BLEND_DZ      = 1000.      # IOP->ERA5 blend zone below the IOP top (~15.5 km)
PAD_SEC       = 2700.      # forcing margin kept beyond [0, endtime]


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
# LES grid and endtime from arm97sd.ini
# -----------------------------------------------------------------------
kmax = zsize = endtime = None
with open('arm97sd.ini') as f:
    section = None
    for line in f:
        s = line.strip()
        if s.startswith('['):
            section = s
        if '=' in s:
            key = s.split('=')[0].strip()
            val = s.split('=')[1].split('#')[0]
            if section == '[grid]' and key == 'ktot':
                kmax = int(val)
            if section == '[grid]' and key == 'zsize':
                zsize = float(val)
            if section == '[time]' and key == 'endtime':
                endtime = float(val)

if os.path.exists('zgrid.txt'):
    z = np.loadtxt('zgrid.txt')
    if len(z) != kmax:
        raise SystemExit(f'zgrid.txt has {len(z)} levels but ktot={kmax}')
else:
    dz = zsize / kmax
    z = np.linspace(0.5*dz, zsize - 0.5*dz, kmax)

# -----------------------------------------------------------------------
# Read IOP file; subset the sim window; reorder levels bottom->top
# -----------------------------------------------------------------------
print(f"Reading {IOP_FILE}...")
iop = xr.open_dataset(IOP_FILE, decode_times=False).squeeze(['lat', 'lon'])

tsec = iop['tsec'].values.astype(float)
# Keep one sample before t=0, interpolated away below -- MicroHH's Timedep
# requires time_ls[0] == 0.0 exactly (negative times overflow its unsigned
# time cast).
mask = (tsec >= IOP_START_SEC - 1200.) & (tsec <= IOP_START_SEC + endtime + PAD_SEC)
if not mask.any():
    raise RuntimeError('IOP subset empty -- check IOP_START_SEC/endtime')
iop = iop.isel(time=np.where(mask)[0])
iop = iop.isel(lev=slice(None, None, -1))          # p descending = bottom->top

time_ls_raw = iop['tsec'].values.astype(float) - IOP_START_SEC
p_iop = iop['lev'].values                          # (lev,) Pa, bottom->top
ps_raw = iop['Ps'].values                          # (time,)

T_raw    = iop['T'].values                          # (time, lev) K
qmr_raw  = iop['q'].values                          # mixing ratio (kg/kg)
u_raw    = iop['u'].values
v_raw    = iop['v'].values
divT_raw = iop['divT'].values                       # horizontal T advection K/s
divq_raw = iop['divq'].values                       # horizontal q advection

if time_ls_raw[0] < 0.:
    w = -time_ls_raw[0] / (time_ls_raw[1] - time_ls_raw[0])
    def interp0(arr):
        row0 = arr[0] + w*(arr[1] - arr[0])
        out = arr.copy()
        out[0] = row0
        return out
    time_ls_raw[0] = 0.
    ps_raw   = interp0(ps_raw)
    T_raw    = interp0(T_raw)
    qmr_raw  = interp0(qmr_raw)
    u_raw    = interp0(u_raw)
    v_raw    = interp0(v_raw)
    divT_raw = interp0(divT_raw)
    divq_raw = interp0(divq_raw)

time_ls = time_ls_raw
n_t = time_ls.size
ps  = ps_raw
T   = T_raw
qmr = qmr_raw
qv  = qmr / (1. + qmr)                             # specific humidity
u_i = u_raw
v_i = v_raw
divT = divT_raw
divq = divq_raw

assert time_ls[0] == 0., "time_ls[0] must be exactly 0.0 (unsigned itime cast)"
print(f"  subset: {n_t} samples, t = {time_ls[0]:.0f} to {time_ls[-1]:.0f} s "
      f"(endtime {endtime:.0f}); t=0 row interpolated to exact origin")

# Hydrostatic z(p, t) from the surface up (heights AGL; Ps time-varying)
Tv = T * (1. + 0.61*qv)
z_iop = np.zeros_like(T)
for t in range(n_t):
    z_iop[t, 0] = Rd*Tv[t, 0]/g * np.log(ps[t]/p_iop[0])
    for k in range(1, p_iop.size):
        z_iop[t, k] = z_iop[t, k-1] \
            + Rd*0.5*(Tv[t, k-1] + Tv[t, k])/g * np.log(p_iop[k-1]/p_iop[k])

print(f"  IOP top: p={p_iop[-1]:.0f} Pa, z~{z_iop[:, -1].min():.0f} m; "
      f"LES top: {zsize:.0f} m")

exner = (p_iop / p00)**(Rd/cp)
th = T / exner[None, :]

# Index of the sample closest to t=0 for the initial profiles
i0 = int(np.argmin(np.abs(time_ls)))

esat0 = 611.21*np.exp(17.502*(T[i0]-273.16)/(T[i0]-32.19))
qsat0 = Rd/Rv*esat0/(p_iop - (1.-Rd/Rv)*esat0)
if np.any(qv[i0] > 0.99*qsat0):
    ksat = np.where(qv[i0] > 0.99*qsat0)[0]
    print(f"  WARNING: initial profile near-saturated at "
          f"z={z_iop[i0, ksat[0]]:.0f}-{z_iop[i0, ksat[-1]]:.0f} m.")

# -----------------------------------------------------------------------
# ERA5 padding profiles and radiation/soil groups
# -----------------------------------------------------------------------
ls2d_root = xr.open_dataset(LS2D_FILE)
if int(ls2d_root.attrs.get('dummy', 0)) == 1:
    print('\n' + '!'*70)
    print('!! WARNING: arm97sd_ls2d_input.nc is a DUMMY stand-in.')
    print('!'*70 + '\n')
ls2d_root.close()

era = xr.open_dataset(LS2D_FILE, group='era5_profiles')
z_era = era['z'].values
t_era = era['time_sec'].values                     # seconds since sim t=0


def era_on_les(name, t_sec):
    prof_t = np.empty((t_era.size, kmax))
    for i in range(t_era.size):
        prof_t[i] = np.interp(z, z_era, era[name].values[i])
    out = np.empty(kmax)
    for k in range(kmax):
        out[k] = np.interp(t_sec, t_era, prof_t[:, k])
    return out


def blend_iop_era(iop_prof, era_prof, ztop):
    w = np.clip((z - (ztop - BLEND_DZ)) / BLEND_DZ, 0., 1.)
    return (1. - w)*iop_prof + w*era_prof


# -----------------------------------------------------------------------
# Initial profiles (IOP nearest t=0, ERA5 above)
# -----------------------------------------------------------------------
ztop0 = z_iop[i0, -1]
thl0 = blend_iop_era(np.interp(z, z_iop[i0], th[i0]),  era_on_les('thl', 0.), ztop0)
qt0  = blend_iop_era(np.interp(z, z_iop[i0], qv[i0]),  era_on_les('qt', 0.),  ztop0)
u0   = blend_iop_era(np.interp(z, z_iop[i0], u_i[i0]), era_on_les('u', 0.),   ztop0)
v0   = blend_iop_era(np.interp(z, z_iop[i0], v_i[i0]), era_on_les('v', 0.),   ztop0)

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
    taper = np.clip((ztop - z) / BLEND_DZ, 0., 1.)

    thl_tend = divT[t] / exner
    qt_tend  = divq[t] / (1. + qmr[t])**2

    thl_ls[t]  = np.interp(z, zt, thl_tend, right=0.) * taper
    qt_ls[t]   = np.interp(z, zt, qt_tend,  right=0.) * taper

    # Nudge targets: IOP u/v (no u_ls/v_ls in ARM97 -- EAMxx fallback)
    u_nudge[t] = blend_iop_era(np.interp(z, zt, u_i[t]),
                               era_on_les('u', time_ls[t]), ztop)
    v_nudge[t] = blend_iop_era(np.interp(z, zt, v_i[t]),
                               era_on_les('v', time_ls[t]), ztop)

# -----------------------------------------------------------------------
# Radiation background + gases, soil
# -----------------------------------------------------------------------
rad  = xr.open_dataset(LS2D_FILE, group='radiation')
soil = xr.open_dataset(LS2D_FILE, group='soil')

o3_z  = np.interp(z, rad['z_ref_grid'].values, rad['o3_z'].values)
h2o_z = np.interp(z, rad['z_ref_grid'].values, rad['h2o_z'].values)

# -----------------------------------------------------------------------
# Write arm97sd_input.nc
# -----------------------------------------------------------------------
print("Saving arm97sd_input.nc...")
f = nc.Dataset('arm97sd_input.nc', mode='w', datamodel='NETCDF4', clobber=True)

add_nc_dim('z', kmax, f)
add_nc_var('z', ('z',), f, z)

grp_init = f.createGroup('init')
add_nc_var('z',   ('z',), grp_init, z)
add_nc_var('thl', ('z',), grp_init, thl0)
add_nc_var('qt',  ('z',), grp_init, qt0)
add_nc_var('u',   ('z',), grp_init, u0)
add_nc_var('v',   ('z',), grp_init, v0)
add_nc_var('nudgefac', ('z',), grp_init, np.ones(kmax)/TAU_NUDGE)

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

grp_rad = f.createGroup('radiation')
add_nc_dim('lay', rad.sizes['lay'], grp_rad)
add_nc_dim('lev', rad.sizes['lev'], grp_rad)
for name in ('z_lay', 'p_lay', 't_lay', 'o3', 'h2o'):
    add_nc_var(name, ('lay',), grp_rad, rad[name].values)
for name in ('z_lev', 'p_lev', 't_lev'):
    add_nc_var(name, ('lev',), grp_rad, rad[name].values)
for gas in ('co2', 'ch4', 'n2o', 'n2', 'o2'):
    add_nc_var(gas, (), grp_rad, rad[gas].values)

grp_soil = f.createGroup('soil')
add_nc_dim('z', soil.sizes['z'], grp_soil)
add_nc_var('z', ('z',), grp_soil, soil['z'].values)
for name in ('theta_soil', 't_soil', 'index_soil', 'root_frac'):
    add_nc_var(name, ('z',), grp_soil, soil[name].values)

grp_val = f.createGroup('validation')
add_nc_dim('time', n_t, grp_val)
add_nc_var('time',  ('time',), grp_val, time_ls)
add_nc_var('shflx', ('time',), grp_val, iop['shflx'].values)
add_nc_var('lhflx', ('time',), grp_val, iop['lhflx'].values)
add_nc_var('Tg',    ('time',), grp_val, iop['Tg'].values)
add_nc_var('Ps',    ('time',), grp_val, ps)
add_nc_var('prec',  ('time',), grp_val, iop['Prec'].values)

f.close()

print("Successfully created arm97sd_input.nc")
zh_ = np.concatenate(([0.], 0.5*(z[1:]+z[:-1]), [zsize]))
dz_ = np.diff(zh_)
print(f"   grid: ktot={kmax}, dz={dz_.min():.1f}-{dz_.max():.1f} m, zsize={zsize:.0f} m")
print(f"   forcings: t = {time_ls[0]:.0f} to {time_ls[-1]:.0f} s ({n_t} times)")
print(f"   nudging: u,v -> IOP u/v, tau={TAU_NUDGE:.0f} s")
print(f"   init surface: thl={thl0[0]:.2f} K, qt={qt0[0]*1e3:.2f} g/kg, "
      f"u={u0[0]:.2f}, v={v0[0]:.2f} m/s")
