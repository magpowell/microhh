"""
GoAmazon composite shallow-cumulus case: input generator for MicroHH.

Forcing is the composite ShCu dataset from Manco & Figueroa (2025, Atmosphere
16(7):789) -- an average of 6 pure, non-precipitating shallow-Cu days from
GoAmazon 2014/15 (Mar10, Sep3, Sep11, Oct1, Oct5, Oct8), centered on the SIPAM
Ponta Pelada site (-3.15, -59.99). No deep transition by construction (that's
the point -- this is the shallow-only counterfactual to the goamazon
single-pulse and arm97sd cases).

Data available at https://ftp.cptec.inpe.br/pesquisa/bamc/MPDI/ as
sfc/snd/lsf (SAM native format, same block-header convention as CASS's
cass_sfc.txt/cass_snd.txt/cass_lsf.txt -- but tab/space-tokenized "day levels
pres0" markers, not CASS's comma-joined "day, levels, pres0" string, hence a
separate parser rather than reusing cass_input.py directly).

Design:
  - t=0 is 06:00 LT (10:00 UTC, Amazon UTC-4) per the paper; 14 h to 20:00 LT.
  - sfc file (prescribed day/sst/H/LE/TAU) is NOT used as a boundary
    condition -- interactive LSM instead (unlike the paper's own SAM
    prescribed-flux setup). sfc is stored in the "validation" group only,
    for comparing interactive-LSM H/LE against the paper's composite.
  - snd (t=0 only) initializes thl/qt/u/v; lsf supplies timedep thl/qt/u/v/w
    large-scale tendencies (w = subsidence, applied via swwls=local as in
    CASS).
  - Radiation background/gases/soil-init reused from the goamazon single-
    pulse case's LS2D/ERA5 file (dry-season Oct 2014 only; the composite
    spans wet+dry season, so this is an approximation -- background/gas/
    soil-init state matters far less than the sfc/snd/lsf forcing itself
    for a boundary-layer-driven shallow-Cu case).

Reads goamazon_shcu.ini from CWD (ktot, zsize, endtime) -- run from the run
dir.
"""

import numpy as np
import netCDF4 as nc
import xarray as xr

float_type = "f8"

Rd = 287.04
Rv = 461.5
cp = 1005.
g  = 9.81
p00 = 1.e5

SFC_FILE = 'sfc'
SND_FILE = 'snd'
LSF_FILE = 'lsf'
LS2D_FILE = 'goamazon_ls2d_input.nc'   # reused from the goamazon case
TAU_NUDGE = 10800.   # 3 h; same convention as goamazon/arm97sd/CASS


def add_nc_var(name, dims, ncgrp, data):
    if name not in ncgrp.variables:
        var = ncgrp.createVariable(name, np.float64, dims) if dims is not None \
            else ncgrp.createVariable(name, np.float64)
        var[:] = data


def add_nc_dim(name, size, ncgrp):
    if name not in ncgrp.dimensions:
        ncgrp.createDimension(name, size)


def parse_sam_blocks(path, ncols):
    """Parse a SAM-native forcing/sounding file: one column-header line,
    then repeating blocks of [<day> <nlevels> <pres0> day levels pres0]
    followed by nlevels data rows of ncols columns each.

    Returns (days, data) where data has shape (n_blocks, nlevels, ncols).
    """
    with open(path) as f:
        lines = f.readlines()

    days = []
    blocks = []
    i = 1  # skip the column-header line
    while i < len(lines):
        toks = lines[i].split()
        if len(toks) >= 6 and toks[-3:] == ['day', 'levels', 'pres0']:
            day = float(toks[0])
            nlev = int(toks[1])
            rows = [ [float(x) for x in lines[i + 1 + k].split()] for k in range(nlev) ]
            days.append(day)
            blocks.append(rows)
            i += 1 + nlev
        else:
            i += 1

    nlev = len(blocks[0])
    assert all(len(b) == nlev for b in blocks), "inconsistent level count across blocks"
    data = np.array(blocks)  # (n_blocks, nlev, ncols)
    assert data.shape[2] == ncols, f"expected {ncols} columns, got {data.shape[2]}"
    return np.array(days), data


# -----------------------------------------------------------------------
# LES grid and endtime from goamazon_shcu.ini
# -----------------------------------------------------------------------
kmax = zsize = endtime = None
with open('goamazon_shcu.ini') as f:
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

dz = zsize / kmax
z = np.linspace(0.5 * dz, zsize - 0.5 * dz, kmax)

# -----------------------------------------------------------------------
# Parse sounding (t=0 only), large-scale forcing, and surface (validation)
# -----------------------------------------------------------------------
print(f"Reading {SND_FILE}, {LSF_FILE}, {SFC_FILE}...")

snd_days, snd_data = parse_sam_blocks(SND_FILE, ncols=6)   # z,p,tp,q,u,v
z_snd, p_snd = snd_data[0, :, 0], snd_data[0, :, 1]
tp_snd, q_snd = snd_data[0, :, 2], snd_data[0, :, 3]
u_snd, v_snd = snd_data[0, :, 4], snd_data[0, :, 5]

thl0 = np.interp(z, z_snd, tp_snd)         # unsaturated t=0 assumption
qt0  = np.interp(z, z_snd, q_snd) / 1000.  # g/kg -> kg/kg
u0   = np.interp(z, z_snd, u_snd)
v0   = np.interp(z, z_snd, v_snd)

lsf_days, lsf_data = parse_sam_blocks(LSF_FILE, ncols=7)   # z,p,tls,qls,u,v,w
time_ls = lsf_days * 86400.                                 # day -> s
n_t = time_ls.size
z_lsf = lsf_data[0, :, 0]                                   # levels fixed across blocks

thl_ls = np.zeros((n_t, kmax))
qt_ls  = np.zeros((n_t, kmax))
w_ls   = np.zeros((n_t, kmax))
u_nudge = np.zeros((n_t, kmax))
v_nudge = np.zeros((n_t, kmax))
for t in range(n_t):
    thl_ls[t]  = np.interp(z, z_lsf, lsf_data[t, :, 2])
    qt_ls[t]   = np.interp(z, z_lsf, lsf_data[t, :, 3])
    u_nudge[t] = np.interp(z, z_lsf, lsf_data[t, :, 4])
    v_nudge[t] = np.interp(z, z_lsf, lsf_data[t, :, 5])
    w_ls[t]    = np.interp(z, z_lsf, lsf_data[t, :, 6])

# Surface (validation only -- interactive LSM computes its own H/LE)
sfc_raw = np.loadtxt(SFC_FILE, skiprows=1)
sfc_time = sfc_raw[:, 0] * 86400.
sfc_sst, sfc_H, sfc_LE, sfc_TAU = sfc_raw[:, 1], sfc_raw[:, 2], sfc_raw[:, 3], sfc_raw[:, 4]

print(f"  {n_t} forcing times, t = {time_ls[0]:.0f} to {time_ls[-1]:.0f} s; "
      f"nlev(snd)={len(z_snd)}, nlev(lsf)={len(z_lsf)}")

# -----------------------------------------------------------------------
# Reuse goamazon's LS2D/ERA5 background: radiation column, gases, soil
# -----------------------------------------------------------------------
rad  = xr.open_dataset(LS2D_FILE, group='radiation')
soil = xr.open_dataset(LS2D_FILE, group='soil')

o3_z  = np.interp(z, rad['z_ref_grid'].values, rad['o3_z'].values)
h2o_z = np.interp(z, rad['z_ref_grid'].values, rad['h2o_z'].values)

# -----------------------------------------------------------------------
# Write goamazon_shcu_input.nc
# -----------------------------------------------------------------------
print("Saving goamazon_shcu_input.nc...")
f = nc.Dataset('goamazon_shcu_input.nc', mode='w', datamodel='NETCDF4', clobber=True)

add_nc_dim('z', kmax, f)
add_nc_var('z', ('z',), f, z)

grp_init = f.createGroup('init')
add_nc_var('z',   ('z',), grp_init, z)
add_nc_var('thl', ('z',), grp_init, thl0)
add_nc_var('qt',  ('z',), grp_init, qt0)
add_nc_var('u',   ('z',), grp_init, u0)
add_nc_var('v',   ('z',), grp_init, v0)
add_nc_var('nudgefac', ('z',), grp_init, np.ones(kmax) / TAU_NUDGE)
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
add_nc_var('w_ls',    ('time_ls', 'z'), grp_td, w_ls)
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
add_nc_dim('time', sfc_time.size, grp_val)
add_nc_var('time', ('time',), grp_val, sfc_time)
add_nc_var('sst',  ('time',), grp_val, sfc_sst)
add_nc_var('shflx', ('time',), grp_val, sfc_H)
add_nc_var('lhflx', ('time',), grp_val, sfc_LE)
add_nc_var('tau',  ('time',), grp_val, sfc_TAU)

f.close()

print("Successfully created goamazon_shcu_input.nc")
print(f"   grid: ktot={kmax}, dz={dz:.1f} m, zsize={zsize:.0f} m")
print(f"   forcing: t = {time_ls[0]:.0f} to {time_ls[-1]:.0f} s ({n_t} times, 3-hourly)")
print(f"   init surface: thl={thl0[0]:.2f} K, qt={qt0[0]*1e3:.2f} g/kg, "
      f"u={u0[0]:.2f}, v={v0[0]:.2f} m/s")
