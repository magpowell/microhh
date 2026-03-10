#!/usr/bin/env python3
"""
Compute composite-mean CAMS aerosol profiles from the local LS2D_CAMS cache.

Reads pre-downloaded CAMS EAC4 data (from download_cams.py), averages over
all composite days with available CAMS data (2003-2009), and writes
cass_cams_composite.nc to $SCRATCH/CASS_LES/shared_data/.

Run once (or re-run idempotently — output is overwritten).
Output is then symlinked into shared/data/ for use by cass_input.py.

Usage:
  python compute_cams_composite.py
"""

import sys
sys.path.insert(0, '/global/homes/m/mpowell/repos/LS2D')

import os
import numpy as np
import netCDF4 as nc
from datetime import datetime
from pathlib import Path

import ls2d as ls2d_pkg
from microhh.cases.cass.shared.cass_utils import read_composite_days

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
SCRATCH      = Path(os.environ.get('SCRATCH', '/pscratch/sd/m/mpowell'))
OUT_DIR      = SCRATCH / 'CASS_LES' / 'shared_data'
OUT_FILE     = OUT_DIR / 'cass_cams_composite.nc'

# -----------------------------------------------------------------------
# LES z grid (production; needed by get_les_input — only _lay output used)
# -----------------------------------------------------------------------
KTOT  = 256
ZSIZE = 6400.
dz = ZSIZE / KTOT
z_prod = np.linspace(0.5 * dz, ZSIZE - 0.5 * dz, KTOT)

# -----------------------------------------------------------------------
# CAMS settings (reads from local cache only — no CDS API calls needed)
# -----------------------------------------------------------------------
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
    'cams_path'   : str(SCRATCH / 'LS2D_CAMS'),
    'cdsapirc'    : '/global/homes/m/mpowell/.cdsapirc',
    'write_log'   : False,
    'data_source' : 'CDS',
    'ntasks'      : 1,
}

aerosol_names = [f'aermr{i:02d}' for i in range(1, 12)]

PHYS_MAX = {
    'aermr01': 5e-8,    # sea salt 0.03–0.5 µm  (tiny at inland SGP)
    'aermr02': 2e-7,    # sea salt 0.5–5 µm
    'aermr03': 5e-7,    # sea salt 5–20 µm
    'aermr04': 2e-7,    # dust 0.03–0.55 µm
    'aermr05': 2e-7,    # dust 0.55–0.9 µm
    'aermr06': 5e-7,    # dust 0.9–20 µm   (coarse: large in strong events)
    'aermr07': 1e-8,    # OM hydrophilic   
    'aermr08': 1e-8,    # OM hydrophobic
    'aermr09': 1e-8,    # BC hydrophilic
    'aermr10': 1e-8,    # BC hydrophobic
    'aermr11': 1e-8,    # sulphate         
}

# -----------------------------------------------------------------------
# Loop over composite days and accumulate
# -----------------------------------------------------------------------
cams_days = read_composite_days(year_min=2003, year_max=2009)
print(f'Processing {len(cams_days)} composite days (2003-2009)...')

n_ok = 0
z_lay_stack = []
aer_lay_stack = {name: [] for name in aerosol_names}

for dt in cams_days:
    settings = dict(base_cams_settings)
    settings['start_date'] = datetime(year=dt.year, month=dt.month, day=dt.day, hour=10)
    settings['end_date']   = datetime(year=dt.year, month=dt.month, day=dt.day, hour=23)

    try:
        cams     = ls2d_pkg.Read_cams(settings, variables=cams_vars)
        cams_les = cams.get_les_input(z_prod)

        # Composite-mean CAMS layer heights and aerosol profiles on native levels.
        # The _lay arrays are on CAMS model levels (independent of LES z).
        z_lay_stack.append(np.nanmean(cams_les.z_lay.values, axis=0))
        for name in aerosol_names:
            arr = np.nanmean(cams_les[f'{name}_lay'].values, axis=0)
            aer_lay_stack[name].append(np.minimum(np.maximum(arr, 0.), PHYS_MAX[name]))

        n_ok += 1
        print(f'  {dt.strftime("%Y-%m-%d")}  ({n_ok}/{len(cams_days)})')

    except Exception as e:
        print(f'  Warning: skipped {dt.strftime("%Y-%m-%d")}: {e}')
        continue

if n_ok == 0:
    raise RuntimeError('No CAMS days were successfully processed. '
                       'Run download_cams.py first.')

# nanmean across days so any NaN levels/days are skipped rather than propagated.
print(f'\nAveraging over {n_ok} days...')
z_lay = np.nanmean(np.stack(z_lay_stack), axis=0)
aer_lay = {name: np.nanmean(np.stack(aer_lay_stack[name]), axis=0)
           for name in aerosol_names}

# -----------------------------------------------------------------------
# Write output
OUT_DIR.mkdir(parents=True, exist_ok=True)
n_lay = z_lay.size

with nc.Dataset(OUT_FILE, 'w', format='NETCDF4') as f:
    f.n_composite_days = n_ok
    f.composite_day_range = '2003-2009 (dscu==1 from shcu_sgp_summer_97to09.nc)'
    f.phys_max_caps = '; '.join(f'{k}={v:.1e}' for k, v in PHYS_MAX.items())
    f.createDimension('lay', n_lay)

    v = f.createVariable('z_lay', 'f8', ('lay',))
    v[:] = z_lay
    v.units = 'm'
    v.long_name = 'Composite-mean CAMS layer height'

    for name in aerosol_names:
        v = f.createVariable(name, 'f8', ('lay',))
        v[:] = aer_lay[name]
        v.long_name = f'Composite-mean {name} mixing ratio on CAMS levels'

print(f'\nWrote {OUT_FILE}')
print(f'  lay = {n_lay}, n_composite_days = {n_ok}')

# Rough AOD sanity check (k_ext from ECMWF IFS docs, m²/kg at 550 nm).
k_ext_check = {
    'aermr07': 6300., 'aermr08': 6300., 'aermr11': 7500.,
    'aermr04':  570., 'aermr05':  300., 'aermr06':  100.,
}
Rd = 287.; p0 = 97300.; T_ref = 270.
idx_s = np.argsort(z_lay)
z_s = z_lay[idx_s]; dz_s = np.gradient(z_s)
rho_s = p0 * np.exp(-z_s / 8500.) / (Rd * T_ref)
aod_total = 0.
print('\n  Rough AOD check (550 nm):')
for name, k in k_ext_check.items():
    q = aer_lay[name][idx_s]
    aod = float(np.sum(q * rho_s * k * dz_s))
    print(f'    {name}: {aod:.4f}')
    aod_total += aod
print(f'    Total (6 species): {aod_total:.4f}  (ARM SGP AERONET Jul ~0.2–0.8)')

print(f'\nNext: symlink into shared/data/:')
print(f'  ln -sf {OUT_FILE} '
      '/global/homes/m/mpowell/repos/microhh/cases/cass/shared/data/cass_cams_composite.nc')
