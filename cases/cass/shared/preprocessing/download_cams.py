import sys
sys.path.insert(0, '/global/homes/m/mpowell/repos/LS2D')

from datetime import datetime
import ls2d

from cass_utils import read_composite_days

# CAMS EAC4 variables to download (model levels + surface)
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
    'eac4_sfc': [
        'surface_pressure'],
}

base_settings = {
    'central_lat' : 36.5,
    'central_lon' : -97.5,
    'area_size'   : 2,           # download ±2° box, regrid to 0.25°
    'case_name'   : 'cass',
    'cams_path'   : '/pscratch/sd/m/mpowell/LS2D_CAMS',
    'cdsapirc'    : '/global/homes/m/mpowell/.cdsapirc',
    'write_log'   : False,
    'data_source' : 'CDS',
    'ntasks'      : 1,
}

# CAMS EAC4 is available from 2003 onwards
composite_days = read_composite_days(year_min=2003, year_max=2009)

n_ok      = 0
n_pending = 0

for dt in composite_days:
    settings = dict(base_settings)
    settings['start_date'] = datetime(year=dt.year, month=dt.month, day=dt.day, hour=10)
    settings['end_date']   = datetime(year=dt.year, month=dt.month, day=dt.day, hour=23)

    try:
        ls2d.download_cams(settings, variables=cams_vars, grid=0.25)
        n_ok += 1
        print(f'  Downloaded CAMS for {dt.strftime("%Y-%m-%d")} ({n_ok}/{len(composite_days)})')
    except SystemExit:
        # LS2D submits an ADS request and calls sys.exit() when it is pending.
        # Exit here too — submitting more requests at once causes ADS to reject
        # them.  Re-run this script once the pending request completes.
        n_pending += 1
        print(f'\nCAMS download: {n_ok} complete, 1 request just submitted.')
        print('Re-run this script once the ADS request completes.')
        import sys; sys.exit(0)
    except Exception as e:
        print(f'  Warning: CAMS download failed for {dt.strftime("%Y-%m-%d")}: {e}')

print(f'\nCAMS download complete: {n_ok}/{len(composite_days)} days')
