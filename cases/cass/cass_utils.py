"""
Utility functions for CASS composite case setup.
"""

from datetime import datetime
import numpy as np
import xarray as xr
from pathlib import Path

# Absolute path to the composite index file
COMPOSITE_NC = Path('/global/homes/m/mpowell/repos/microhh/cases/cass/shcu_sgp_summer_97to09.nc')


def read_composite_days(year_min=None, year_max=None, nc_path=None):
    """
    Read shallow cumulus composite days from shcu_sgp_summer_97to09.nc.

    Selects all (year, month, day) combinations where dscu == 1.
    Invalid calendar dates (e.g. June 31) are silently skipped.

    Parameters
    ----------
    year_min : int, optional
        Earliest year to include (inclusive).
    year_max : int, optional
        Latest year to include (inclusive).
    nc_path : str or Path, optional
        Override path to the composite NetCDF file.

    Returns
    -------
    list of datetime
        Sorted list of composite days (time set to 00:00 UTC).
    """
    if nc_path is None:
        nc_path = COMPOSITE_NC

    ds = xr.open_dataset(nc_path)
    dscu = ds['dscu']

    days = []
    for year in ds.year.values:
        year_int = int(round(float(year)))
        if year_min is not None and year_int < year_min:
            continue
        if year_max is not None and year_int > year_max:
            continue
        for mon in ds.mon.values:
            mon_int = int(round(float(mon)))
            for day in ds.day.values:
                day_int = int(round(float(day)))
                val = dscu.sel(year=year, mon=mon, day=day).item()
                if val is not None and not np.isnan(val) and val > 0:
                    try:
                        days.append(datetime(year_int, mon_int, day_int))
                    except ValueError:
                        pass  # invalid calendar date

    ds.close()
    days.sort()

    label = ''
    if year_min is not None or year_max is not None:
        lo = year_min if year_min is not None else '?'
        hi = year_max if year_max is not None else '?'
        label = f' ({lo}-{hi})'
    print(f'Found {len(days)} composite days{label}')
    return days
