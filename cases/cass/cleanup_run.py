import os
import glob

to_rm = glob.glob('*00*')
to_rm += glob.glob('*.txt')
to_rm += glob.glob('*.nc')
to_rm += glob.glob('*.bin')
to_rm += glob.glob('*.err')
to_rm += glob.glob('cass.out')

exclude = glob.glob('cass_*.txt')
exclude += glob.glob('cass_input.nc')
exclude += glob.glob('cass_ls2d_input.nc')
exclude += glob.glob('van_*.nc')
exclude += glob.glob('cloud_coef*.nc')
exclude += glob.glob('coefficients_*.nc')
exclude += glob.glob('*.xy.nc')
to_rm = [f for f in to_rm if os.path.basename(f) not in set(exclude)]

for f in to_rm:
    if os.path.exists(f):
        os.remove(f)
