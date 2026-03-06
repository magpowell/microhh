#!/bin/bash
#SBATCH --qos=shared
#SBATCH --output=mhh-%j.out
#SBATCH --error=mhh-%j.err
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=CASS_MICROHH_LES
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH -A m1266

# Change to the case directory (passed via --export)
cd $SIM_DIR
rm *00*
rm *.txt
rm cass.out
rm *.xy.nc q*.nc thl.nc T.nc w.nc

./download_cass_data.sh

module load conda
conda activate xr_env

# NOTE: cass_ls2d_input.py and download_cams.py are run as separate
# preprocessing steps before submitting this job.  They loop over all
# 119 composite days (ERA5) / 60 composite days (CAMS 2003-2009) and
# write the composite-averaged cass_ls2d_input.nc to the run directory.

python cass_input.py

# Print start time and location
echo "Starting CASS LES simulation at $(date)"
echo "Running in directory: $(pwd)"

./microhh init cass
./microhh run cass

python 3d_to_nc.py
python cross_to_nc.py
