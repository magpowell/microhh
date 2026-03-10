#!/bin/bash
#SBATCH --job-name=cass_era5
#SBATCH --output=era5_download_%j.log
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -A m1266
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=ALL

module load conda
conda activate xr_env

cd "${SIM_DIR:-$PWD}"

# Loop: each call to cass_ls2d_input.py processes all locally cached ERA5
# days, then either submits the next CDS request and exits, or writes
# cass_ls2d_input.nc when all 119 composite days are done.
# We keep looping (with a 60s sleep between retries) until the output exists.

echo "Starting ERA5 composite download/processing loop at $(date)"

until [ -f cass_ls2d_input.nc ]; do
    python cass_ls2d_input.py
    if [ ! -f cass_ls2d_input.nc ]; then
        echo "CDS request pending — sleeping ~10 mins before retry ($(date))"
        sleep 800
    fi
done

echo "Done: cass_ls2d_input.nc written at $(date)"
