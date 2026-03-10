#!/bin/bash
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=/pscratch/sd/m/mpowell/CASS_LES/logs/debug-%j.out
#SBATCH --error=/pscratch/sd/m/mpowell/CASS_LES/logs/debug-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266
#
# --constraint and --job-name are passed at submit time
# SIM_DIR: single run directory, passed via --export

mkdir -p /pscratch/sd/m/mpowell/CASS_LES/logs

cd "$SIM_DIR"
echo "Debug run in $(pwd) at $(date)"
echo "Clearing previous outputs..."
rm -f cass.default.*.nc cass.column.*.nc b.xy.nc cass.out
rm -f *.[0-9][0-9][0-9][0-9][0-9][0-9][0-9]
rm -f *.xy.* *_kernel.txt
./microhh init cass
./microhh run cass
echo "Done at $(date)"
