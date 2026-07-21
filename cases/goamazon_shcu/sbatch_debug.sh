#!/bin/bash
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=/pscratch/sd/m/mpowell/GOAMAZON_SHCU_LES/logs/debug-%j.out
#SBATCH --error=/pscratch/sd/m/mpowell/GOAMAZON_SHCU_LES/logs/debug-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266
#
# Same pattern as cases/cass/shared/sbatch_debug.sh.
# --constraint and --job-name passed at submit time.
# SIM_DIR: single run directory, passed via --export.

mkdir -p /pscratch/sd/m/mpowell/GOAMAZON_SHCU_LES/logs

cd "$SIM_DIR"
echo "Debug run in $(pwd) at $(date)"
rm -f goamazon_shcu.default.*.nc goamazon_shcu.column.*.nc goamazon_shcu.out
rm -f *.[0-9][0-9][0-9][0-9][0-9][0-9][0-9]
rm -f *.xy.* *_kernel.txt
./microhh init goamazon_shcu
./microhh run goamazon_shcu
echo "Done at $(date)"
