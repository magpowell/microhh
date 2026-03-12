#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=cass_cr_comp
#SBATCH --output=/pscratch/sd/m/mpowell/CASS_LES/logs/cr_composite-%j.out
#SBATCH --error=/pscratch/sd/m/mpowell/CASS_LES/logs/cr_composite-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266
#
# Run cloud-root composite prep for one (expt, rt, rep).
# Passed via --export:
#   RUN_DIR    — full path to rep run directory (must have 3d_to_nc output)
#   OUTPUT_DIR — full path to output directory for events_xz.nc / events_yz.nc
#
# Prerequisites:
#   1. Run simulation to completion.
#   2. Run 3d_to_nc.py in the run directory:
#        cd $RUN_DIR && python 3d_to_nc.py -v thl qt ql w
#      (see submit_3d_to_nc.sh)
#
# Example manual submit:
#   sbatch --export=ALL,RUN_DIR=.../base/2stream/rep_01,OUTPUT_DIR=.../composite/base/2stream/rep_01 \
#       cloud_root_composite_prep.sbatch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=/pscratch/sd/m/mpowell/CASS_LES/logs
mkdir -p "$LOG_DIR"

echo "===== cloud_root_composite_prep ====="
echo "RUN_DIR    : ${RUN_DIR}"
echo "OUTPUT_DIR : ${OUTPUT_DIR}"
echo "Start      : $(date)"
echo ""

module load conda
conda activate xr_env

export OMP_NUM_THREADS=4   # scipy may use OpenMP for interpolation

python "${SCRIPT_DIR}/cloud_root_composite_prep.py" \
    --run-dir    "${RUN_DIR}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "Finished : $(date)"
