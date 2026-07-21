#!/bin/bash
# Submit the raytracer grid fit test: 4 grid candidates on one hbm80g debug
# node (one GPU each), endtime=1800 s, GPU memory logged to gpu_mem.log.
#
# Afterwards check, per dir:
#   max memory:  sort -t, -k2 -h fit_test/*/raytracer/gpu_mem.log | tail -1
#   completion:  grep -c . fit_test/*/raytracer/goamazon.out
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
FIT="$SCRATCH/GOAMAZON_LES/fit_test"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRATCH/GOAMAZON_LES/logs"

SIM_DIRS_LIST=()
for grid in dx200_k256 dz80_k320 dx200_i640 dx375_k256; do
    SIM_DIRS_LIST+=("$FIT/$grid/raytracer")
done
SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

sbatch \
    --qos=debug \
    --constraint="gpu&hbm80g" \
    --time=00:30:00 \
    --job-name=goam_fit \
    --export=ALL,SIM_DIRS="$SIM_DIRS",GPU_MEM_LOG=1 \
    "$SCRIPT_DIR/sbatch_runs.sh"
echo "Submitted goamazon fit test (4 grids, hbm80g debug)"
