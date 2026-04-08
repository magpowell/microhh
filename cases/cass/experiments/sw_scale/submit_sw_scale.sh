#!/bin/bash
# Submit CASS sw_scale experiment runs: 1 job (raytracer only), 4 reps.
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
EXP="$SCRATCH/CASS_LES/experiments/sw_scale"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRATCH/CASS_LES/logs"

# raytracer only (2stream runs not needed: 1D reference is no_aerosols_zero_wind)
SIM_DIRS_LIST=()
for rep in 01 02 03 04; do
    SIM_DIRS_LIST+=("$EXP/raytracer/rep_$rep")
done
SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

sbatch \
    --constraint="gpu&hbm80g" \
    --time=22:00:00 \
    --job-name=sw_scale_rt \
    --export=ALL,SIM_DIRS="$SIM_DIRS" \
    "$SCRIPT_DIR/sbatch_sw_scale.sh"
echo "Submitted sw_scale raytracer"
