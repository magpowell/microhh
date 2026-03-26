#!/bin/bash
# Submit CASS no_aerosols experiment runs: 2 jobs (2stream + raytracer), 4 reps each.
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
EXP="$SCRATCH/CASS_LES/experiments/no_aerosols"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRATCH/CASS_LES/logs"

# 2stream
SIM_DIRS_LIST=()
for rep in 01 02 03 04; do
    SIM_DIRS_LIST+=("$EXP/2stream/rep_$rep")
done
SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

sbatch \
    --constraint=gpu \
    --time=10:00:00 \
    --job-name=noaer_2s \
    --export=ALL,SIM_DIRS="$SIM_DIRS" \
    "$SCRIPT_DIR/sbatch_no_aerosols.sh"
echo "Submitted no_aerosols 2stream"

# raytracer
SIM_DIRS_LIST=()
for rep in 01 02 03 04; do
    SIM_DIRS_LIST+=("$EXP/raytracer/rep_$rep")
done
SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

sbatch \
    --constraint="gpu&hbm80g" \
    --time=22:00:00 \
    --job-name=noaer_rt \
    --export=ALL,SIM_DIRS="$SIM_DIRS" \
    "$SCRIPT_DIR/sbatch_no_aerosols.sh"
echo "Submitted no_aerosols raytracer"
