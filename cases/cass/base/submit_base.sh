#!/bin/bash
# Submit CASS base runs: 2 jobs (2stream + raytracer), 4 reps each on 1 node.
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
BASE="$SCRATCH/CASS_LES/base"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRATCH/CASS_LES/logs"

# --- 2stream: 4 reps, standard gpu constraint ---
SIM_DIRS_LIST=()
for rep in 01 02 03 04; do
    SIM_DIRS_LIST+=("$BASE/2stream/rep_$rep")
done
SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

sbatch \
    --constraint=gpu \
    --time=10:00:00 \
    --export=ALL,SIM_DIRS="$SIM_DIRS" \
    "$SCRIPT_DIR/sbatch_base.sh"
echo "Submitted 2stream base job"

# --- raytracer: 4 reps, hbm80g constraint ---
SIM_DIRS_LIST=()
for rep in 01 02 03 04; do
    SIM_DIRS_LIST+=("$BASE/raytracer/rep_$rep")
done
SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

sbatch \
    --constraint="gpu&hbm80g" \
    --time=22:00:00 \
    --export=ALL,SIM_DIRS="$SIM_DIRS" \
    "$SCRIPT_DIR/sbatch_base.sh"
echo "Submitted raytracer base job"
