#!/bin/bash
# Submit CASS soil_moisture experiment runs.
# For each theta value: 2 jobs (2stream + raytracer), 4 reps each on 1 node.
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
EXP="$SCRATCH/CASS_LES/experiments/soil_moisture"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRATCH/CASS_LES/logs"

# theta labels must match setup_soil_moisture.py theta_label() — e.g. 0.1 -> theta_0p1
declare -A THETA_LABELS=(
    [0.1]="theta_0p1"
    [0.2]="theta_0p2"
    [0.3]="theta_0p3"
    [0.4]="theta_0p4"
)
THETA_VALUES=(0.1 0.2 0.3 0.4)

for theta in "${THETA_VALUES[@]}"; do
    label="${THETA_LABELS[$theta]}"
    VAL_DIR="$EXP/$label"

    # 2stream
    SIM_DIRS_LIST=()
    for rep in 01 02 03 04; do
        SIM_DIRS_LIST+=("$VAL_DIR/2stream/rep_$rep")
    done
    SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

    sbatch \
        --constraint=gpu \
        --time=18:00:00 \
        --job-name="sm${label}_2s" \
        --export=ALL,SIM_DIRS="$SIM_DIRS" \
        "$SCRIPT_DIR/sbatch_soil_moisture.sh"
    echo "Submitted theta=${theta} 2stream"

    # raytracer
    SIM_DIRS_LIST=()
    for rep in 01 02 03 04; do
        SIM_DIRS_LIST+=("$VAL_DIR/raytracer/rep_$rep")
    done
    SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

    sbatch \
        --constraint="gpu&hbm80g" \
        --time=28:00:00 \
        --job-name="sm${label}_rt" \
        --export=ALL,SIM_DIRS="$SIM_DIRS" \
        "$SCRIPT_DIR/sbatch_soil_moisture.sh"
    echo "Submitted theta=${theta} raytracer"
done
