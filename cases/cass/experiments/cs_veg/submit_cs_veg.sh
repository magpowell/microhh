#!/bin/bash
# Submit CASS cs_veg experiment runs.
# For each cs_veg value: 2 jobs (2stream + raytracer), 4 reps each on 1 node.
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
EXP="$SCRATCH/CASS_LES/experiments/cs_veg"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRATCH/CASS_LES/logs"

CS_VEG_VALUES=(0 41840 418400 4184000 41840000)

for val in "${CS_VEG_VALUES[@]}"; do
    VAL_DIR="$EXP/cs_veg_${val}"

    # 2stream
    SIM_DIRS_LIST=()
    for rep in 01 02 03 04; do
        SIM_DIRS_LIST+=("$VAL_DIR/2stream/rep_$rep")
    done
    SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

    sbatch \
        --constraint=gpu \
        --time=18:00:00 \
        --job-name="csv${val}_2s" \
        --export=ALL,SIM_DIRS="$SIM_DIRS" \
        "$SCRIPT_DIR/sbatch_cs_veg.sh"
    echo "Submitted cs_veg=${val} 2stream"

    # raytracer
    SIM_DIRS_LIST=()
    for rep in 01 02 03 04; do
        SIM_DIRS_LIST+=("$VAL_DIR/raytracer/rep_$rep")
    done
    SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

    sbatch \
        --constraint="gpu&hbm80g" \
        --time=28:00:00 \
        --job-name="csv${val}_rt" \
        --export=ALL,SIM_DIRS="$SIM_DIRS" \
        "$SCRIPT_DIR/sbatch_cs_veg.sh"
    echo "Submitted cs_veg=${val} raytracer"
done
