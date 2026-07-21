#!/bin/bash
# Submit CASS rs_scale experiment: 5 values x 2 RT = 10 jobs, 4 reps each.
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
EXP="$SCRATCH/CASS_LES/experiments/rs_scale"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRATCH/CASS_LES/logs"

for val_dir in rs_0p25 rs_0p5 rs_2 rs_4; do
    # 2stream
    SIM_DIRS_LIST=()
    for rep in 01 02 03 04; do
        SIM_DIRS_LIST+=("$EXP/$val_dir/2stream/rep_$rep")
    done
    SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

    sbatch \
        --constraint=gpu \
        --time=10:00:00 \
        --job-name="rs_${val_dir}_2s" \
        --export=ALL,SIM_DIRS="$SIM_DIRS" \
        "$SCRIPT_DIR/sbatch_rs_scale.sh"
    echo "Submitted $val_dir 2stream"

    # raytracer
    SIM_DIRS_LIST=()
    for rep in 01 02 03 04; do
        SIM_DIRS_LIST+=("$EXP/$val_dir/raytracer/rep_$rep")
    done
    SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

    sbatch \
        --constraint="gpu&hbm80g" \
        --time=22:00:00 \
        --job-name="rs_${val_dir}_rt" \
        --export=ALL,SIM_DIRS="$SIM_DIRS" \
        "$SCRIPT_DIR/sbatch_rs_scale.sh"
    echo "Submitted $val_dir raytracer"
done

echo ""
echo "rs_scale=1.0 skipped (reuse no_aerosols_zero_wind data)."
echo "Submitted 8 jobs (4 values x 2 RT)."
