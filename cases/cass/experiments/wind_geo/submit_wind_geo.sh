#!/bin/bash
# Submit CASS wind_geo experiment runs.
# For each u value: 2 jobs (2stream + raytracer), 4 reps each on 1 node.
# u=0.0 excluded (redundant with no_aerosols_zero_wind).
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
EXP="$SCRATCH/CASS_LES/experiments/wind_geo"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRATCH/CASS_LES/logs"

# u values and their labels — must match setup_wind_geo.py u_label()
# e.g. 0.0 -> u_0p0,  2.5 -> u_2p5,  10.0 -> u_10p0
U_VALUES=(2.5 5.0 7.5 10.0)

for u in "${U_VALUES[@]}"; do
    label="u_$(echo "$u" | sed 's/\./p/')"
    VAL_DIR="$EXP/$label"

    # 2stream
    SIM_DIRS_LIST=()
    for rep in 01 02 03 04; do
        SIM_DIRS_LIST+=("$VAL_DIR/2stream/rep_$rep")
    done
    SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

    sbatch \
        --constraint=gpu \
        --time=10:00:00 \
        --job-name="wg${label}_2s" \
        --export=ALL,SIM_DIRS="$SIM_DIRS" \
        "$SCRIPT_DIR/sbatch_wind_geo.sh"
    echo "Submitted wind_geo u=${u} m/s 2stream"

    # raytracer
    SIM_DIRS_LIST=()
    for rep in 01 02 03 04; do
        SIM_DIRS_LIST+=("$VAL_DIR/raytracer/rep_$rep")
    done
    SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

    sbatch \
        --constraint="gpu&hbm80g" \
        --time=22:00:00 \
        --job-name="wg${label}_rt" \
        --export=ALL,SIM_DIRS="$SIM_DIRS" \
        "$SCRIPT_DIR/sbatch_wind_geo.sh"
    echo "Submitted wind_geo u=${u} m/s raytracer"
done
