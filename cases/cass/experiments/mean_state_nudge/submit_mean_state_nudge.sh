#!/bin/bash
# Submit CASS mean_state_nudge experiment: 1 raytracer job, 4 reps.
#
# Usage:
#   ./submit_mean_state_nudge.sh --timescale 3600
#
# Requires: no_aerosols 2stream runs complete and nudge_profiles.nc extracted.
set -euo pipefail

TIMESCALE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --timescale) TIMESCALE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$TIMESCALE" ]]; then
    echo "Usage: $0 --timescale SECONDS"
    exit 1
fi

LABEL="nudge_${TIMESCALE}s"
SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
EXP="$SCRATCH/CASS_LES/experiments/mean_state_nudge/$LABEL"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRATCH/CASS_LES/logs"

SIM_DIRS_LIST=()
for rep in 01 02 03 04; do
    SIM_DIRS_LIST+=("$EXP/raytracer/rep_$rep")
done
SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

sbatch \
    --constraint="gpu&hbm80g" \
    --time=22:00:00 \
    --job-name="msn_rt_${TIMESCALE}" \
    --export=ALL,SIM_DIRS="$SIM_DIRS" \
    "$SCRIPT_DIR/sbatch_mean_state_nudge.sh"

echo "Submitted mean_state_nudge/$LABEL raytracer (4 reps)"
