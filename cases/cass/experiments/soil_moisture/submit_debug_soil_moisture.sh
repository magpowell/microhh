#!/bin/bash
# Set up and submit debug runs for soil_moisture experiment.
# Usage: submit_debug_soil_moisture.sh [theta_value ...]
# Default: all four values (0.1 0.2 0.3 0.4)
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_DEBUG="/global/homes/m/mpowell/repos/microhh/cases/cass/shared/sbatch_debug.sh"

VALUES=("${@:-0.1 0.2 0.3 0.4}")
if [ $# -gt 0 ]; then
    VALUES=("$@")
else
    VALUES=(0.1 0.2 0.3 0.4)
fi

mkdir -p "$SCRATCH/CASS_LES/logs"

echo "Setting up debug soil_moisture runs for theta: ${VALUES[*]} ..."
python "$SCRIPT_DIR/setup_soil_moisture.py" --debug --values "${VALUES[@]}"

for THETA in "${VALUES[@]}"; do
    LABEL="theta_$(echo "$THETA" | sed 's/\./p/')"

    # 2stream
    SIM_DIR="$SCRATCH/CASS_LES/debug/soil_moisture/$LABEL/2stream/rep_01"
    sbatch \
        --constraint=gpu \
        --job-name="dbg_sm_${LABEL}_2s" \
        --export=ALL,SIM_DIR="$SIM_DIR" \
        "$SBATCH_DEBUG"
    echo "Submitted debug soil_moisture theta=$THETA 2stream"

    # raytracer
    SIM_DIR="$SCRATCH/CASS_LES/debug/soil_moisture/$LABEL/raytracer/rep_01"
    sbatch \
        --constraint=gpu \
        --job-name="dbg_sm_${LABEL}_rt" \
        --export=ALL,SIM_DIR="$SIM_DIR" \
        "$SBATCH_DEBUG"
    echo "Submitted debug soil_moisture theta=$THETA raytracer"
done
