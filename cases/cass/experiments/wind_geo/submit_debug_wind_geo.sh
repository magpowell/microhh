#!/bin/bash
# Set up and submit debug runs for wind_geo experiment.
# Usage: submit_debug_wind_geo.sh [u_value ...]
# Default: all five values including u=0 (0.0 2.5 5.0 7.5 10.0)
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_DEBUG="/global/homes/m/mpowell/repos/microhh/cases/cass/shared/sbatch_debug.sh"

if [ $# -gt 0 ]; then
    VALUES=("$@")
else
    VALUES=(2.5 5.0 7.5 10.0)
fi

mkdir -p "$SCRATCH/CASS_LES/logs"

echo "Setting up debug wind_geo runs for u: ${VALUES[*]} ..."
python "$SCRIPT_DIR/setup_wind_geo.py" --debug --values "${VALUES[@]}"

for U in "${VALUES[@]}"; do
    LABEL="u_$(echo "$U" | sed 's/\./p/')"

    # 2stream
    SIM_DIR="$SCRATCH/CASS_LES/debug/wind_geo/$LABEL/2stream/rep_01"
    sbatch \
        --constraint=gpu \
        --job-name="dbg_wg_${LABEL}_2s" \
        --export=ALL,SIM_DIR="$SIM_DIR" \
        "$SBATCH_DEBUG"
    echo "Submitted debug wind_geo u=$U 2stream"

    # raytracer
    SIM_DIR="$SCRATCH/CASS_LES/debug/wind_geo/$LABEL/raytracer/rep_01"
    sbatch \
        --constraint="gpu&hbm80g" \
        --job-name="dbg_wg_${LABEL}_rt" \
        --export=ALL,SIM_DIR="$SIM_DIR" \
        "$SBATCH_DEBUG"
    echo "Submitted debug wind_geo u=$U raytracer"
done
