#!/bin/bash
# Set up and submit debug runs for cs_veg experiment (cs_veg=42000, rep_01, 64x64 grid).
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_DEBUG="/global/homes/m/mpowell/repos/microhh/cases/cass/shared/sbatch_debug.sh"

mkdir -p "$SCRATCH/CASS_LES/logs"

echo "Setting up debug cs_veg runs (cs_veg=42000)..."
python "$SCRIPT_DIR/setup_cs_veg.py" --debug --values 42000

# 2stream
SIM_DIR="$SCRATCH/CASS_LES/debug/cs_veg/cs_veg_42000/2stream/rep_01"
sbatch \
    --constraint=gpu \
    --job-name=dbg_csv42k_2s \
    --export=ALL,SIM_DIR="$SIM_DIR" \
    "$SBATCH_DEBUG"
echo "Submitted debug cs_veg=42000 2stream"

# raytracer
SIM_DIR="$SCRATCH/CASS_LES/debug/cs_veg/cs_veg_42000/raytracer/rep_01"
sbatch \
    --constraint=gpu \
    --job-name=dbg_csv42k_rt \
    --export=ALL,SIM_DIR="$SIM_DIR" \
    "$SBATCH_DEBUG"
echo "Submitted debug cs_veg=0 raytracer"
