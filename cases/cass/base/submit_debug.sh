#!/bin/bash
# Debug run: small grid test on debug QOS.

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <radiation (2stream|raytracer)>"
    exit 1
fi

radiation=$1

if [[ "$radiation" != "2stream" && "$radiation" != "raytracer" ]]; then
    echo "Error: <radiation> must be '2stream' or 'raytracer'"
    exit 1
fi

sim_dir=$SCRATCH/CASS_LES_$radiation

if [ ! -d "$sim_dir" ]; then
    echo "Error: Simulation directory does not exist: $sim_dir"
    exit 1
fi

echo "=== Debug: CASS LES ($radiation) on debug QOS ==="
sbatch --qos=debug --constraint=gpu --time=00:30:00 \
    --export=ALL,SIM_DIR="$sim_dir" \
    sbatch_cass_les.sh

echo "Submitted debug job for CASS with ($radiation)"
