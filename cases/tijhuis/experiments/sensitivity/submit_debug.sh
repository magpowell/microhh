#!/bin/bash
# Debug run: 4 seed cases on 4 GPUs to test parallel srun logic.

BASE=/pscratch/sd/m/mpowell/SENSITIVITY_LES_CUMULUS
SBATCH_SCRIPT=$BASE/sbatch_sensitivity.sh

echo "=== Debug: 4 seed cases in parallel on 1 node ==="
sbatch --qos=debug --constraint=gpu --time=00:10:00 \
    --ntasks-per-node=4 --gres=gpu:4 \
    --export=ALL,SIM_DIRS="$BASE/seed_experiment/standard/seed_1:$BASE/seed_experiment/standard/seed_2:$BASE/seed_experiment/standard/seed_3:$BASE/seed_experiment/standard/seed_4" \
    --job-name="DEBUG_parallel" \
    "$SBATCH_SCRIPT"
