#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <radiation (2stream|raytracer)>"
    exit 1
fi

radiation=$1

# Validate radiation flag
if [[ "$radiation" != "2stream" && "$radiation" != "raytracer" ]]; then
    echo "Error: <radiation> must be '2stream' or 'raytracer'"
    exit 1
fi

sim_dir=$SCRATCH/CASS_LES_$radiation

# Check that sim_dir exists
if [ ! -d "$sim_dir" ]; then
    echo "Error: Simulation directory does not exist: $sim_dir"
    exit 1
fi

# Check that sim_dir has files inside
if [ -z "$(ls -A "$sim_dir")" ]; then
    echo "Error: Simulation directory is empty: $sim_dir"
    exit 1
fi

# Set constraint and time based on radiation type
if [[ "$radiation" == "raytracer" ]]; then
    constraint="gpu&hbm80g"
    time="20:00:00"
else
    constraint="gpu"
    time="12:00:00"
fi

# Submit the job, passing sim_dir as an environment variable
sbatch --constraint="$constraint" --time="$time" --export=ALL,SIM_DIR="$sim_dir" sbatch_cass_les.sh

echo "Submitted job for CASS with ($radiation)"
