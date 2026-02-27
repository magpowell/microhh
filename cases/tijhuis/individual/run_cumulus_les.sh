#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <sim_date> <radiation (standard|rt)>"
    exit 1
fi

sim_date=$1
radiation=$2

# Validate radiation flag
if [[ "$radiation" != "standard" && "$radiation" != "rt" ]]; then
    echo "Error: <radiation> must be 'standard' or 'rt'"
    exit 1
fi

sim_dir=$SCRATCH/inputfiles_cumulus_cases_2/$sim_date/$radiation

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
if [[ "$radiation" == "rt" ]]; then
    constraint="gpu&hbm80g"
    time="15:00:00"
else
    constraint="gpu"
    time="10:00:00"
fi

# Submit the job, passing sim_dir as an environment variable
sbatch --constraint="$constraint" --time="$time" --export=ALL,SIM_DIR="$sim_dir" sbatch_cumulus_les.sh

echo "Submitted job for $sim_date ($radiation)"
