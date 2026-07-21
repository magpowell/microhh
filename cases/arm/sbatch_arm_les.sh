#!/bin/bash
#SBATCH --qos=shared
#SBATCH --output=mhh-%j.out
#SBATCH --error=mhh-%j.err
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=18:00:00
#SBATCH --job-name=ARM_MICROHH_LES
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH -A m1266

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <radiation (standard|rt)>"
    exit 1
fi

radiation=$1
sim_dir=$SCRATCH/ARM_LES/$radiation

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

cd $sim_dir
rm *00*
rm *.txt
rm arm.out

# Print start time and location
echo "Starting ARM LES simulation at $(date)"
echo "Running in directory: $(pwd)"

./microhh init arm
./microhh run arm
