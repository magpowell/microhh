#!/bin/bash
#SBATCH --qos=shared
#SBATCH --output=mhh-%j.out
#SBATCH --error=mhh-%j.err
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=CUMULUS_MICROHH_LES
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH -A m1266

# Change to the case directory (passed via --export)
cd $SIM_DIR

rm *00*
rm *.txt
rm cabauw.out

# Print start time and location
echo "Starting LES simulation at $(date)"
echo "Running in directory: $(pwd)"

./microhh init cabauw
./microhh run cabauw
